/*
 * DS4 chat demo server (zero external deps).
 *
 *  - serves the chat UI (index.html) on http://127.0.0.1:PORT
 *  - proxies POST /v1/chat/completions to the ds4-server backend (SSE streamed
 *    straight through) so the browser stays single-origin
 *  - GET /api/metrics returns live inference-host stats:
 *      phys_mb / phys_peak_mb   process RAM of ds4-server (footprint -p PID)
 *      disk_mbps                disk0 throughput now (iostat, read+write; under
 *                               SSD streaming this is read-dominated)
 *      disk_cum_gb              cumulative disk0 bytes since this server started
 *      server_up                whether a ds4-server process was found
 *
 * Usage: PORT=3000 BACKEND=http://127.0.0.1:8000 node webdemo/server.js
 */
'use strict';
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');

const PORT = parseInt(process.env.PORT || '3000', 10);
const BACKEND = process.env.BACKEND || 'http://127.0.0.1:8000';
const DISK = process.env.DISK || 'disk0';
const backendUrl = new URL(BACKEND);

// ---- live host metrics -----------------------------------------------------
const metrics = {
  phys_mb: null, phys_peak_mb: null,
  disk_mbps: 0, disk_cum_gb: 0,
  server_up: false, pid: null, ts: Date.now(),
};

// Find the ds4-server PID (cached, refreshed lazily).
let pidCacheAt = 0;
function refreshPid() {
  const now = Date.now();
  if (metrics.pid && now - pidCacheAt < 4000) return;
  pidCacheAt = now;
  execFile('pgrep', ['-f', 'ds4-server'], (err, out) => {
    const pid = !err && out.trim() ? parseInt(out.trim().split(/\s+/)[0], 10) : null;
    metrics.pid = pid || null;
    metrics.server_up = !!pid;
    if (!pid) { metrics.phys_mb = null; metrics.phys_peak_mb = null; }
  });
}

function sampleFootprint() {
  refreshPid();
  if (!metrics.pid) return;
  execFile('footprint', ['-p', String(metrics.pid)], (err, out) => {
    if (err || !out) return;
    const cur = out.match(/phys_footprint:\s*([\d.]+)\s*(KB|MB|GB)/);
    const peak = out.match(/phys_footprint_peak:\s*([\d.]+)\s*(KB|MB|GB)/);
    const toMb = (m) => m ? (parseFloat(m[1]) * (m[2] === 'GB' ? 1024 : m[2] === 'KB' ? 1 / 1024 : 1)) : null;
    if (cur) metrics.phys_mb = toMb(cur);
    if (peak) metrics.phys_peak_mb = toMb(peak);
  });
}

// Long-running iostat: one line per second with disk0 MB/s; integrate cumulative.
function startIostat() {
  const io = spawn('iostat', ['-d', '-w', '1', DISK]);
  let lastTs = Date.now();
  io.stdout.on('data', (buf) => {
    const lines = buf.toString().split('\n');
    for (const line of lines) {
      const cols = line.trim().split(/\s+/);
      // disk0 -d columns: KB/t  tps  MB/s   (numeric data rows only)
      if (cols.length >= 3 && /^[\d.]+$/.test(cols[cols.length - 1]) && /^[\d.]+$/.test(cols[0])) {
        const mbps = parseFloat(cols[cols.length - 1]);
        if (!Number.isNaN(mbps)) {
          const now = Date.now();
          const dt = Math.min(5, (now - lastTs) / 1000) || 1;
          lastTs = now;
          metrics.disk_mbps = mbps;
          metrics.disk_cum_gb += (mbps * dt) / 1024;
          metrics.ts = now;
        }
      }
    }
  });
  io.on('exit', () => setTimeout(startIostat, 1000)); // resilient restart
}

setInterval(sampleFootprint, 1500);
refreshPid();
startIostat();

// ---- http server -----------------------------------------------------------
function serveStatic(res, file, type) {
  fs.readFile(path.join(__dirname, file), (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': type });
    res.end(data);
  });
}

function proxyChat(req, res) {
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', () => {
    const opts = {
      hostname: backendUrl.hostname,
      port: backendUrl.port || 80,
      path: '/v1/chat/completions',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    };
    const up = http.request(opts, (upRes) => {
      res.writeHead(upRes.statusCode, {
        'Content-Type': upRes.headers['content-type'] || 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      upRes.pipe(res);
    });
    up.on('error', (e) => {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'backend unreachable: ' + e.message + ' (is ds4-server running on ' + BACKEND + '?)' }));
    });
    up.end(body);
  });
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  if (req.method === 'POST' && u.pathname === '/v1/chat/completions') return proxyChat(req, res);
  if (req.method === 'GET' && u.pathname === '/api/metrics') {
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
    return res.end(JSON.stringify(metrics));
  }
  if (u.pathname === '/' || u.pathname === '/index.html') return serveStatic(res, 'index.html', 'text/html; charset=utf-8');
  res.writeHead(404); res.end('not found');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`DS4 chat demo on http://127.0.0.1:${PORT}  (backend ${BACKEND}, disk ${DISK})`);
});
