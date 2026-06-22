// Render ssd_streaming_viz.html with the real A/B results injected, and save a PNG.
// Usage: node bench/render_viz.js
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const here = __dirname;
const htmlPath = 'file://' + path.join(here, 'ssd_streaming_viz.html');
const resultsPath = path.join(here, 'out', 'ab_results.json');

(async () => {
  let ab = null;
  try { ab = JSON.parse(fs.readFileSync(resultsPath, 'utf8')); } catch (e) {}

  const CHROME = process.env.CHROME_PATH ||
    '/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta';
  const browser = await puppeteer.launch({ headless: 'new', executablePath: CHROME });
  const page = await browser.newPage();
  await page.setViewport({ width: 1240, height: 1700, deviceScaleFactor: 2 });
  await page.goto(htmlPath, { waitUntil: 'networkidle0' });

  if (ab) {
    await page.evaluate((d) => {
      const set = (k, v) => { const el = document.querySelector(`[data-k="${k}"]`); if (el) el.textContent = v; };
      for (const k in d) set(k, d[k]);
    }, ab);
  }
  // warm the animation a touch so the page isn't empty-looking
  await page.evaluate(() => { document.querySelector('#play') && document.querySelector('#play').click(); });
  await new Promise(r => setTimeout(r, 1200));

  const out = path.join(here, 'out', 'ssd_streaming_viz.png');
  await page.screenshot({ path: out, fullPage: true });
  console.log('wrote', out, ab ? '(with A/B results)' : '(no ab_results.json yet)');
  await browser.close();
})();
