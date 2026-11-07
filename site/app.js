// Renders the model results from data.json as dependency-free SVG charts.
// Charts read colors from CSS variables, so they redraw on theme change instead of flipping.

const SVG = "http://www.w3.org/2000/svg";
const safeStorage = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } },
};
const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
const usdK = (v) => "$" + Math.round(v / 1000).toLocaleString("en-US") + "k";
const signed = (v, digits = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);
const signedUsd = (v) => (v >= 0 ? "+" : "−") + usd(Math.abs(v));
const pct = (v, digits = 1) => signed(v, digits) + "%";

const state = { growth: 0.4, model: "same_year" };
let DATA;

function h(tag, attrs = {}, parent, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

function s(tag, attrs = {}, parent, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

function linear([d0, d1], [r0, r1]) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw);
  const ticks = [];
  const top = Math.ceil(hi / step) * step;
  for (let v = Math.floor(lo / step) * step; v <= top + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return { ticks, lo: ticks[0], hi: ticks.at(-1) };
}

// Tooltip: built with textContent only. Rows are [color, name, value].
const tip = document.getElementById("tooltip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  h("p", { class: "tt-title" }, tip, title);
  for (const [color, name, value] of rows) {
    const row = h("div", { class: "tt-row" }, tip);
    if (color) h("span", { class: "key-line", style: `background:${color}` }, row);
    h("strong", {}, row, value);
    h("span", { class: "tt-name" }, row, name);
  }
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: ht } = window;
  const box = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + box.width > w - 8) x = evt.clientX - box.width - pad;
  if (y + box.height > ht - 8) y = evt.clientY - box.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.hidden = true);

function legend(parent, items) {
  const ul = h("ul", { class: "legend" }, parent);
  for (const it of items) {
    const li = h("li", {}, ul);
    h("span", { class: it.kind === "band" ? "key-band" : it.kind === "dot" ? "key-dot" : "key-line", style: `background:${it.color}` }, li);
    li.appendChild(document.createTextNode(it.name));
  }
}

/**
 * Line chart with optional bands, a reference line, end labels and a crosshair tooltip.
 * series: [{name, color, points: [{x, y}], band?: [{x, lo, hi}], endLabel?}]
 */
function lineChart(container, opts) {
  const { series, yFormat, xFormat = String, tooltip, ref, height = 320, yDomain, xTicks, legendItems } = opts;
  container.replaceChildren();
  if (opts.title) h("p", { class: "chart-title" }, container, opts.title);
  if (opts.subtitle) h("p", { class: "chart-sub" }, container, opts.subtitle);
  if (legendItems) legend(container, legendItems);

  const width = Math.max(container.clientWidth, 300);
  const m = { top: 12, right: opts.rightMargin ?? 64, bottom: 28, left: opts.leftMargin ?? 56 };
  const xs = series[0].points.map((p) => p.x);
  const allY = series.flatMap((se) => [...se.points.map((p) => p.y), ...(se.band || []).flatMap((b) => [b.lo, b.hi])]);
  if (ref) allY.push(ref.y);
  const yt = niceTicks(...(yDomain || [Math.min(...allY), Math.max(...allY)]), 5);
  const x = linear([Math.min(...xs), Math.max(...xs)], [m.left, width - m.right]);
  const y = linear([yt.lo, yt.hi], [height - m.bottom, m.top]);

  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.ariaLabel || opts.title || "" }, container);
  const gy = s("g", { class: "axis" }, svg);
  for (const t of yt.ticks) {
    s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }, gy);
    s("text", { x: m.left - 8, y: y(t), "text-anchor": "end", "dominant-baseline": "middle" }, gy, yFormat(t));
  }
  const gx = s("g", { class: "axis" }, svg);
  s("line", { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }, gx);
  for (const t of xTicks || xs) {
    s("text", { x: x(t), y: height - m.bottom + 18, "text-anchor": "middle" }, gx, xFormat(t));
  }

  if (ref) {
    s("line", { x1: m.left, x2: width - m.right, y1: y(ref.y), y2: y(ref.y), stroke: css("--text-muted"), "stroke-width": 1 }, svg);
    s("text", { class: "ref-label", x: width - m.right - 4, y: y(ref.y) + (ref.labelAbove ? -6 : 14), "text-anchor": "end" }, svg, ref.label);
  }

  for (const se of series) {
    if (!se.band) continue;
    const top = se.band.map((b) => `${x(b.x)},${y(b.hi)}`);
    const bottom = se.band.slice().reverse().map((b) => `${x(b.x)},${y(b.lo)}`);
    s("polygon", { points: [...top, ...bottom].join(" "), fill: se.color, opacity: 0.18 }, svg);
  }
  const labels = [];
  for (const se of series) {
    const d = se.points.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join("");
    s("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = se.points.at(-1);
    s("circle", { cx: x(last.x), cy: y(last.y), r: 4, fill: se.color, stroke: css("--surface-1"), "stroke-width": 2 }, svg);
    if (se.endLabel) labels.push({ y: y(last.y), text: se.endLabel, x: x(last.x) + 8 });
  }
  // Push end labels apart so they never overlap.
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 15);
  for (const l of labels) s("text", { class: "direct-label", x: l.x, y: l.y, "dominant-baseline": "middle" }, svg, l.text);

  const cross = s("line", { class: "crosshair", y1: m.top, y2: height - m.bottom, visibility: "hidden" }, svg);
  const hit = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent", tabindex: 0 }, svg);
  const snap = (px) => xs.reduce((best, v) => (Math.abs(x(v) - px) < Math.abs(x(best) - px) ? v : best), xs[0]);
  const move = (evt, xv) => {
    cross.setAttribute("x1", x(xv));
    cross.setAttribute("x2", x(xv));
    cross.setAttribute("visibility", "visible");
    showTip(evt, xFormat(xv), tooltip(xv));
  };
  hit.addEventListener("pointermove", (evt) => {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    move(evt, snap(pt.matrixTransform(svg.getScreenCTM().inverse()).x));
  });
  hit.addEventListener("pointerleave", () => (cross.setAttribute("visibility", "hidden"), hideTip()));
  let focusIdx = xs.length - 1;
  hit.addEventListener("keydown", (evt) => {
    if (!["ArrowLeft", "ArrowRight"].includes(evt.key)) return;
    focusIdx = Math.min(xs.length - 1, Math.max(0, focusIdx + (evt.key === "ArrowRight" ? 1 : -1)));
    const r = hit.getBoundingClientRect();
    move({ clientX: r.left + x(xs[focusIdx]) - m.left, clientY: r.top + 20 }, xs[focusIdx]);
    evt.preventDefault();
  });
  hit.addEventListener("blur", () => (cross.setAttribute("visibility", "hidden"), hideTip()));
}

function table(container, headers, rows, numeric = []) {
  container.replaceChildren();
  const wrap = h("div", { class: "table-scroll" }, container);
  const t = h("table", {}, wrap);
  const tr = h("tr", {}, h("thead", {}, t));
  headers.forEach((col, i) => h("th", numeric.includes(i) ? { class: "num" } : {}, tr, col));
  const body = h("tbody", {}, t);
  for (const r of rows) {
    const row = h("tr", {}, body);
    r.forEach((cell, i) => h("td", numeric.includes(i) ? { class: "num" } : {}, row, cell));
  }
}

// ---------- Sections ----------

function renderTiles() {
  const el = document.getElementById("tiles");
  el.replaceChildren();
  const { same_year: sy, lagged: lg } = DATA.elasticity;
  const pass = Object.values(DATA.passthrough).map((p) => p.coef);
  const tiles = [
    ["Long-run price change for 1% more people", `${sy.point.toFixed(1)}% same-year · ${lg.point.toFixed(1)}% lagged`, "Two timing assumptions, both associations, after inflation"],
    ["City prices follow the county", pass.map((v) => v.toFixed(2)).join(" / "), "Grass Valley / Nevada City change per 1% county change"],
    ...DATA.cities.map((c) => [`Typical ${c} home, ${DATA.base_year}`, usd(DATA.base_values[c]), "Zillow Home Value Index, yearly average"]),
  ];
  for (const [label, value, note] of tiles) {
    const t = h("div", { class: "tile" }, el);
    h("p", { class: "label" }, t, label);
    h("p", { class: "value" }, t, value);
    h("p", { class: "note" }, t, note);
  }
}

function scenarioRows(growth, model) {
  const g = Math.round(growth * 10) / 10;
  return Object.fromEntries(
    DATA.scenario_grid.filter((r) => r.growth === g && r.model === model).map((r) => [r.city, r]),
  );
}

function renderExplorer() {
  const rows = scenarioRows(state.growth, state.model);
  const flat = scenarioRows(0, state.model);
  const years = Array.from({ length: DATA.horizon + 1 }, (_, i) => DATA.base_year + i);
  const end = DATA.base_year + DATA.horizon;
  document.getElementById("growth-out").textContent = pct(state.growth) + "/yr";

  const popChange = 100 * ((1 + state.growth / 100) ** DATA.horizon - 1);
  const readouts = document.getElementById("readouts");
  readouts.replaceChildren();
  for (const city of DATA.cities) {
    const r = rows[city];
    const base = DATA.base_values[city];
    const med = r.median.at(-1);
    const card = h("div", { class: "readout" }, readouts);
    h("p", { class: "city" }, card, `${city} in ${end}`);
    h("p", { class: "big" }, card, usd(med));
    // Simulated values are rounded to $100, so the difference is too.
    h("p", { class: "delta" }, card, `${signedUsd(Math.round((med - base) / 100) * 100)} (${pct(100 * (med / base - 1))}) from population alone`);
    h("p", { class: "range" }, card, `90% range ${usd(r.p5.at(-1))} to ${usd(r.p95.at(-1))} · county population ${pct(popChange)} over ${DATA.horizon} years`);
  }

  const charts = document.getElementById("explorer-charts");
  charts.replaceChildren();
  // Both cities share one y-range so their charts compare directly.
  const vals = DATA.cities.flatMap((c) => [...rows[c].p5, ...rows[c].p95, DATA.base_values[c]]);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const color = css("--series-1");
  // Lay out every card before drawing, so each chart measures its final width.
  const divs = DATA.cities.map(() => h("div", { class: "chart" }, h("div", { class: "card" }, charts)));
  for (const [ci, city] of DATA.cities.entries()) {
    const r = rows[city];
    lineChart(divs[ci], {
      title: city,
      subtitle: `Typical home value, ${DATA.base_year} dollars`,
      legendItems: [
        { name: `${pct(state.growth)}/yr population`, color },
        { name: "90% range", color, kind: "band" },
      ],
      height: 300,
      yDomain: [lo, hi],
      yFormat: usdK,
      series: [{
        name: city, color,
        points: years.map((yr, i) => ({ x: yr, y: r.median[i] })),
        band: years.map((yr, i) => ({ x: yr, lo: r.p5[i], hi: r.p95[i] })),
        endLabel: usdK(r.median.at(-1)),
      }],
      ref: { y: DATA.base_values[city], label: "Flat population", labelAbove: state.growth < 0 },
      xTicks: years.filter((yr) => (yr - DATA.base_year) % 2 === 0),
      tooltip: (yr) => {
        const i = yr - DATA.base_year;
        return [
          [color, "middle estimate", usd(r.median[i])],
          [null, "90% range", `${usdK(r.p5[i])}–${usdK(r.p95[i])}`],
          [css("--text-muted"), "flat population", usd(flat[city].median[i])],
        ];
      },
    });
  }

  const presets = [-0.5, 0, 0.4, 1, 2, 3];
  table(
    document.getElementById("explorer-table"),
    ["Growth per year", ...DATA.cities.flatMap((c) => [`${c} ${end}`, "Change"])],
    presets.map((g) => {
      const r = scenarioRows(g, state.model);
      return [pct(g) + "/yr", ...DATA.cities.flatMap((c) => [usd(r[c].median.at(-1)), pct(100 * (r[c].median.at(-1) / DATA.base_values[c] - 1))])];
    }),
    [1, 2, 3, 4],
  );
}

function setupControls() {
  const slider = document.getElementById("growth");
  slider.addEventListener("input", () => {
    state.growth = +slider.value;
    syncPresets();
    renderExplorer();
  });

  const presetEl = document.getElementById("presets");
  const trendRate = Math.round(DATA.trend_growth * 1000) / 10;
  const presets = [["Decline", -0.5], ["Flat", 0], [`Trend ${pct(trendRate)}`, trendRate], ["Growth", 1], ["Fast", 2]];
  for (const [label, g] of presets) {
    const b = h("button", { type: "button", "aria-pressed": "false", "data-g": g }, presetEl, label);
    b.addEventListener("click", () => {
      state.growth = g;
      slider.value = g;
      syncPresets();
      renderExplorer();
    });
  }
  function syncPresets() {
    for (const b of presetEl.children) b.setAttribute("aria-pressed", String(+b.dataset.g === state.growth));
  }
  state.growth = trendRate;
  slider.value = trendRate;
  syncPresets();

  const modelEl = document.getElementById("model-choice");
  for (const [label, key] of [["Same-year", "same_year"], ["Lagged", "lagged"]]) {
    const b = h("button", { type: "button", role: "radio", "aria-checked": String(state.model === key), "data-m": key }, modelEl, label);
    b.addEventListener("click", () => {
      state.model = key;
      for (const c of modelEl.children) c.setAttribute("aria-checked", String(c.dataset.m === key));
      renderExplorer();
    });
  }
}

function renderHistory() {
  const hist = DATA.history;
  const base = hist[0];
  const names = { population: "Nevada County population", "Grass Valley": "Grass Valley home value", "Nevada City": "Nevada City home value" };
  const colors = { population: css("--series-1"), "Grass Valley": css("--series-2"), "Nevada City": css("--series-3") };
  const keys = Object.keys(names);
  const idx = (r, k) => (100 * r[k]) / base[k];
  lineChart(document.getElementById("history-chart"), {
    title: `Indexed to ${base.year} = 100, home values after inflation`,
    legendItems: keys.map((k) => ({ name: names[k], color: colors[k] })),
    height: 360,
    rightMargin: 150,
    yFormat: (v) => v.toFixed(0),
    series: keys.map((k) => ({
      name: names[k], color: colors[k],
      points: hist.map((r) => ({ x: r.year, y: idx(r, k) })),
      endLabel: names[k].replace(" home value", ""),
    })),
    xTicks: hist.map((r) => r.year).filter((yr) => yr % 4 === 1),
    tooltip: (yr) => {
      const r = hist.find((d) => d.year === yr);
      return keys.map((k) => [colors[k], names[k], k === "population" ? r[k].toLocaleString("en-US") : usd(r[k])]);
    },
  });
  table(
    document.getElementById("history-table"),
    ["Year", "County population", "Grass Valley value", "Nevada City value"],
    hist.map((r) => [r.year, r.population.toLocaleString("en-US"), usd(r["Grass Valley"]), usd(r["Nevada City"])]),
    [1, 2, 3],
  );
}

function renderCounties() {
  const { rows, start, end } = DATA.counties;
  const nev = rows.find((r) => r.fips === "06057");
  document.getElementById("counties-lede").textContent =
    `Each dot is a California county. It shows population change and home-price change after inflation from ${start} to ${end}. ` +
    `Nevada County grew ${pct(nev.pop_change)} and its prices rose ${pct(nev.price_change, 0)}. ` +
    `This simple comparison ignores income, building and timing. The model accounts for those, so its estimate is the better guide.`;

  const container = document.getElementById("counties-chart");
  container.replaceChildren();
  const highlight = css("--series-1");
  const muted = css("--text-muted");
  legend(container, [{ name: "Nevada County", color: highlight, kind: "dot" }, { name: "Other California counties", color: muted, kind: "dot" }, { name: "Simple straight-line fit", color: css("--text-secondary") }]);

  const width = Math.max(container.clientWidth, 300);
  const height = 400;
  const m = { top: 12, right: 24, bottom: 44, left: 56 };
  const xt = niceTicks(Math.min(...rows.map((r) => r.pop_change)), Math.max(...rows.map((r) => r.pop_change)), 6);
  const yt = niceTicks(Math.min(...rows.map((r) => r.price_change)), Math.max(...rows.map((r) => r.price_change)), 5);
  const x = linear([xt.lo, xt.hi], [m.left, width - m.right]);
  const y = linear([yt.lo, yt.hi], [height - m.bottom, m.top]);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Scatter plot of county population change against real home-price change" }, container);
  const g = s("g", { class: "axis" }, svg);
  for (const t of yt.ticks) {
    s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }, g);
    s("text", { x: m.left - 8, y: y(t), "text-anchor": "end", "dominant-baseline": "middle" }, g, t + "%");
  }
  for (const t of xt.ticks) s("text", { x: x(t), y: height - m.bottom + 18, "text-anchor": "middle" }, g, t + "%");
  s("text", { x: (m.left + width - m.right) / 2, y: height - 6, "text-anchor": "middle" }, g, `Population change, ${start}–${end}`);
  s("text", { x: 12, y: (m.top + height - m.bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.top + height - m.bottom) / 2})` }, g, "Real home-price change");

  // Ordinary least squares fit across counties, for orientation only.
  const n = rows.length;
  const mx = rows.reduce((a, r) => a + r.pop_change, 0) / n;
  const my = rows.reduce((a, r) => a + r.price_change, 0) / n;
  const slope = rows.reduce((a, r) => a + (r.pop_change - mx) * (r.price_change - my), 0) / rows.reduce((a, r) => a + (r.pop_change - mx) ** 2, 0);
  const fx = [xt.lo, xt.hi].map((v) => Math.min(Math.max(v, mx - 60), mx + 60));
  s("line", { x1: x(fx[0]), y1: y(my + slope * (fx[0] - mx)), x2: x(fx[1]), y2: y(my + slope * (fx[1] - mx)), stroke: css("--text-secondary"), "stroke-width": 1.5 }, svg);

  const others = rows.filter((r) => r !== nev);
  for (const r of others) s("circle", { cx: x(r.pop_change), cy: y(r.price_change), r: 5, fill: muted, opacity: 0.75, stroke: css("--surface-1"), "stroke-width": 2 }, svg);
  s("circle", { cx: x(nev.pop_change), cy: y(nev.price_change), r: 7, fill: highlight, stroke: css("--surface-1"), "stroke-width": 2 }, svg);
  s("text", { class: "direct-label", x: x(nev.pop_change) + 11, y: y(nev.price_change), "dominant-baseline": "middle", "font-weight": 600 }, svg, "Nevada County");

  // Nearest-point hover so small dots are easy to hit.
  const hit = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" }, svg);
  const ring = s("circle", { r: 9, fill: "none", stroke: css("--text-primary"), "stroke-width": 1.5, visibility: "hidden" }, svg);
  hit.addEventListener("pointermove", (evt) => {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const best = rows.reduce((a, r) => ((x(r.pop_change) - p.x) ** 2 + (y(r.price_change) - p.y) ** 2 < (x(a.pop_change) - p.x) ** 2 + (y(a.price_change) - p.y) ** 2 ? r : a));
    ring.setAttribute("cx", x(best.pop_change));
    ring.setAttribute("cy", y(best.price_change));
    ring.setAttribute("visibility", "visible");
    showTip(evt, `${best.name} County`, [
      [null, "population change", pct(best.pop_change)],
      [null, "real price change", pct(best.price_change)],
      [null, `residents in ${end}`, best.population.toLocaleString("en-US")],
    ]);
  });
  hit.addEventListener("pointerleave", () => (ring.setAttribute("visibility", "hidden"), hideTip()));

  table(
    document.getElementById("counties-table"),
    ["County", "Population change", "Real price change", `Residents ${end}`],
    rows.slice().sort((a, b) => b.pop_change - a.pop_change).map((r) => [r.name, pct(r.pop_change), pct(r.price_change), r.population.toLocaleString("en-US")]),
    [1, 2, 3],
  );
}

function renderImpulse() {
  const yrs = Array.from({ length: DATA.horizon + 1 }, (_, i) => i);
  const spec = [["same_year", "Same-year estimate", css("--series-1")], ["lagged", "Lagged estimate", css("--series-2")]];
  lineChart(document.getElementById("impulse-chart"), {
    title: "Cumulative change in real home prices after a one-time 1% population increase",
    legendItems: spec.flatMap(([, name, color]) => [{ name, color }]).concat([{ name: "90% range", color: css("--text-muted"), kind: "band" }]),
    height: 320,
    yFormat: (v) => v.toFixed(1) + "%",
    xFormat: (v) => (v === 0 ? "Arrival" : `Year ${v}`),
    xTicks: yrs.filter((v) => v % 2 === 0),
    rightMargin: 64,
    series: spec.map(([k, name, color]) => {
      const r = DATA.impulse[k];
      return {
        name, color,
        points: yrs.map((i) => ({ x: i, y: r.median[i] })),
        band: yrs.map((i) => ({ x: i, lo: r.p5[i], hi: r.p95[i] })),
        endLabel: pct(r.median.at(-1)),
      };
    }),
    tooltip: (i) => spec.map(([k, name, color]) => [color, name, `${pct(DATA.impulse[k].median[i], 2)} (${DATA.impulse[k].p5[i].toFixed(1)} to ${DATA.impulse[k].p95[i].toFixed(1)})`]),
  });
  table(
    document.getElementById("impulse-table"),
    ["Years after the increase", "Same-year estimate", "Lagged estimate"],
    yrs.map((i) => [i, pct(DATA.impulse.same_year.median[i], 2), pct(DATA.impulse.lagged.median[i], 2)]),
    [1, 2],
  );
}

function renderMethod() {
  const p = DATA.panel;
  document.getElementById("method-panel").textContent =
    `The model compares yearly price growth after inflation with population growth in ${p.counties} California counties from ${p.first_year} to ${p.last_year} (${p.n.toLocaleString("en-US")} county-years). ` +
    "It also accounts for last year's price growth, income growth and new building permits.";
  const pass = Object.entries(DATA.passthrough).map(([c, v]) => `${c} ${v.coef.toFixed(2)} (90% range ${v.lo.toFixed(2)} to ${v.hi.toFixed(2)})`);
  document.getElementById("method-pass").textContent =
    `Each city moves almost one-for-one with the county: ${pass.join(", ")}. A value of 1.00 means a 1% county change brings a 1% city change.`;

  const el = document.getElementById("coef-tables");
  el.replaceChildren();
  for (const [key, title] of [["same_year", "Same-year estimate"], ["lagged", "Lagged estimate"]]) {
    const box = h("div", {}, el);
    h("h4", {}, box, title);
    const div = h("div", {}, box);
    table(div, ["Term", "Coefficient", "90% interval", "p-value"],
      DATA.coefficients[key].map((r) => [r.term, r.coef.toFixed(3), `${r.lo.toFixed(3)} to ${r.hi.toFixed(3)}`, r.p.toFixed(3)]), [1, 2, 3]);
    const e = DATA.elasticity[key];
    h("p", {}, box, `Long-run price change per 1% more people: ${e.point.toFixed(2)}% (90% interval ${e.lo.toFixed(2)} to ${e.hi.toFixed(2)}).`);
  }
}

function renderAll() {
  renderTiles();
  renderExplorer();
  renderHistory();
  renderCounties();
  renderImpulse();
  renderMethod();
}

function setupTheme() {
  const btn = document.getElementById("theme-toggle");
  const isDark = () => document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const label = () => (btn.textContent = isDark() ? "Light theme" : "Dark theme");
  label();
  btn.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    safeStorage.set("theme", next);
    label();
    renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => (label(), renderAll()));
}

DATA = await (await fetch("data.json")).json();
for (const node of document.querySelectorAll("[data-bind]")) node.textContent = DATA[node.dataset.bind];
setupControls();
setupTheme();
renderAll();
let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 150);
});
