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

const state = { growth: 0, model: "peer", path: null, value: 0, dollars: "real", futures: false };
const FUTURES = 40;
const VALUE_RANGE = [50_000, 20_000_000];
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

// Tooltip: built with textContent only. Rows are [color, name, value], or [text] for a plain line.
const tip = document.getElementById("tooltip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  h("p", { class: "tt-title" }, tip, title);
  for (const [color, name, value] of rows) {
    if (name === undefined) {
      h("p", { class: "tt-note" }, tip, color);
      continue;
    }
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
 * markers: [{x, label, kind, rows}] dated events, drawn as dashed verticals with a tooltip of rows.
 * spaghetti: [[y, ...]] faint 1px lines in the first series color, one y per series[0] x.
 */
const MARKER_KINDS = {
  market: { name: "Market turn", color: "--text-muted" },
  fire: { name: "Wildfire", color: "--series-2" },
  insurance: { name: "Insurance event", color: "--series-4" },
};
const LABEL_ROW = 13;
const LABEL_ROWS_MAX = 5;
const LABEL_CHAR_PX = 6.2;
const clip = (text, max = 22) => (text.length > max ? text.slice(0, max - 1) + "…" : text);

// Greedy row assignment: a label takes the first row whose last label ends before it starts.
// A label that fits no row is dropped; its marker and tooltip stay.
function layoutMarkerLabels(markers, x) {
  const ends = [];
  return [...markers].sort((a, b) => a.x - b.x).map((mk) => {
    const left = x(mk.x) + 8;
    const right = left + clip(mk.label).length * LABEL_CHAR_PX;
    let row = ends.findIndex((e) => e < left);
    if (row < 0 && ends.length < LABEL_ROWS_MAX) row = ends.length;
    if (row >= 0) ends[row] = right + 6;
    return { ...mk, row };
  });
}
function lineChart(container, opts) {
  const { series, yFormat, xFormat = String, tooltip, ref, height: baseHeight = 320, yDomain, xTicks, legendItems, markers = [], spaghetti = [] } = opts;
  container.replaceChildren();
  if (opts.title) h("p", { class: "chart-title" }, container, opts.title);
  if (opts.subtitle) h("p", { class: "chart-sub" }, container, opts.subtitle);
  if (legendItems) legend(container, legendItems);
  if (markers.length) {
    const kinds = [...new Set(markers.map((mk) => mk.kind))];
    legend(container, kinds.map((k) => ({ name: MARKER_KINDS[k].name, color: css(MARKER_KINDS[k].color), kind: "dot" })));
  }

  const width = Math.max(container.clientWidth, 300);
  const m = { top: 12, right: opts.rightMargin ?? 64, bottom: 28, left: opts.leftMargin ?? 56 };
  const xs = series[0].points.map((p) => p.x);
  const allY = series.flatMap((se) => [...se.points.map((p) => p.y), ...(se.band || []).flatMap((b) => [b.lo, b.hi])]);
  allY.push(...spaghetti.flat());
  if (ref) allY.push(ref.y);
  const yt = niceTicks(...(yDomain || [Math.min(...allY), Math.max(...allY)]), 5);
  const x = linear([Math.min(...xs), Math.max(...xs)], [m.left, width - m.right]);
  const placed = layoutMarkerLabels(markers, x);
  const labelRows = Math.max(0, ...placed.map((mk) => mk.row + 1));
  m.top += labelRows * LABEL_ROW;
  const height = baseHeight + labelRows * LABEL_ROW;
  const y = linear([yt.lo, yt.hi], [height - m.bottom, m.top]);

  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.ariaLabel || opts.title || "" }, container);
  const gy = s("g", { class: "axis" }, svg);
  for (const t of yt.ticks) {
    s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }, gy);
    s("text", { x: m.left - 8, y: y(t), "text-anchor": "end", "dominant-baseline": "middle" }, gy, yFormat(t));
  }
  const gx = s("g", { class: "axis" }, svg);
  s("line", { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }, gx);
  // Labels closer than 40px overlap at phone width, so later ones give way.
  let lastTick = -Infinity;
  for (const t of xTicks || xs) {
    if (x(t) - lastTick < 40) continue;
    lastTick = x(t);
    s("text", { x: x(t), y: height - m.bottom + 18, "text-anchor": "middle" }, gx, xFormat(t));
  }

  if (ref) {
    s("line", { x1: m.left, x2: width - m.right, y1: y(ref.y), y2: y(ref.y), stroke: css("--text-muted"), "stroke-width": 1 }, svg);
    s("text", { class: "ref-label", x: width - m.right - 4, y: y(ref.y) + (ref.labelAbove ? -6 : 14), "text-anchor": "end" }, svg, ref.label);
  }

  if (spaghetti.length) {
    const g = s("g", { class: "spaghetti" }, svg);
    for (const ys of spaghetti) {
      const d = ys.map((v, i) => `${i ? "L" : "M"}${x(xs[i])},${y(v)}`).join("");
      s("path", { d, fill: "none", stroke: series[0].color, "stroke-width": 1, opacity: 0.12 }, g);
    }
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

  // Hit strips cover only the circle and label rows, so the crosshair stays reachable inside the plot.
  for (const mk of placed) {
    const top = mk.row >= 0 ? 12 + mk.row * LABEL_ROW : m.top;
    const color = css(MARKER_KINDS[mk.kind].color);
    const g = s("g", { class: "marker" }, svg);
    s("line", { class: "marker-line", x1: x(mk.x), x2: x(mk.x), y1: top, y2: height - m.bottom }, g);
    s("circle", { cx: x(mk.x), cy: top, r: 3.5, fill: color }, g);
    if (mk.row >= 0) s("text", { class: "marker-label", x: x(mk.x) + 8, y: top, "dominant-baseline": "middle" }, g, clip(mk.label));
    s("rect", { x: x(mk.x) - 6, y: top - 6, width: 12, height: Math.max(m.top - top, 6) + 6, fill: "transparent" }, g);
    g.addEventListener("pointermove", (evt) => showTip(evt, xFormat(mk.x), mk.rows));
    g.addEventListener("pointerleave", hideTip);
  }
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
  const hl = DATA.headline;
  const lp = hl.lp10;
  const pass = Object.values(DATA.passthrough).map((p) => p.coef);
  const tiles = [
    [`Price change after ${DATA.horizon} years for 1% more people, peer counties`, pct(lp.point), `${hl.counties} counties under ${hl.max_pop.toLocaleString("en-US")} people. 90% range ${lp.lo.toFixed(1)}% to ${lp.hi.toFixed(1)}%. An association, after inflation`],
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

// The effect is linear in log1p(growth), so the exported 1%/yr draws rescale exactly to any constant rate.
// These draws feed only the possible-futures lines and count.
function unitValues(growth, model, city, n) {
  const scale = Math.log1p(growth / 100) / Math.log1p(0.01);
  const base = DATA.base_values[city];
  return DATA.unit_paths[model][city].slice(0, n).map((path) => [base, ...path.map((e) => base * Math.exp(e * scale))]);
}

// Quantiles of all 1,000 draws, rounded to $100 like the exported bands. A negative scale swaps the tails.
function unitRows(growth, model) {
  const scale = Math.log1p(growth / 100) / Math.log1p(0.01);
  const [lo, hi] = scale >= 0 ? ["p5", "p95"] : ["p95", "p5"];
  return Object.fromEntries(DATA.cities.map((city) => {
    const q = DATA.unit_quantiles[model][city];
    const base = DATA.base_values[city];
    const row = (k) => [base, ...q[k].map((e) => base * Math.exp(e * scale))].map((v) => Math.round(v / 100) * 100);
    return [city, { p5: row(lo), median: row("p50"), p95: row(hi) }];
  }));
}

// Two decimals only when the rate is off the slider's 0.1 grid, e.g. the Recent preset.
const rateText = (g) => pct(g, Math.abs(g * 10 - Math.round(g * 10)) < 1e-9 ? 1 : 2);

// A path preset (state.path) replaces the constant rate on the slider.
function activePath() {
  return state.path ? DATA.scenario_paths[state.path] : null;
}

function pathAverage(path) {
  return 100 * Math.expm1(path.growth.reduce((a, g) => a + Math.log1p(g), 0) / path.growth.length);
}

// Residents and households are rounded to the nearest 10: the inputs are not precise to the unit.
function renderPeople(path) {
  const rate = path ? 100 * path.growth[0] : state.growth;
  const subject = path ? `The state projection's first year, ${pct(rate)}/yr,` : `${rateText(rate)}/yr`;
  const round10 = (v) => Math.round(v / 10) * 10;
  const n = (v) => v.toLocaleString("en-US");
  const residents = round10(Math.abs((DATA.population_latest.count * rate) / 100));
  const permits = `The county permitted ${n(DATA.permits_latest.count)} homes in ${DATA.permits_latest.year}.`;
  const move = residents === 0
    ? "holds the population flat."
    : `is about ${n(residents)} ${rate < 0 ? "fewer" : "more"} residents a year, about ${n(round10(residents / DATA.persons_per_household))} households.`;
  document.getElementById("people-line").textContent = `${subject} ${move} ${permits}`;
}

function parseValue(text) {
  const digits = text.replace(/[^0-9]/g, "");
  const v = digits ? +digits : NaN;
  return v >= VALUE_RANGE[0] && v <= VALUE_RANGE[1] ? v : null;
}

// Hash keys: g growth %, m model, v home value, p preset key, d dollars. Invalid values are ignored.
function readHash(presets) {
  const q = new URLSearchParams(location.hash.slice(1));
  const slider = document.getElementById("growth");
  if (["peer", "all"].includes(q.get("m"))) state.model = q.get("m");
  if (q.get("d") === "nominal") state.dollars = "nominal";
  const key = q.get("p");
  if (key === "dof") state.path = "dof";
  else if (Object.hasOwn(presets, key)) state.growth = presets[key].growth;
  else if (/^-?\d+(\.\d+)?$/.test(q.get("g") ?? "")) {
    const g = Math.round(+q.get("g") * 10) / 10;
    if (g >= +slider.min && g <= +slider.max) state.growth = g;
  }
  if (/^\d+$/.test(q.get("v") ?? "")) state.value = parseValue(q.get("v")) ?? state.value;
}

function renderExplorer() {
  const path = activePath();
  const rows = path ? path[state.model] : unitRows(state.growth, state.model);
  // Unit draws rescale only a constant rate, so the state projection keeps its band.
  const futures = state.futures && !path
    ? Object.fromEntries(DATA.cities.map((c) => [c, unitValues(state.growth, state.model, c, FUTURES)]))
    : null;
  const growth = path ? pathAverage(path) : state.growth;
  const rateLabel = path ? "State projection" : `${rateText(growth)}/yr population`;
  const years = Array.from({ length: DATA.horizon + 1 }, (_, i) => DATA.base_year + i);
  const end = DATA.base_year + DATA.horizon;
  document.getElementById("growth-out").textContent = path ? "State projection" : rateText(state.growth) + "/yr";
  const outside = !path && (state.growth < DATA.support.lo || state.growth > DATA.support.hi);
  document.getElementById("support-note").textContent = outside
    ? "Few California counties sustained this rate for a decade. Treat this as a stretch, not a scenario."
    : "";

  renderPeople(path);

  const popChange = path
    ? 100 * (path.growth.reduce((a, g) => a * (1 + g), 1) - 1)
    : 100 * ((1 + state.growth / 100) ** DATA.horizon - 1);
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
    const own = (v) => usd((state.value * v) / base);
    h("p", { class: "home" }, card, `A ${usd(state.value)} home: ${own(med)} (range ${own(r.p5.at(-1))} to ${own(r.p95.at(-1))})`);
    if (futures) {
      const above = futures[city].filter((v) => v.at(-1) > base).length;
      h("p", { class: "range" }, card, state.growth === 0
        ? `At ${rateText(0)}/yr every simulated future equals today's value.`
        : `In ${above} of ${FUTURES} simulated futures the value is above today's.`);
    }
  }

  const charts = document.getElementById("explorer-charts");
  charts.replaceChildren();
  // Both cities share one y-range so their charts compare directly.
  const vals = DATA.cities.flatMap((c) => [...rows[c].p5, ...rows[c].p95, DATA.base_values[c], ...(futures ? futures[c].flat() : [])]);
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
        { name: rateLabel, color },
        futures ? { name: `${FUTURES} simulated futures`, color, kind: "band" } : { name: "90% range", color, kind: "band" },
      ],
      height: 300,
      yDomain: [lo, hi],
      yFormat: usdK,
      series: [{
        name: city, color,
        points: years.map((yr, i) => ({ x: yr, y: r.median[i] })),
        band: futures ? undefined : years.map((yr, i) => ({ x: yr, lo: r.p5[i], hi: r.p95[i] })),
        endLabel: usdK(r.median.at(-1)),
      }],
      spaghetti: futures?.[city],
      ref: { y: DATA.base_values[city], label: "Flat population", labelAbove: growth < 0 },
      xTicks: years.filter((yr) => (yr - DATA.base_year) % 2 === 0),
      tooltip: (yr) => {
        const i = yr - DATA.base_year;
        return [
          [color, "middle estimate", usd(r.median[i])],
          [null, "90% range", `${usdK(r.p5[i])}–${usdK(r.p95[i])}`],
          [css("--text-muted"), "flat population", usd(DATA.base_values[city])],
        ];
      },
    });
  }

  const tableRow = (label, r) => [label, ...DATA.cities.flatMap((c) => [usd(r[c].median.at(-1)), pct(100 * (r[c].median.at(-1) / DATA.base_values[c] - 1))])];
  table(
    document.getElementById("explorer-table"),
    ["Scenario", ...DATA.cities.flatMap((c) => [`${c} ${end}`, "Change"])],
    [
      ...DATA.presets.map((p) => tableRow(`${p.label} (${pct(p.growth, 2)}/yr)`, unitRows(p.growth, state.model))),
      tableRow(DATA.scenario_paths.dof.label, DATA.scenario_paths.dof[state.model]),
    ],
    [1, 2, 3, 4],
  );
}

function setupControls() {
  const slider = document.getElementById("growth");
  const [min, max] = [+slider.min, +slider.max];
  const edge = (v) => Math.min(100, Math.max(0, (100 * (v - min)) / (max - min)));
  const shade = (from, to) => h("div", { class: "slider-shade", style: `left:${from}%;width:${to - from}%` }, document.getElementById("slider-track"));
  shade(0, edge(DATA.support.lo));
  shade(edge(DATA.support.hi), 100);

  const presetEl = document.getElementById("presets");
  const valueEl = document.getElementById("home-value");
  const baseValue = DATA.base_values["Grass Valley"];
  const defaultGrowth = DATA.presets.find((p) => p.key === "recent").growth;
  // Readouts use the exact rate, so a preset is pressed only at its own rate, not at the nearest slider step.
  const same = (a, b) => Math.abs(a - b) < 1e-9;
  const buttons = DATA.presets.map((p) => {
    const b = h("button", { type: "button", "aria-pressed": "false" }, presetEl, `${p.label} ${pct(p.growth, 2)}`);
    b.addEventListener("click", () => {
      state.path = null;
      state.growth = p.growth;
      slider.value = p.growth;
      sync();
    });
    return [b, p];
  });
  const modelEl = document.getElementById("model-choice");
  for (const [label, key] of [["Peer counties", "peer"], ["All counties", "all"]]) {
    const b = h("button", { type: "button", role: "radio", "data-m": key }, modelEl, label);
    b.addEventListener("click", () => {
      state.model = key;
      sync();
    });
  }
  const dollarsEl = document.getElementById("dollars-choice");
  for (const [label, key] of [["Today's dollars", "real"], ["Sticker price", "nominal"]]) {
    const b = h("button", { type: "button", role: "radio", "data-d": key }, dollarsEl, label);
    b.addEventListener("click", () => {
      state.dollars = key;
      sync();
      renderHistory();
    });
  }
  const pathBtn = h("button", { type: "button", "aria-pressed": "false" }, presetEl, "State projection");
  pathBtn.addEventListener("click", () => {
    state.path = "dof";
    sync();
  });
  const futuresBtn = document.getElementById("futures");
  futuresBtn.addEventListener("click", () => {
    state.futures = !state.futures;
    sync();
  });

  function writeHash() {
    const q = new URLSearchParams();
    if (state.model !== "peer") q.set("m", state.model);
    if (state.dollars !== "real") q.set("d", state.dollars);
    if (state.path) q.set("p", state.path);
    else if (!same(state.growth, defaultGrowth)) {
      const hit = DATA.presets.find((p) => same(p.growth, state.growth));
      if (hit) q.set("p", hit.key);
      else q.set("g", String(Math.round(state.growth * 10) / 10));
    }
    if (state.value !== baseValue) q.set("v", String(state.value));
    const hash = q.toString();
    history.replaceState(null, "", location.pathname + location.search + (hash ? "#" + hash : ""));
  }

  function sync() {
    for (const [b, p] of buttons) b.setAttribute("aria-pressed", String(!state.path && same(p.growth, state.growth)));
    for (const c of modelEl.children) c.setAttribute("aria-checked", String(c.dataset.m === state.model));
    for (const c of dollarsEl.children) c.setAttribute("aria-checked", String(c.dataset.d === state.dollars));
    slider.value = state.growth;
    writeHash();
    pathBtn.setAttribute("aria-pressed", String(!!state.path));
    slider.disabled = !!state.path;
    slider.setAttribute("aria-disabled", String(!!state.path));
    futuresBtn.setAttribute("aria-checked", String(state.futures));
    futuresBtn.disabled = !!state.path;
    renderExplorer();
  }

  slider.addEventListener("input", () => {
    state.path = null;
    state.growth = +slider.value;
    sync();
  });

  valueEl.addEventListener("input", () => {
    const v = parseValue(valueEl.value);
    if (v === null) return;
    state.value = v;
    sync();
  });
  valueEl.addEventListener("blur", () => { valueEl.value = usd(state.value); });

  const copyBtn = document.getElementById("copy-link");
  copyBtn.addEventListener("click", async () => {
    writeHash();
    try {
      await navigator.clipboard.writeText(location.href);
      copyBtn.textContent = "Copied";
    } catch {
      copyBtn.textContent = "Copy failed";
    }
    setTimeout(() => { copyBtn.textContent = "Copy link to this scenario"; }, 2000);
  });

  state.growth = defaultGrowth;
  state.value = baseValue;
  readHash(Object.fromEntries(DATA.presets.map((p) => [p.key, p])));
  valueEl.value = usd(state.value);
  sync();
}

const EVENT_DATE = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

// One marker per year: labels join with " · ", and the tooltip lists each event with its date.
function historyMarkers() {
  const byYear = new Map();
  for (const e of DATA.events) byYear.set(e.year, [...(byYear.get(e.year) ?? []), e]);
  return [...byYear].map(([year, events]) => ({
    x: year,
    // A year with mixed kinds takes the first (earliest) event's color.
    kind: events[0].kind,
    label: events.map((e) => e.label).join(" · "),
    rows: events.map((e) => [css(MARKER_KINDS[e.kind].color), e.label, e.date.length > 4 ? EVENT_DATE.format(new Date(e.date)) : e.date]),
  }));
}

function renderHistory() {
  const hist = DATA.history;
  const base = hist[0];
  const sticker = state.dollars === "nominal";
  const names = { population: "Nevada County population", "Grass Valley": "Grass Valley home value", "Nevada City": "Nevada City home value" };
  const colors = { population: css("--series-1"), "Grass Valley": css("--series-2"), "Nevada City": css("--series-3") };
  const keys = Object.keys(names);
  const field = (k) => (sticker && k !== "population" ? `${k}_nominal` : k);
  const idx = (r, k) => (100 * r[field(k)]) / base[field(k)];
  const cities = keys.filter((k) => k !== "population");
  const chartEl = document.getElementById("history-chart");
  // End labels need 150px; on a narrow screen the legend names the series instead.
  const narrow = chartEl.clientWidth < 520;
  lineChart(chartEl, {
    title: sticker
      ? `Indexed to ${base.year} = 100, home values at sticker price`
      : `Indexed to ${base.year} = 100, home values after inflation`,
    legendItems: keys.map((k) => ({ name: names[k], color: colors[k] })),
    markers: historyMarkers(),
    height: 360,
    rightMargin: narrow ? 64 : 150,
    yFormat: (v) => v.toFixed(0),
    series: keys.map((k) => ({
      name: names[k], color: colors[k],
      points: hist.map((r) => ({ x: r.year, y: idx(r, k) })),
      endLabel: narrow ? undefined : names[k].replace(" home value", ""),
    })),
    xTicks: hist.map((r) => r.year).filter((yr) => yr % 4 === 1),
    tooltip: (yr) => {
      const r = hist.find((d) => d.year === yr);
      const pop = [colors.population, names.population, r.population.toLocaleString("en-US")];
      if (sticker) {
        return [pop, ...cities.map((k) => [`In ${yr} a typical ${k} home listed at ${usd(r[`${k}_nominal`])}, which is ${usd(r[k])} in ${DATA.base_year} dollars.`])];
      }
      return [pop, ...cities.map((k) => [colors[k], `${names[k]}, ${DATA.base_year} dollars (sticker price then ${usd(r[`${k}_nominal`])})`, usd(r[k])])];
    },
  });
  table(
    document.getElementById("history-table"),
    ["Year", "County population", "Grass Valley value", "Nevada City value", "Grass Valley sticker price", "Nevada City sticker price"],
    hist.map((r) => [
      r.year, r.population.toLocaleString("en-US"), usd(r["Grass Valley"]), usd(r["Nevada City"]),
      usd(r["Grass Valley_nominal"]), usd(r["Nevada City_nominal"]),
    ]),
    [1, 2, 3, 4, 5],
  );
}

const EXPLAINED_BARS = [
  { key: "actual_pct", name: "Actual change since 2001", color: "--text-secondary", opacity: 1 },
  { key: "population_pct", name: "Part that goes with population growth", color: "--series-1", opacity: 1 },
  { key: "other_pct", name: "Remainder (other forces)", color: "--series-3", opacity: 0.5 },
];

/** Three bars per city on one shared percent axis, drawn as small multiples. rows: {city: explained entry}. */
function waterfallChart(container, rows) {
  container.replaceChildren();
  const cities = Object.keys(rows);
  legend(container, EXPLAINED_BARS.map((b) => ({ name: b.name, color: css(b.color) })));
  const all = cities.flatMap((c) => [0, rows[c].actual_pct, rows[c].population_lo, rows[c].population_hi, rows[c].other_pct]);
  const yt = niceTicks(Math.min(...all), Math.max(...all), 5);
  const grid = h("div", { class: "multiples" }, container);
  // Lay out every box before drawing, so each chart measures its final width.
  const boxes = cities.map((city) => {
    const box = h("div", {}, grid);
    h("p", { class: "chart-title" }, box, city);
    return box;
  });
  for (const [ci, city] of cities.entries()) {
    const box = boxes[ci];
    const e = rows[city];
    const width = Math.max(box.clientWidth, 300);
    const m = { top: 16, right: 8, bottom: 20, left: 48 };
    const height = 300;
    const y = linear([yt.lo, yt.hi], [height - m.bottom, m.top]);
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${city}: actual change since ${e.start_year} ${pct(e.actual_pct, 0)}, part that goes with population growth ${pct(e.population_pct, 0)}, remainder from other forces ${pct(e.other_pct, 0)}` }, box);
    const ax = s("g", { class: "axis" }, svg);
    for (const v of yt.ticks) {
      s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(v), y2: y(v) }, ax);
      s("text", { x: m.left - 6, y: y(v), "text-anchor": "end", "dominant-baseline": "middle" }, ax, v.toFixed(0) + "%");
    }
    const slot = (width - m.left - m.right) / EXPLAINED_BARS.length;
    const bw = Math.min(slot * 0.6, 70);
    EXPLAINED_BARS.forEach((b, i) => {
      const v = e[b.key];
      const cx = m.left + slot * (i + 0.5);
      const g = s("g", {}, svg);
      s("rect", { x: cx - bw / 2, y: Math.min(y(0), y(v)), width: bw, height: Math.abs(y(v) - y(0)), fill: css(b.color), "fill-opacity": b.opacity }, g);
      if (b.key === "population_pct") {
        const w = { stroke: css("--text-primary"), "stroke-width": 1.5 };
        s("line", { x1: cx, x2: cx, y1: y(e.population_lo), y2: y(e.population_hi), ...w }, g);
        for (const end of [e.population_lo, e.population_hi]) s("line", { x1: cx - 5, x2: cx + 5, y1: y(end), y2: y(end), ...w }, g);
      }
      const top = b.key === "population_pct" ? Math.min(y(e.population_hi), y(v)) : y(Math.max(v, 0));
      s("text", { class: "direct-label", x: cx, y: (v >= 0 ? top : y(v)) - 5, "text-anchor": "middle" }, g, pct(v, 0));
      s("rect", { x: cx - slot / 2, y: m.top, width: slot, height: height - m.top - m.bottom, fill: "transparent" }, g);
      g.addEventListener("pointermove", (evt) => showTip(evt, `${city}: ${b.name}`, [
        [css(b.color), `${e.start_year} to ${e.end_year}`, pct(v, 1)],
        ...(b.key === "population_pct" ? [[css(b.color), "90% range", `${signed(e.population_lo)} to ${signed(e.population_hi)}`]] : []),
      ]));
      g.addEventListener("pointerleave", hideTip);
    });
    s("line", { x1: m.left, x2: width - m.right, y1: y(0), y2: y(0), stroke: css("--text-muted"), "stroke-width": 1 }, svg);
  }
}

function renderExplained() {
  const ex = DATA.explained;
  const cities = Object.keys(ex);
  const hist = DATA.history;
  const grew = 100 * (hist.at(-1).population / hist[0].population - 1);
  const first = ex[cities[0]];
  const share = (c) => Math.round((100 * ex[c].population_pct) / ex[c].actual_pct);
  const [c1, c2] = cities;
  const lo = Math.min(...cities.map((c) => ex[c].population_lo));
  const hi = Math.max(...cities.map((c) => ex[c].population_hi));
  document.getElementById("explained-lede").textContent =
    `The county grew about ${grew.toFixed(0)}% since ${first.start_year}. Run through the model's own arithmetic, that growth goes with roughly ${share(c1)}% of ${c1}'s ${ex[c1].actual_pct.toFixed(0)}% real price change and ${share(c2)}% of ${c2}'s ${ex[c2].actual_pct.toFixed(0)}%. ` +
    `The 90% range on the population part runs from ${lo.toFixed(0)}% to ${hi.toFixed(0)}%, ${lo < 0 && hi > 0 ? "so it crosses zero" : "so it stays on one side of zero"}. ` +
    "The rest is other forces the model holds fixed: mortgage rates, incomes, the pandemic, insurance. The two parts compound, so they do not add up to the actual change. Responses beyond ten years after a change hold the ten-year value.";
  waterfallChart(document.getElementById("explained-chart"), ex);
  table(
    document.getElementById("explained-table"),
    ["City", "Measure", `Change ${first.start_year} to ${first.end_year}`, "90% range"],
    cities.flatMap((c) => EXPLAINED_BARS.map((b) => [
      c, b.name, pct(ex[c][b.key], 1),
      b.key === "population_pct" ? `${signed(ex[c].population_lo)} to ${signed(ex[c].population_hi)}` : "",
    ])),
    [2, 3],
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
  const spec = [["lp", "Direct estimate by horizon (local projection)", css("--series-1")], ["ar1", "Price-momentum model (AR(1))", css("--series-2")]];
  const imp = DATA.impulse;
  lineChart(document.getElementById("impulse-chart"), {
    title: "Cumulative change in real home prices after a one-time 1% population increase, peer counties",
    legendItems: spec.map(([, name, color]) => ({ name, color })).concat([{ name: "90% range, direct estimate", color: spec[0][2], kind: "band" }]),
    height: 320,
    yFormat: (v) => v.toFixed(1) + "%",
    xFormat: (v) => (v === 0 ? "Year before" : `Year ${v}`),
    xTicks: yrs.filter((v) => v % 2 === 0),
    rightMargin: 64,
    series: spec.map(([k, name, color]) => ({
      name, color,
      points: yrs.map((i) => ({ x: i, y: imp[k].median[i] })),
      band: k === "lp" ? yrs.map((i) => ({ x: i, lo: imp.lp.p5[i], hi: imp.lp.p95[i] })) : undefined,
      endLabel: pct(imp[k].median.at(-1)),
    })),
    tooltip: (i) => [
      [spec[0][2], "direct estimate", `${pct(imp.lp.median[i], 2)} (${imp.lp.p5[i].toFixed(1)} to ${imp.lp.p95[i].toFixed(1)})`],
      [spec[1][2], "momentum model", pct(imp.ar1.median[i], 2)],
    ],
  });
  table(
    document.getElementById("impulse-table"),
    ["Years after the increase (1 = same year)", "Direct estimate", "90% range", "Momentum model"],
    yrs.slice(1).map((i) => [i, pct(imp.lp.median[i], 2), `${imp.lp.p5[i].toFixed(2)} to ${imp.lp.p95[i].toFixed(2)}`, pct(imp.ar1.median[i], 2)]),
    [1, 2, 3],
  );
}

const SPEC_LABELS = {
  "peer/same_year": "Peer counties, same-year",
  "peer/lagged": "Peer counties, lagged",
  "all/same_year": "All counties, same-year",
  "all/lagged": "All counties, lagged",
  "large/same_year": "Large counties only",
  "all/2001-2010": "All counties, 2001-2010",
  "all/2011-2024": "All counties, 2011-2024",
  "all/drop_2020_2022": "All counties, without 2020-2022",
  "all/three_price_lags": "All counties, three price lags",
  "all/sign_split": "All counties, sign split (growth term)",
  "all/components": "Migration rate, natural change control (all)",
  "peer/components": "Migration rate, natural change control (peer)",
};

/** Dot-and-range chart. rows: [{spec, lr, lr_lo, lr_hi}], one per line; the headline row is filled. */
function rangeChart(container, rows, headline) {
  container.replaceChildren();
  const width = Math.max(container.clientWidth, 300);
  // At phone width a left label column would squeeze the plot, so each label sits above its row.
  const stacked = width < 520;
  const rowH = stacked ? 44 : 28;
  const m = { top: 8, right: 16, bottom: 28, left: stacked ? 16 : Math.min(230, width * 0.42) };
  const height = m.top + m.bottom + rowH * rows.length;
  const xt = niceTicks(Math.min(0, ...rows.map((r) => r.lr_lo)), Math.max(...rows.map((r) => r.lr_hi)), 5);
  const x = linear([xt.lo, xt.hi], [m.left, width - m.right]);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Long-run price change for 1% more people under each sample choice, with 90% ranges" }, container);
  const ax = s("g", { class: "axis" }, svg);
  for (const v of xt.ticks) {
    s("line", { class: "gridline", x1: x(v), x2: x(v), y1: m.top, y2: height - m.bottom }, ax);
    s("text", { x: x(v), y: height - m.bottom + 18, "text-anchor": "middle" }, ax, v.toFixed(1) + "%");
  }
  rows.forEach((r, i) => {
    const rowTop = m.top + rowH * i;
    const cy = stacked ? rowTop + 30 : rowTop + rowH / 2;
    const isHead = r.spec === headline;
    const g = s("g", { tabindex: 0 }, svg);
    const label = stacked
      ? { x: m.left, y: rowTop + 12, "text-anchor": "start" }
      : { x: m.left - 10, y: cy, "text-anchor": "end" };
    s("text", { class: "direct-label", ...label, "dominant-baseline": "middle" }, g, SPEC_LABELS[r.spec] || r.spec);
    s("line", { x1: x(r.lr_lo), x2: x(r.lr_hi), y1: cy, y2: cy, stroke: css("--series-1"), "stroke-width": 2, "stroke-linecap": "round" }, g);
    s("circle", { cx: x(r.lr), cy, r: 5, fill: isHead ? css("--series-1") : css("--surface-1"), stroke: css("--series-1"), "stroke-width": 2 }, g);
    s("rect", { x: 0, y: rowTop, width, height: rowH, fill: "transparent" }, g);
    const show = (evt) => showTip(evt, SPEC_LABELS[r.spec] || r.spec, [[css("--series-1"), "Estimate", `${r.lr.toFixed(2)}%`], [css("--series-1"), "90% range", `${r.lr_lo.toFixed(2)} to ${r.lr_hi.toFixed(2)}`]]);
    g.addEventListener("pointermove", show);
    g.addEventListener("pointerleave", hideTip);
    g.addEventListener("focus", () => { const b = g.getBoundingClientRect(); show({ clientX: b.left + x(r.lr), clientY: b.top }); });
    g.addEventListener("blur", hideTip);
  });
}

function renderMethod() {
  const p = DATA.panel;
  document.getElementById("method-panel").textContent =
    `The headline model compares yearly price growth after inflation with population growth in ${p.counties} California counties with fewer than ${DATA.headline.max_pop.toLocaleString("en-US")} people, the group that includes Nevada County, from ${p.first_year} to ${p.last_year} (${p.n.toLocaleString("en-US")} county-years). ` +
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
    const e = DATA.elasticity[`peer_${key}`];
    h("p", {}, box, `Price change per 1% more people, long-run, momentum model: ${e.point.toFixed(2)}% (90% interval ${e.lo.toFixed(2)} to ${e.hi.toFixed(2)}).`);
  }
}

function renderValidation() {
  const v = DATA.validation;
  document.getElementById("val-start").textContent = DATA.validation_first_origin;
  const corr = (r) => (r.corr == null ? "n/a" : r.corr.toFixed(2));
  table(document.getElementById("val-table"),
    ["Years ahead", "Model", "Peer RMSE", "Peer correlation", "Peer pairs", "All-county RMSE", "All-county correlation", "All-county pairs"],
    v.peer.map((r, i) => {
      const a = v.all[i];
      return [r.horizon, r.model, r.rmse.toFixed(3), corr(r), r.n.toLocaleString("en-US"), a.rmse.toFixed(3), corr(a), a.n.toLocaleString("en-US")];
    }), [0, 2, 3, 4, 5, 6, 7]);
}

function renderSpecCurve() {
  rangeChart(document.getElementById("spec-chart"), DATA.spec_curve, "peer/same_year");
  const c = DATA.components.peer;
  document.getElementById("spec-components").textContent =
    `Splitting population growth into net migration and natural change (births minus deaths) gives a migration coefficient of ${c.b_mig.toFixed(2)} on the peer set; natural change, which cannot respond to this year's prices, has a coefficient of ${c.b_natural.toFixed(2)}. The large negative natural-change coefficient comes from within-county timing, above all the 2020-2021 death surge in older counties while prices rose; it reflects age structure, not a price response.`;
}

function renderAll() {
  renderTiles();
  renderExplorer();
  renderHistory();
  renderExplained();
  renderCounties();
  renderImpulse();
  renderMethod();
  renderSpecCurve();
  renderValidation();
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
