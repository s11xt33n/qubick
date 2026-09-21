/* qhnn — сайт с результатами. Данные: data/*.json (qhnn.experiments.export_site). */
"use strict";

const MODELS = ["hybrid", "quantum", "classical", "classical_matched", "bottleneck"];
const MODEL_LABEL = {
  hybrid: "Hybrid QNN", quantum: "Quantum QNN", classical: "Classical MLP",
  classical_matched: "MLP (равное число параметров)", bottleneck: "Bottleneck (без квантовой части)",
};
const MODEL_DESC = {
  hybrid: ["Encoder → QuantumLayer → Head", "Основной объект исследования"],
  quantum: ["Признаки (PCA до n кубитов) → QuantumLayer → линейное считывание", "Чистый квантовый подход"],
  classical: ["MLP 32-16, ReLU", "Сильный классический baseline"],
  classical_matched: ["MLP с одним скрытым слоем, число параметров ≈ как у гибрида", "Честное сравнение по размеру"],
  bottleneck: ["Encoder → Linear(n, n) + tanh → Head", "Абляция: вклад именно квантового слоя"],
};
// категориальные слоты в фиксированном порядке: цвет закреплён за моделью
const MODEL_SLOT = { hybrid: 1, quantum: 2, classical: 3, classical_matched: 4, bottleneck: 5 };
const DS_LABEL = {
  iris: "Iris", wine: "Wine", breast_cancer: "Breast Cancer", moons: "Moons", circles: "Circles",
  "vision:mnist": "MNIST", "vision:fashion": "Fashion-MNIST", "vision:pneumonia": "PneumoniaMNIST",
  "vision:breast": "BreastMNIST",
};
const SERIES_LABEL = { tabular: "Табличные данные", vision: "Изображения", sweep: "Кубиты × глубина",
  ablation: "Абляция", shots: "Shots" };

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (m) => css(`--s${MODEL_SLOT[m]}`);
const fmt = (x, d = 3) => (x == null ? "—" : Number(x).toFixed(d));
const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
const uniq = (a) => [...new Set(a)];
const charts = {};
const data = {};

// ---------------------------------------------------------------- тема
(function theme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
    renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
})();

// ---------------------------------------------------------------- ECharts база
function chart(id) {
  const el = $(id);
  if (!el || typeof echarts === "undefined") return null;
  if (!charts[id]) charts[id] = echarts.init(el, null, { renderer: "svg" });
  return charts[id];
}
window.addEventListener("resize", () => Object.values(charts).forEach((c) => c.resize()));

function base(extra = {}) {
  const ink2 = css("--ink-2"), muted = css("--muted"), grid = css("--grid"), axis = css("--axis");
  const axisStyle = {
    axisLine: { lineStyle: { color: axis } }, axisTick: { show: false },
    axisLabel: { color: muted, fontSize: 12 }, splitLine: { lineStyle: { color: grid, width: 1 } },
    nameTextStyle: { color: ink2, fontSize: 12 },
  };
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css("--ink") },
    grid: { left: 56, right: 20, top: 64, bottom: 44, containLabel: false },
    legend: { top: 0, left: 0, textStyle: { color: ink2, fontSize: 12 }, itemWidth: 14, itemHeight: 8,
      icon: "roundRect" },
    tooltip: {
      backgroundColor: css("--surface"), borderColor: css("--border"), borderWidth: 1,
      textStyle: { color: css("--ink"), fontSize: 12 }, extraCssText: "box-shadow:0 2px 10px rgba(0,0,0,.12)",
    },
    xAxis: { ...axisStyle, splitLine: { show: false } },
    yAxis: { ...axisStyle },
    animationDuration: 400,
    ...extra,
  };
}

// строка тултипа: короткая линия цвета серии, значение впереди, имя за ним
function tipRow(col, value, label) {
  const key = `<span style="display:inline-block;width:12px;height:2px;background:${col};vertical-align:middle;margin-right:6px"></span>`;
  return `<div>${key}<b>${value}</b> <span style="color:${css("--ink-2")}">${esc(label)}</span></div>`;
}
function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }

// планки погрешностей (custom series)
function errorBars(points, col, xIndexOffset) {
  return {
    type: "custom", silent: true, z: 5, data: points,
    renderItem: (params, api) => {
      const x = api.value(0), lo = api.value(1), hi = api.value(2);
      if (lo == null || hi == null || isNaN(lo)) return;
      const p1 = api.coord([x, lo]), p2 = api.coord([x, hi]);
      const off = xIndexOffset ? xIndexOffset(params, api) : 0;
      const cx = p1[0] + off;
      const s = { stroke: css("--ink-2"), lineWidth: 1 };
      return { type: "group", children: [
        { type: "line", shape: { x1: cx, y1: p1[1], x2: cx, y2: p2[1] }, style: s },
        { type: "line", shape: { x1: cx - 3, y1: p1[1], x2: cx + 3, y2: p1[1] }, style: s },
        { type: "line", shape: { x1: cx - 3, y1: p2[1], x2: cx + 3, y2: p2[1] }, style: s },
      ] };
    },
  };
}

function placeholder(id, text) {
  const c = chart(id);
  if (!c) return;
  c.setOption({ title: { text, left: "center", top: "middle",
    textStyle: { color: css("--muted"), fontSize: 14, fontWeight: "normal" } },
    xAxis: { show: false }, yAxis: { show: false }, series: [] }, true);
}

function table(id, head, rows) {
  const t = $(id);
  if (!t) return;
  t.textContent = "";
  const thead = t.createTHead().insertRow();
  head.forEach((h) => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
  const tb = t.createTBody();
  rows.forEach((r) => {
    const tr = tb.insertRow();
    r.forEach((c) => {
      const td = tr.insertCell();
      if (c && typeof c === "object") { td.textContent = c.text; if (c.cls) td.className = c.cls; }
      else td.textContent = c;
    });
  });
}

function fillSelect(id, values, labels) {
  const s = $(id);
  if (!s) return;
  const prev = s.value;
  s.textContent = "";
  values.forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = labels ? labels[v] || v : v; s.appendChild(o); });
  if (values.includes(prev)) s.value = prev;
}

// ---------------------------------------------------------------- загрузка
async function load(name) {
  try {
    const r = await fetch(`data/${name}.json?t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

async function loadAll() {
  const names = ["progress", "tabular", "vision", "sweep", "ablation", "shots", "barren", "speed"];
  const res = await Promise.all(names.map(load));
  names.forEach((n, i) => { if (res[i]) data[n] = res[i]; });
  renderAll();
}

function renderAll() {
  const steps = [renderModels, renderProgress, renderOverview, renderTabular, renderVision,
    renderSweep, renderAblation, renderShots, renderBarren, renderSpeed];
  steps.forEach((f) => { try { f(); } catch (e) { console.error(f.name, e); } });
}

// ---------------------------------------------------------------- модели
function renderModels() {
  table("models-table", ["Модель", "Архитектура", "Роль в сравнении"],
    MODELS.map((m) => [MODEL_LABEL[m], MODEL_DESC[m][0], MODEL_DESC[m][1]]));
}

// ---------------------------------------------------------------- прогресс
function renderProgress() {
  const p = data.progress;
  if (!p) return;
  const s = p.series;
  let done = 0, total = 0;
  Object.values(s).forEach((v) => { done += v.done; total += v.total; });
  $("hero-num").textContent = total ? Math.floor((100 * done) / total) + "%" : "—";
  $("hero-cap").textContent = `${done.toLocaleString("ru")} из ${total.toLocaleString("ru")} обучений моделей · обновлено ${p.updated}`;
  $("updated").textContent = "данные обновлены " + p.updated;
  const tiles = $("tiles");
  tiles.textContent = "";
  Object.entries(s).forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "tile";
    const l = document.createElement("div"); l.className = "label"; l.textContent = SERIES_LABEL[k] || k;
    const val = document.createElement("div"); val.className = "value";
    val.textContent = `${v.done.toLocaleString("ru")} / ${v.total.toLocaleString("ru")}`;
    const m = document.createElement("div"); m.className = "meter";
    const f = document.createElement("div"); f.style.width = (v.total ? (100 * v.done) / v.total : 0) + "%";
    m.appendChild(f); d.append(l, val, m); tiles.appendChild(d);
  });
  const mon = p.monitor || [];
  const t = mon.map((r) => r.time);
  const line = (id, key, max, unit) => {
    const c = chart(id);
    if (!c) return;
    c.setOption(base({
      grid: { left: 40, right: 12, top: 12, bottom: 28 },
      legend: { show: false },
      tooltip: { ...base().tooltip, trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: css("--muted"), width: 1 } },
        formatter: (ps) => `${ps[0].axisValue}<br>` + tipRow(css("--accent"), `${ps[0].value}${unit}`, "") },
      xAxis: { ...base().xAxis, type: "category", data: t, boundaryGap: false,
        axisLabel: { color: css("--muted"), fontSize: 11, interval: Math.max(0, Math.floor(t.length / 6)) } },
      yAxis: { ...base().yAxis, type: "value", max, min: 0 },
      series: [{ type: "line", data: mon.map((r) => r[key]), showSymbol: false, smooth: false,
        lineStyle: { width: 2, color: css("--accent") }, areaStyle: { color: css("--accent"), opacity: 0.1 } }],
    }), true);
  };
  line("ch-cpu", "cpu_load", 100, "%");
  line("ch-gpu", "gpu_util", 100, "%");
  line("ch-temp", "gpu_temp", null, " °C");
  line("ch-power", "gpu_power", 300, " Вт");
}

// ---------------------------------------------------------------- обзор
function renderOverview() {
  const k = $("kpis");
  k.textContent = "";
  const p = data.progress;
  let done = 0;
  if (p) Object.values(p.series).forEach((v) => (done += v.done));
  const kp = [
    [done.toLocaleString("ru"), "обученных моделей"],
    ["158", "автотестов"],
    ["9", "наборов данных"],
    ["2–16", "кубитов"],
  ];
  if (data.speed) {
    const sp = data.speed;
    const get = (b, m, n) => (sp.find((r) => r.backend === b && r.diff_method === m && r.n_qubits === n && r.device === "cpu") || {}).step_time;
    const ours = get("torch", "parameter-shift", 6), pl = get("pennylane", "parameter-shift", 6);
    if (ours && pl) kp.push([`×${Math.round(pl / ours)}`, "быстрее PennyLane (parameter-shift, 6 кубитов)"]);
  }
  kp.forEach(([v, l]) => {
    const d = document.createElement("div"); d.className = "kpi";
    const a = document.createElement("div"); a.className = "value"; a.textContent = v;
    const b = document.createElement("div"); b.className = "label"; b.textContent = l;
    d.append(a, b); k.appendChild(d);
  });

  const f = $("findings");
  f.textContent = "";
  const items = [];
  const tb = data.tabular;
  if (tb) {
    const sig = (vs) => tb.tests.filter((r) => r.vs === vs && r.significant && r.mean_diff > 0).map((r) => DS_LABEL[r.dataset]);
    const sigQ = sig("quantum");
    if (sigQ.length) items.push(`Гибридная модель значимо точнее чистой квантовой сети на ${sigQ.join(", ")}: классический encoder снимает ограничение «признаков не больше, чем кубитов».`);
    const wins = tb.tests.filter((r) => ["classical_matched", "classical"].includes(r.vs) && r.significant && r.mean_diff > 0).length;
    const losses = tb.tests.filter((r) => ["classical_matched", "classical"].includes(r.vs) && r.significant && r.mean_diff < 0).length;
    items.push(`На табличных данных гибрид не превосходит классическую MLP того же размера: значимых побед ${wins}, значимых поражений ${losses} из ${tb.tests.filter((r) => ["classical_matched", "classical"].includes(r.vs)).length} сравнений.`);
    const t = (m) => tb.summary.filter((r) => r.model === m).reduce((a, r) => a + r.train_time_mean, 0);
    const ratio = t("hybrid") / t("classical_matched");
    if (isFinite(ratio)) items.push(`Квантовый слой дорог: обучение гибрида в среднем в ${ratio.toFixed(0)} раз медленнее классической модели того же размера (симуляция схемы на классическом железе).`);
  }
  if (data.barren) {
    const b = data.barren.filter((r) => r.cost === "global");
    const Ls = uniq(b.map((r) => r.n_layers));
    const L = Ls.includes(5) ? 5 : Ls[0];
    const row = (n) => b.find((r) => r.n_qubits === n && r.n_layers === L);
    const lo = row(2), hi = row(Math.max(...b.map((r) => r.n_qubits)));
    if (lo && hi) items.push(`Barren plateaus подтверждены: для глобальной функции стоимости дисперсия градиента падает примерно в ${Math.round(lo.grad_var / hi.grad_var).toLocaleString("ru")} раз при росте с 2 до ${hi.n_qubits} кубитов.`);
  }
  if (!items.length) items.push("Результаты загружаются…");
  items.forEach((s) => { const li = document.createElement("li"); li.textContent = s; f.appendChild(li); });
}

// ---------------------------------------------------------------- табличные данные
function renderTabular() {
  const d = data.tabular;
  if (!d) return;
  const ds = ["iris", "wine", "breast_cancer", "moons", "circles"].filter((x) => d.summary.some((r) => r.dataset === x));
  const models = MODELS.filter((m) => d.summary.some((r) => r.model === m));
  const get = (dset, m) => d.summary.find((r) => r.dataset === dset && r.model === m) || {};
  // точка + планка ±std для каждой модели, модели разнесены внутри группы датасета
  const n = models.length, step = 16;
  const series = models.map((m, i) => ({
    name: MODEL_LABEL[m], type: "custom", z: 3,
    itemStyle: { color: color(m) },
    encode: { x: 0, y: [1, 2, 3], tooltip: [1] },
    data: ds.map((x, j) => [j, get(x, m).accuracy_mean, get(x, m).accuracy_mean - get(x, m).accuracy_std,
      get(x, m).accuracy_mean + get(x, m).accuracy_std]),
    renderItem: (params, api) => {
      const off = (i - (n - 1) / 2) * step;
      const c = api.coord([api.value(0), api.value(1)]);
      const lo = api.coord([api.value(0), api.value(2)]), hi = api.coord([api.value(0), api.value(3)]);
      const x = c[0] + off, col = color(m);
      return { type: "group", children: [
        { type: "line", shape: { x1: x, y1: lo[1], x2: x, y2: hi[1] }, style: { stroke: col, lineWidth: 2, opacity: 0.55 } },
        { type: "circle", shape: { cx: x, cy: c[1], r: 5 }, style: { fill: col, stroke: css("--surface"), lineWidth: 2 } },
      ] };
    },
  }));
  const c = chart("ch-tabular");
  if (c) c.setOption(base({
    grid: { left: 56, right: 20, top: 64, bottom: 36 },
    tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: css("--grid"), opacity: 0.35 } },
      formatter: (ps) => {
        const x = ds[ps[0].dataIndex];
        return `<b>${DS_LABEL[x]}</b><br>` + models.map((m) => tipRow(color(m),
          `${fmt(get(x, m).accuracy_mean)} ± ${fmt(get(x, m).accuracy_std)}`, MODEL_LABEL[m])).join("");
      } },
    legend: { ...base().legend, icon: "circle", itemWidth: 9, itemHeight: 9 },
    xAxis: { ...base().xAxis, type: "category", data: ds.map((x) => DS_LABEL[x]) },
    yAxis: { ...base().yAxis, type: "value", max: 1, interval: 0.05,
      min: (v) => Math.floor(v.min * 20) / 20, axisLabel: { color: css("--muted"), formatter: (v) => v.toFixed(2) } },
    series,
  }), true);

  table("tab-tests", ["Набор данных", ...models.filter((m) => m !== "hybrid").map((m) => "vs " + MODEL_LABEL[m])],
    ds.map((x) => [DS_LABEL[x], ...models.filter((m) => m !== "hybrid").map((m) => {
      const r = d.tests.find((t) => t.dataset === x && t.vs === m);
      if (!r) return "—";
      const s = `${r.mean_diff > 0 ? "+" : ""}${(r.mean_diff * 100).toFixed(1)} п.п. (${r.wins}:${r.losses}, p=${fmt(r.p_value, 3)})`;
      return { text: s, cls: r.significant ? "sig" : "" };
    })]));
  table("tab-summary", ["Набор", "Модель", "Accuracy", "F1 (macro)", "Параметров", "Эпох", "Время, с"],
    d.summary.map((r) => [DS_LABEL[r.dataset], MODEL_LABEL[r.model], `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`,
      `${fmt(r.f1_mean)} ± ${fmt(r.f1_std)}`, Math.round(r.n_params_mean), Math.round(r.epochs_mean), fmt(r.train_time_mean, 1)]));
}

// ---------------------------------------------------------------- изображения
function renderVision() {
  const d = data.vision;
  if (!d || !d.summary.length) { placeholder("ch-vision", "Расчёт идёт — данные появятся здесь автоматически"); return; }
  const dss = Object.keys(DS_LABEL).filter((x) => x.startsWith("vision:") && d.summary.some((r) => r.dataset === x));
  fillSelect("vis-ds", dss, DS_LABEL);
  const ds = $("vis-ds").value;
  // показываем только те числа кубитов, для которых уже есть результаты
  const qAvail = uniq(d.summary.filter((r) => r.dataset === ds && r.n_qubits > 0).map((r) => r.n_qubits)).sort((a, b) => a - b);
  fillSelect("vis-q", qAvail.map(String));
  const q = +$("vis-q").value;
  const rows = d.summary.filter((r) => r.dataset === ds && (r.n_qubits === 0 || r.n_qubits === q));
  const ns = uniq(rows.map((r) => r.n_train)).sort((a, b) => a - b);
  const models = MODELS.filter((m) => rows.some((r) => r.model === m));
  const get = (m, n) => rows.find((r) => r.model === m && r.n_train === n) || {};
  $("vis-title").textContent = `${DS_LABEL[ds]}: accuracy в зависимости от размера обучающей выборки (${q} кубитов)`;
  const c = chart("ch-vision");
  if (c) c.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: css("--muted"), width: 1 } },
      formatter: (ps) => `<b>${ps[0].axisValue} примеров</b><br>` + models.map((m) => {
        const r = get(m, +ps[0].axisValue);
        return tipRow(color(m), `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`, MODEL_LABEL[m]);
      }).join("") },
    xAxis: { ...base().xAxis, type: "category", data: ns.map(String), name: "Обучающих примеров", nameLocation: "middle", nameGap: 28 },
    yAxis: { ...base().yAxis, type: "value", scale: true, name: "Accuracy" },
    series: models.map((m) => ({ name: MODEL_LABEL[m], type: "line", data: ns.map((n) => get(m, n).accuracy_mean),
      lineStyle: { width: 2, color: color(m) }, itemStyle: { color: color(m), borderColor: css("--surface"), borderWidth: 2 },
      symbolSize: 8, emphasis: { focus: "series" } })),
  }), true);
  const others = models.filter((m) => m !== "hybrid");
  const tests = (d.tests || []).filter((t) => t.dataset === ds && t.n_qubits === q);
  table("vis-tests", ["Примеров", ...others.map((m) => "vs " + MODEL_LABEL[m])],
    ns.map((n) => [String(n), ...others.map((m) => {
      const r = tests.find((t) => t.n_train === n && t.vs === m);
      if (!r) return "—";
      return { text: `${r.mean_diff > 0 ? "+" : ""}${(r.mean_diff * 100).toFixed(1)} п.п. (${r.wins}:${r.losses})`,
        cls: r.significant ? "sig" : "" };
    })]));
}

// ---------------------------------------------------------------- кубиты × глубина
function renderSweep() {
  const d = data.sweep;
  if (!d || !d.summary.length) { placeholder("ch-sweep-heat", "Расчёт идёт…"); return; }
  const dss = ["breast_cancer", "wine", "moons"].filter((x) => d.summary.some((r) => r.dataset === x));
  fillSelect("sw-ds", dss, DS_LABEL);
  const ds = $("sw-ds").value, m = $("sw-model").value;
  const rows = d.summary.filter((r) => r.dataset === ds && r.model === m);
  const qs = uniq(d.summary.map((r) => r.n_qubits)).sort((a, b) => a - b);
  const Ls = uniq(d.summary.map((r) => r.n_layers)).sort((a, b) => a - b);
  const heat = [];
  rows.forEach((r) => heat.push([qs.indexOf(r.n_qubits), Ls.indexOf(r.n_layers), r.accuracy_mean, r.accuracy_std, r.runs]));
  const c = chart("ch-sweep-heat");
  if (c) c.setOption(base({
    grid: { left: 56, right: 20, top: 16, bottom: 70 },
    legend: { show: false },
    tooltip: { ...base().tooltip, formatter: (p) => `<b>${fmt(p.value[2])} ± ${fmt(p.value[3])}</b><br>${qs[p.value[0]]} кубитов, ${Ls[p.value[1]]} слоёв<br><span style="color:${css("--muted")}">${p.value[4]} запусков</span>` },
    xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
    yAxis: { ...base().yAxis, type: "category", data: Ls.map(String), name: "Слои", splitLine: { show: false } },
    visualMap: { min: 0.5, max: 1, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemHeight: 160,
      textStyle: { color: css("--muted") }, inRange: { color: [css("--r250"), css("--r450"), css("--r650")] } },
    series: [{ type: "heatmap", data: heat, itemStyle: { borderColor: css("--surface"), borderWidth: 2, borderRadius: 4 },
      label: { show: true, formatter: (p) => fmt(p.value[2], 2), color: "#fff", fontSize: 12 } }],
  }), true);
  const ramp = ["--r250", "--r450", "--r650", "--r550"];
  const t = chart("ch-sweep-time");
  if (t) t.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>${ps[0].axisValue} кубитов</b><br>` +
      ps.map((p) => tipRow(p.color, p.value == null ? "—" : `${fmt(p.value, 2)} с`, p.seriesName)).join("") },
    xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
    yAxis: { ...base().yAxis, type: "log", name: "с / эпоха" },
    series: Ls.map((L, i) => ({ name: `L = ${L}`, type: "line", symbolSize: 8,
      data: qs.map((q) => { const r = rows.find((x) => x.n_qubits === q && x.n_layers === L); return r ? r.time_per_epoch_mean : null; }),
      lineStyle: { width: 2, color: css(ramp[i % ramp.length]) }, itemStyle: { color: css(ramp[i % ramp.length]), borderColor: css("--surface"), borderWidth: 2 } })),
  }), true);
}

// ---------------------------------------------------------------- абляция
function renderAblation() {
  const d = data.ablation;
  if (!d || !d.summary.length) { ["encoding", "ansatz", "reupload"].forEach((f) => placeholder(`ch-ab-${f}`, "Расчёт идёт…")); return; }
  const dss = ["moons", "circles", "breast_cancer"].filter((x) => d.summary.some((r) => r.dataset === x));
  fillSelect("ab-ds", dss, DS_LABEL);
  const ds = $("ab-ds").value;
  const rows = d.summary.filter((r) => r.dataset === ds);
  const factorLabel = { angle: "RY(x)", angle_x: "RX(x)", dense: "H·RZ·RY", strong: "Strongly entangling", basic: "RY + CNOT",
    hea: "HEA (RY·RZ + CZ)", false: "нет", true: "да" };
  ["encoding", "ansatz", "reupload"].forEach((f) => {
    const levels = uniq(rows.map((r) => String(r[f])));
    const models = ["hybrid", "quantum"].filter((m) => rows.some((r) => r.model === m));
    const avg = (m, lv) => { const rr = rows.filter((r) => r.model === m && String(r[f]) === lv); return rr.length ? rr.reduce((a, r) => a + r.accuracy_mean, 0) / rr.length : null; };
    const c = chart(`ch-ab-${f}`);
    if (c) c.setOption(base({
      grid: { left: 48, right: 12, top: 30, bottom: 30 },
      tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: css("--grid"), opacity: 0.4 } },
        formatter: (ps) => `<b>${ps[0].axisValue}</b><br>` + ps.map((p) => tipRow(p.color, fmt(p.value), p.seriesName)).join("") },
      xAxis: { ...base().xAxis, type: "category", data: levels.map((l) => factorLabel[l] || l) },
      yAxis: { ...base().yAxis, type: "value", scale: true },
      series: models.map((m) => ({ name: MODEL_LABEL[m], type: "bar", barWidth: 18, barGap: "12%",
        itemStyle: { color: color(m), borderRadius: [4, 4, 0, 0] }, data: levels.map((l) => avg(m, l)) })),
    }), true);
  });
  const top = [...rows].sort((a, b) => b.accuracy_mean - a.accuracy_mean).slice(0, 6);
  table("ab-top", ["Модель", "Кодирование", "Анзац", "Re-upl.", "Accuracy"],
    top.map((r) => [MODEL_LABEL[r.model], factorLabel[r.encoding] || r.encoding, factorLabel[r.ansatz] || r.ansatz,
      factorLabel[String(r.reupload)], `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`]));
}

// ---------------------------------------------------------------- shots
function renderShots() {
  const d = data.shots;
  if (!d || !d.summary.length) return;
  const order = [100, 1000, 10000, 0];
  const lbl = (s) => (s === 0 ? "точно (∞)" : s.toLocaleString("ru"));
  const dss = uniq(d.summary.map((r) => r.dataset));
  const cols = [css("--s1"), css("--s2")];
  const c = chart("ch-shots");
  if (c) c.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>shots: ${ps[0].axisValue}</b><br>` +
      ps.filter((p) => p.seriesType === "line").map((p) => {
        const r = d.summary.find((x) => x.dataset === dss[p.seriesIndex] && lbl(x.shots) === p.axisValue) || {};
        return tipRow(p.color, `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`, p.seriesName);
      }).join("") },
    xAxis: { ...base().xAxis, type: "category", data: order.map(lbl), name: "Число измерений", nameLocation: "middle", nameGap: 28 },
    yAxis: { ...base().yAxis, type: "value", scale: true, name: "Accuracy" },
    series: dss.map((x, i) => ({ name: DS_LABEL[x], type: "line", symbolSize: 8,
      data: order.map((s) => (d.summary.find((r) => r.dataset === x && r.shots === s) || {}).accuracy_mean),
      lineStyle: { width: 2, color: cols[i] }, itemStyle: { color: cols[i], borderColor: css("--surface"), borderWidth: 2 } })),
  }), true);
}

// ---------------------------------------------------------------- barren plateaus
function renderBarren() {
  const d = data.barren;
  if (!d) return;
  const Ls = uniq(d.map((r) => r.n_layers)).sort((a, b) => a - b);
  const qs = uniq(d.map((r) => r.n_qubits)).sort((a, b) => a - b);
  const ramp = ["--r250", "--r350", "--r550", "--r650"];
  ["local", "global"].forEach((cost) => {
    const c = chart(`ch-barren-${cost}`);
    if (!c) return;
    c.setOption(base({
      tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>${ps[0].axisValue} кубитов</b><br>` +
        ps.map((p) => tipRow(p.color, p.value == null ? "—" : Number(p.value).toExponential(2), p.seriesName)).join("") },
      xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
      yAxis: { ...base().yAxis, type: "log", name: "Var[∂C/∂θ]", axisLabel: { color: css("--muted"), formatter: (v) => Number(v).toExponential(0) } },
      series: Ls.map((L, i) => ({ name: `L = ${L}`, type: "line", symbolSize: 8,
        data: qs.map((q) => { const r = d.find((x) => x.cost === cost && x.n_layers === L && x.n_qubits === q); return r ? r.grad_var : null; }),
        lineStyle: { width: 2, color: css(ramp[i % ramp.length]) }, itemStyle: { color: css(ramp[i % ramp.length]), borderColor: css("--surface"), borderWidth: 2 } })),
    }), true);
  });
}

// ---------------------------------------------------------------- скорость
function renderSpeed() {
  const d = data.speed;
  if (!d) { placeholder("ch-speed", "Замер запускается последним, на свободном сервере — результаты появятся после завершения всех экспериментов"); return; }
  const qs = uniq(d.map((r) => r.n_qubits)).sort((a, b) => a - b);
  const keys = uniq(d.map((r) => `${r.backend}|${r.diff_method}|${r.device}`));
  const name = (k) => { const [b, m, dev] = k.split("|"); return `${b === "torch" ? "qhnn" : "PennyLane"}, ${m}${b === "torch" ? ", " + dev.toUpperCase() : ""}`; };
  const c = chart("ch-speed");
  if (c) c.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>${ps[0].axisValue} кубитов</b><br>` +
      ps.filter((p) => p.value != null).map((p) => tipRow(p.color, `${fmt(p.value, 1)} мс`, p.seriesName)).join("") },
    xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
    yAxis: { ...base().yAxis, type: "log", name: "мс" },
    series: keys.map((k, i) => ({ name: name(k), type: "line", symbolSize: 8, connectNulls: false,
      lineStyle: { width: 2, color: css(`--s${i + 1}`), type: k.startsWith("pennylane") ? "dashed" : "solid" },
      itemStyle: { color: css(`--s${i + 1}`), borderColor: css("--surface"), borderWidth: 2 },
      data: qs.map((q) => { const r = d.find((x) => `${x.backend}|${x.diff_method}|${x.device}` === k && x.n_qubits === q); return r ? r.step_time * 1000 : null; }) })),
  }), true);
}

["vis-ds", "vis-q"].forEach((id) => $(id).addEventListener("change", renderVision));
["sw-ds", "sw-model"].forEach((id) => $(id).addEventListener("change", renderSweep));
$("ab-ds").addEventListener("change", renderAblation);

loadAll();
setInterval(loadAll, 30000);
