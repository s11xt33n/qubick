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
  ablation: "Абляция", shots: "Shots", vision_q12: "12 кубитов (V100)", init: "Инициализация" };
const INIT_LABEL = { uniform: "uniform U[0, 2π)", small: "small N(0, 0.1²)", zero: "zero" };

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (m) => css(`--s${MODEL_SLOT[m]}`);
const fmt = (x, d = 3) => (x == null ? "—" : Number(x).toFixed(d));
const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
const uniq = (a) => [...new Set(a)];
const pos = (v) => (v != null && v > 0 ? v : null);  // для логарифмических шкал: 0 и отрицательные — пропуск
const charts = {};
const data = {};
const ENC_LABEL = { bn: "BN + (π/2)·tanh", pi2: "(π/2)·tanh", pi: "π·tanh (исходный)" };
let ENC = "bn";
try { ENC = localStorage.getItem("enc") || "bn"; } catch (e) {}

// строки сводки для выбранного encoder'а: модели без encoder'а (enc = "-") остаются всегда;
// если выбранного варианта в серии ещё нет — берём ближайший доступный
// Выбор варианта encoder'а для среза данных, который реально рисуется.
// key(r) — «точка» графика (датасет, размер выборки, ...). Выбранный пользователем
// вариант берётся, если он покрывает столько же точек, сколько самый полный;
// иначе показывается самый полный, а в метке — сколько точек выбранного уже готово.
function pickEnc(rows, section, key = (r) => r.dataset) {
  const hyb = rows.filter((r) => r.enc && r.enc !== "-" && (!r.model || r.model === "hybrid"));
  const cov = {};
  ["bn", "pi2", "pi"].forEach((e) => { cov[e] = new Set(hyb.filter((r) => r.enc === e).map(key)).size; });
  const best = Math.max(0, ...Object.values(cov));
  const e = best === 0 ? undefined : (cov[ENC] === best ? ENC : ["bn", "pi2", "pi"].find((x) => cov[x] === best));
  if (section) tagSection(section, e, cov[ENC], best);
  return { enc: e, rows: rows.filter((r) => !r.enc || r.enc === "-" || r.enc === e) };
}
// метка в заголовке эксперимента: какой encoder гибрида реально показан
function tagSection(section, e, have = 0, total = 0) {
  const h = document.querySelector(`#${section} .exp-title`);
  if (!h) return;
  let t = h.querySelector(".enc-tag");
  if (!t) { t = document.createElement("span"); t.className = "enc-tag"; h.appendChild(t); }
  if (!e) { t.textContent = "encoder: данные ещё считаются"; t.classList.add("fallback"); return; }
  t.textContent = `encoder: ${ENC_LABEL[e]}` + (e === ENC ? "" :
    have > 0 ? ` · выбранный посчитан частично (${have} из ${total})` : " · выбранный ещё считается");
  t.classList.toggle("fallback", e !== ENC);
}
function encNote(id, e) {
  const el = $(id);
  if (el && e && e !== ENC) el.dataset.encNote = `(encoder: ${ENC_LABEL[e]} — выбранный ещё считается)`;
}

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
// высота шапки меняется (одна или две строки) — от неё зависят липкие вкладки экспериментов
function syncHeader() {
  const h = document.querySelector("header.top");
  if (h) document.documentElement.style.setProperty("--hdr", h.offsetHeight + "px");
}
window.addEventListener("resize", () => { syncHeader(); Object.values(charts).forEach((c) => c.resize()); });
syncHeader();

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
    legend: { type: "scroll", top: 0, left: 0, right: 0, textStyle: { color: ink2, fontSize: 12 }, itemWidth: 14, itemHeight: 8,
      icon: "roundRect", pageIconColor: css("--ink-2"), pageTextStyle: { color: muted } },
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
  const names = ["progress", "tabular", "vision", "sweep", "ablation", "shots", "barren", "speed",
    "vision_q12", "init", "barren_init", "encoder", "figures", "server"];
  const res = await Promise.all(names.map(load));
  names.forEach((n, i) => { if (res[i]) data[n] = res[i]; });
  renderAll();
}

function renderAll() {
  const steps = [renderModels, renderProgress, renderOverview, renderTabular, renderVision,
    renderSweep, renderAblation, renderShots, renderBarren, renderSpeed, renderQ12, renderInit,
    renderEncoder, renderGallery, renderEncSwitch, renderServer];
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
    ["2–20", "кубитов в экспериментах"],
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
  const tb0 = data.tabular;
  const tb = tb0 ? { summary: pickEnc(tb0.summary).rows, tests: (tb0.tests || []).filter((t) => t.enc === pickEnc(tb0.summary).enc) } : null;
  if (tb && tb.tests.length) {
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
  const d0 = data.tabular;
  if (!d0) return;
  const pk = pickEnc(d0.summary, "tabular");
  const d = { summary: pk.rows, tests: (d0.tests || []).filter((t) => t.enc === pk.enc) };
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
  const d0 = data.vision;
  if (!d0 || !d0.summary.length) { placeholder("ch-vision", "Расчёт идёт — данные появятся здесь автоматически"); return; }
  const dss = Object.keys(DS_LABEL).filter((x) => x.startsWith("vision:") && d0.summary.some((r) => r.dataset === x));
  fillSelect("vis-ds", dss, DS_LABEL);
  const ds = $("vis-ds").value;
  // показываем только те числа кубитов, для которых уже есть результаты
  const qAvail = uniq(d0.summary.filter((r) => r.dataset === ds && r.n_qubits > 0).map((r) => r.n_qubits)).sort((a, b) => a - b);
  fillSelect("vis-q", qAvail.map(String));
  const q = +$("vis-q").value;
  const slice = d0.summary.filter((r) => r.dataset === ds && (r.n_qubits === 0 || r.n_qubits === q));
  const pk = pickEnc(slice, "vision", (r) => r.n_train);
  const d = { summary: pk.rows, tests: (d0.tests || []).filter((t) => t.enc === pk.enc) };
  const rows = d.summary;
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
  const d0 = data.sweep;
  if (!d0 || !d0.summary.length) { placeholder("ch-sweep-heat", "Расчёт идёт…"); return; }
  const d = { summary: pickEnc(d0.summary.filter((r) => r.dataset === ($("sw-ds").value || "breast_cancer")), "sweep",
    (r) => `${r.n_qubits}|${r.n_layers}`).rows.concat(d0.summary.filter((r) => r.dataset !== ($("sw-ds").value || "breast_cancer"))) };
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
      data: qs.map((q) => { const r = rows.find((x) => x.n_qubits === q && x.n_layers === L); return r ? pos(r.time_per_epoch_mean) : null; }),
      lineStyle: { width: 2, color: css(ramp[i % ramp.length]) }, itemStyle: { color: css(ramp[i % ramp.length]), borderColor: css("--surface"), borderWidth: 2 } })),
  }), true);
}

// ---------------------------------------------------------------- абляция
function renderAblation() {
  const d0 = data.ablation;
  if (!d0 || !d0.summary.length) { ["encoding", "ansatz", "reupload"].forEach((f) => placeholder(`ch-ab-${f}`, "Расчёт идёт…")); return; }
  const d = { summary: pickEnc(d0.summary, "ablation", (r) => `${r.dataset}|${r.encoding}|${r.ansatz}|${r.reupload}`).rows };
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
  const d0 = data.shots;
  if (!d0 || !d0.summary.length) return;
  const d = { summary: pickEnc(d0.summary, "shots", (r) => `${r.dataset}|${r.shots}`).rows };
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
        data: qs.map((q) => { const r = d.find((x) => x.cost === cost && x.n_layers === L && x.n_qubits === q); return r ? pos(r.grad_var) : null; }),
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
      data: qs.map((q) => { const r = d.find((x) => `${x.backend}|${x.diff_method}|${x.device}` === k && x.n_qubits === q); return r ? pos(r.step_time * 1000) : null; }) })),
  }), true);
}

// ---------------------------------------------------------------- 12 кубитов
function renderQ12() {
  const d0 = data.vision_q12;
  if (!d0 || !d0.summary.length) { placeholder("ch-q12", "Расчёт идёт на Tesla V100 — данные появятся автоматически"); return; }
  const d = { summary: pickEnc(d0.summary, "q12", (r) => `${r.dataset}|${r.n_train}`).rows };
  const v = data.vision ? { summary: data.vision.summary.filter((r) => !r.enc || r.enc === "-" || r.enc === pickEnc(d0.summary).enc) } : null;
  const dss = Object.keys(DS_LABEL).filter((x) => d.summary.some((r) => r.dataset === x));
  fillSelect("q12-ds", dss, DS_LABEL);
  const ds = $("q12-ds").value;
  const rows = [];
  if (v) v.summary.filter((r) => r.dataset === ds && r.model === "hybrid").forEach((r) => rows.push(r));
  d.summary.filter((r) => r.dataset === ds && r.model === "hybrid").forEach((r) => rows.push({ ...r, n_qubits: 12 }));
  const ns = uniq(d.summary.filter((r) => r.dataset === ds).map((r) => r.n_train)).sort((a, b) => a - b);
  const qs = uniq(rows.map((r) => r.n_qubits)).sort((a, b) => a - b);
  const ramp = ["--r250", "--r350", "--r450", "--r550", "--r650", "--r650"];
  $("q12-title").textContent = `${DS_LABEL[ds]}: Hybrid QNN — accuracy в зависимости от числа кубитов`;
  const c = chart("ch-q12");
  if (c) c.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>${ps[0].axisValue} кубитов</b><br>` +
      ps.filter((p) => p.value != null).map((p) => tipRow(p.color, fmt(p.value), p.seriesName)).join("") },
    xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
    yAxis: { ...base().yAxis, type: "value", scale: true, name: "Accuracy" },
    series: ns.map((n, i) => ({ name: `${n} примеров`, type: "line", symbolSize: 8,
      data: qs.map((q) => { const r = rows.find((x) => x.n_qubits === q && x.n_train === n); return r ? r.accuracy_mean : null; }),
      lineStyle: { width: 2, color: css(ramp[i]) }, itemStyle: { color: css(ramp[i]), borderColor: css("--surface"), borderWidth: 2 } })),
  }), true);
}

// ---------------------------------------------------------------- инициализация
function renderInit() {
  const b = data.barren_init;
  const cols = { uniform: css("--s1"), small: css("--s2"), zero: css("--s3") };
  ["local", "global"].forEach((cost) => {
    if (!b) { placeholder(`ch-binit-${cost}`, "Расчёт идёт на Tesla V100…"); return; }
    // берём самую глубокую схему, для которой уже посчитано хотя бы 3 точки
    const Ls = uniq(b.map((r) => r.n_layers)).sort((x, y) => y - x);
    const L = Ls.find((l) => b.filter((r) => r.n_layers === l && r.cost === cost).length >= 3 * uniq(b.map((r) => r.init)).length) || Ls[Ls.length - 1];
    const rows = b.filter((r) => r.cost === cost && r.n_layers === L);
    const h = $(`ch-binit-${cost}`).previousElementSibling;
    if (h) h.textContent = `Var[∂C/∂θ], ${cost === "local" ? "локальная" : "глобальная"} стоимость, L = ${L}`;
    const qs = uniq(b.map((r) => r.n_qubits)).sort((x, y) => x - y);
    const inits = uniq(rows.map((r) => r.init));
    const c = chart(`ch-binit-${cost}`);
    if (c) c.setOption(base({
      tooltip: { ...base().tooltip, trigger: "axis", formatter: (ps) => `<b>${ps[0].axisValue} кубитов</b><br>` +
        ps.filter((p) => p.value != null).map((p) => tipRow(p.color, Number(p.value).toExponential(2), p.seriesName)).join("") },
      xAxis: { ...base().xAxis, type: "category", data: qs.map(String), name: "Кубиты", nameLocation: "middle", nameGap: 26 },
      yAxis: { ...base().yAxis, type: "log", name: "Var[∂C/∂θ]", axisLabel: { color: css("--muted"), formatter: (v) => Number(v).toExponential(0) } },
      series: inits.map((it) => ({ name: INIT_LABEL[it] || it, type: "line", symbolSize: 8,
        data: qs.map((q) => { const r = rows.find((x) => x.n_qubits === q && x.init === it); return r ? pos(r.grad_var) : null; }),
        lineStyle: { width: 2, color: cols[it] }, itemStyle: { color: cols[it], borderColor: css("--surface"), borderWidth: 2 } })),
    }), true);
  });
  const d0 = data.init;
  if (!d0 || !d0.summary.length) { placeholder("ch-init", "Расчёт идёт — данные появятся автоматически"); return; }
  const d = { summary: pickEnc(d0.summary, "init", (r) => `${r.dataset}|${r.n_qubits}|${r.n_layers}|${r.init}`).rows };
  const dss = ["breast_cancer", "moons", "wine"].filter((x) => d.summary.some((r) => r.dataset === x));
  fillSelect("init-ds", dss, DS_LABEL);
  const ds = $("init-ds").value, m = $("init-model").value;
  const rows = d.summary.filter((r) => r.dataset === ds && r.model === m);
  const cfgs = uniq(rows.map((r) => `${r.n_qubits}|${r.n_layers}`)).sort();
  const lbl = (k) => { const [q, L] = k.split("|"); return `${q} кубитов, L=${L}`; };
  const c = chart("ch-init");
  if (c) c.setOption(base({
    tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: css("--grid"), opacity: 0.35 } },
      formatter: (ps) => `<b>${ps[0].axisValue}</b><br>` + ps.map((p) => {
        const r = rows.find((x) => lbl(`${x.n_qubits}|${x.n_layers}`) === p.axisValue && (INIT_LABEL[x.init] || x.init) === p.seriesName) || {};
        return tipRow(p.color, `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`, p.seriesName); }).join("") },
    xAxis: { ...base().xAxis, type: "category", data: cfgs.map(lbl) },
    yAxis: { ...base().yAxis, type: "value", scale: true, name: "Accuracy" },
    series: ["uniform", "small"].map((it) => ({ name: INIT_LABEL[it], type: "bar", barWidth: 18, barGap: "12%",
      itemStyle: { color: cols[it], borderRadius: [4, 4, 0, 0] },
      data: cfgs.map((k) => { const r = rows.find((x) => `${x.n_qubits}|${x.n_layers}` === k && x.init === it); return r ? r.accuracy_mean : null; }) })),
  }), true);
}

// ---------------------------------------------------------------- находка: encoder
function renderEncoder() {
  const d = data.encoder;
  if (!d || !d.summary.length) { placeholder("ch-encoder", "Расчёт идёт…"); return; }
  const order = ["iris", "wine", "breast_cancer", "moons", "circles", "vision:mnist", "vision:fashion", "vision:pneumonia", "vision:breast"];
  const ds = order.filter((x) => d.summary.some((r) => r.dataset === x && r.model === "hybrid"));
  const encs = ["pi", "pi2", "bn"].filter((e) => d.summary.some((r) => r.enc === e && r.model === "hybrid"));
  const cols = { pi: css("--s8"), pi2: css("--r350"), bn: css("--s1") };
  const get = (x, m, e) => d.summary.find((r) => r.dataset === x && r.model === m && r.enc === e) || {};
  const c = chart("ch-encoder");
  if (c) c.setOption(base({
    grid: { left: 48, right: 16, top: 40, bottom: 64 },
    tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: css("--grid"), opacity: 0.35 } },
      formatter: (ps) => `<b>${ps[0].axisValue}</b><br>` + ps.map((p) => {
        const r = get(ds[p.dataIndex], "hybrid", encs[p.seriesIndex]);
        return tipRow(p.color, `${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}`, p.seriesName);
      }).join("") },
    xAxis: { ...base().xAxis, type: "category", data: ds.map((x) => DS_LABEL[x]), axisLabel: { color: css("--muted"), rotate: 30, fontSize: 11 } },
    yAxis: { ...base().yAxis, type: "value", min: 0, max: 1, name: "Accuracy" },
    series: encs.map((e) => ({ name: ENC_LABEL[e], type: "bar", barMaxWidth: 16, barGap: "15%",
      itemStyle: { color: cols[e], borderRadius: [4, 4, 0, 0] }, data: ds.map((x) => get(x, "hybrid", e).accuracy_mean) })),
  }), true);
  table("enc-table", ["Набор", ...encs.map((e) => "Hybrid, " + ENC_LABEL[e]), "Bottleneck"],
    ds.map((x) => [DS_LABEL[x], ...encs.map((e) => fmt(get(x, "hybrid", e).accuracy_mean)),
      fmt((get(x, "bottleneck", encs.includes("bn") ? "bn" : encs[encs.length - 1])).accuracy_mean)]));
}

// ---------------------------------------------------------------- галерея рисунков
function renderGallery() {
  const g = $("gallery-grid"), f = data.figures;
  if (!g) return;
  g.textContent = "";
  if (!f || !f.length) { const p = document.createElement("p"); p.className = "note"; p.textContent = "Рисунки появятся после первого прогона анализа."; g.appendChild(p); return; }
  const prog = (data.progress && data.progress.series) || {};
  const when = (t) => { const d = new Date(t * 1000); return d.toLocaleString("ru", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }); };
  f.forEach((it) => {
    const a = document.createElement("a"); a.href = `figures/${it.file}?t=${it.mtime || ""}`; a.target = "_blank"; a.rel = "noopener";
    const img = document.createElement("img"); img.loading = "lazy"; img.src = `figures/${it.file}?t=${it.mtime || ""}`; img.alt = it.title;
    const s = document.createElement("span"); s.className = "g-title"; s.textContent = it.title;
    const m = document.createElement("span"); m.className = "g-meta";
    const pr = it.series && prog[it.series];
    let status = "", cls = "";
    if (pr && pr.total) {
      const done = pr.done >= pr.total;
      status = done ? "финальный" : "промежуточный";
      cls = done ? "final" : "partial";
      m.textContent = `построен ${when(it.mtime)} · по ${pr.done.toLocaleString("ru")} из ${pr.total.toLocaleString("ru")} запусков (${Math.floor(100 * pr.done / pr.total)}%)`;
    } else {
      status = "расчёт на GPU";
      cls = "partial";
      m.textContent = `построен ${when(it.mtime)}`;
    }
    const b = document.createElement("span"); b.className = `g-badge ${cls}`; b.textContent = status;
    a.append(img, s, m, b); g.appendChild(a);
  });
}

// какие варианты encoder'а уже есть в данных (по гибридной модели во всех сериях)
function encAvailability() {
  const cnt = { bn: 0, pi2: 0, pi: 0 };
  ["tabular", "vision", "vision_q12", "sweep", "ablation", "shots", "init"].forEach((k) => {
    const d = data[k];
    if (!d || !d.summary) return;
    d.summary.forEach((r) => { if (r.model === "hybrid" && cnt[r.enc] != null) cnt[r.enc] += r.runs || 1; });
  });
  return cnt;
}
function renderEncSwitch() {
  const cnt = encAvailability(), sel = $("enc");
  [...sel.options].forEach((o) => {
    const has = cnt[o.value] > 0;
    o.disabled = !has;
    o.textContent = ENC_LABEL[o.value] + (has ? "" : " — ещё считается");
  });
  const shown = cnt[ENC] > 0 ? ENC : ["bn", "pi2", "pi"].find((e) => cnt[e] > 0);
  if (shown && sel.value !== shown && !(cnt[sel.value] > 0)) sel.value = shown;
  const st = $("enc-status");
  if (st) {
    const fb = [...document.querySelectorAll(".enc-tag.fallback")].length;
    st.textContent = `Выбран encoder ${ENC_LABEL[ENC]}.` + (fb ? ` В ${fb} из ${document.querySelectorAll(".enc-tag").length} экспериментов он ещё считается — там временно показан другой вариант (см. метку у заголовка); графики обновятся автоматически.` : " Во всех экспериментах показан именно он.");
  }
}

$("enc").value = ENC;
$("enc").addEventListener("change", (e) => { ENC = e.target.value; try { localStorage.setItem("enc", ENC); } catch (x) {} renderAll(); });
["q12-ds"].forEach((id) => $(id).addEventListener("change", renderQ12));
["init-ds", "init-model"].forEach((id) => $(id).addEventListener("change", renderInit));
["vis-ds", "vis-q"].forEach((id) => $(id).addEventListener("change", renderVision));
["sw-ds", "sw-model"].forEach((id) => $(id).addEventListener("change", renderSweep));
$("ab-ds").addEventListener("change", renderAblation);


// ---------------------------------------------------------------- сервер: итоги и история
function renderServer() {
  const d = data.server, p = data.progress;
  // «живой» ли сервер: данные свежее 10 минут
  const fresh = p && p.updated && (Date.now() - new Date(p.updated.replace(" ", "T")).getTime()) < 10 * 60 * 1000;
  const st = $("live-state");
  if (st) st.textContent = fresh ? "Идут расчёты — графики «Сейчас» обновляются автоматически."
    : `Расчёты завершены${d && d.totals && d.totals.end ? " " + d.totals.end : ""}. Сервер выключен — ниже итоговая статистика прогона.`;
  const eb = $("live-eyebrow");
  if (eb) eb.lastChild.textContent = fresh ? "В реальном времени" : "Прогон завершён";
  eb?.querySelector(".live-dot")?.classList.toggle("off", !fresh);
  if (!d || !d.totals) return;
  const t = d.totals;
  const tiles = [
    [(t.runs || 0).toLocaleString("ru"), "обученных моделей"],
    [(t.barren_points || 0).toLocaleString("ru"), "точек barren plateaus (до 20 кубитов)"],
    [t.train_core_hours != null ? `${Math.round(t.train_core_hours).toLocaleString("ru")} ч` : "—", "процессорного времени на обучение"],
    [t.wall_hours != null ? `${t.wall_hours.toFixed(1)} ч` : "—", "длительность прогона"],
    [t.cpu_avg != null ? `${Math.round(t.cpu_avg)}%` : "—", "средняя загрузка CPU"],
    [t.gpu_util_avg != null ? `${Math.round(t.gpu_util_avg)}%` : "—", "средняя загрузка V100"],
    [t.gpu_temp_max != null ? `${t.gpu_temp_max} °C` : "—", "пиковая температура V100"],
    [t.gpu_energy_kwh != null ? `${t.gpu_energy_kwh} кВт·ч` : "—", "энергия V100"],
  ];
  const k = $("srv-totals");
  k.textContent = "";
  tiles.forEach(([v, l]) => {
    const e = document.createElement("div"); e.className = "kpi";
    const a1 = document.createElement("div"); a1.className = "value"; a1.textContent = v;
    const b1 = document.createElement("div"); b1.className = "label"; b1.textContent = l;
    e.append(a1, b1); k.appendChild(e);
  });
  const period = (document.querySelector("#hist-period .active") || {}).dataset?.min || "all";
  let hs;
  if (period !== "all" && Number(period) <= 60 && p && p.monitor && p.monitor.length) {
    // короткие периоды — исходные замеры каждые 15 секунд
    hs = p.monitor.slice(-Number(period) * 4).map((r) => ({ ...r, t: String(r.time).slice(-8) }));
  } else {
    const all = d.history || [];
    hs = period === "all" ? all : all.slice(-Number(period));
  }
  const xs = hs.map((r) => r.t);
  const axisX = { ...base().xAxis, type: "category", data: xs, boundaryGap: false,
    axisLabel: { color: css("--muted"), fontSize: 11, interval: Math.max(0, Math.floor(xs.length / 8)) } };
  const tip = (unit) => ({ ...base().tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: css("--muted"), width: 1 } },
    formatter: (ps) => `${ps[0].axisValue}<br>` + ps.map((q) => tipRow(q.color, q.value == null ? "—" : `${q.value}${unit}`, q.seriesName)).join("") });
  const line = (name, key, col) => ({ name, type: "line", showSymbol: false, data: hs.map((r) => r[key]),
    lineStyle: { width: 2, color: col }, itemStyle: { color: col } });
  let c = chart("ch-hist-load");
  if (c) c.setOption(base({ tooltip: tip("%"), grid: { left: 44, right: 16, top: 40, bottom: 30 }, xAxis: axisX,
    yAxis: { ...base().yAxis, type: "value", min: 0, max: 100 },
    series: [line("CPU (80 потоков)", "cpu_load", css("--s1")), line("Tesla V100", "gpu_util", css("--s2"))] }), true);
  c = chart("ch-hist-temp");
  if (c) c.setOption(base({ legend: { show: false }, tooltip: tip(" °C"), grid: { left: 40, right: 12, top: 12, bottom: 28 }, xAxis: axisX,
    yAxis: { ...base().yAxis, type: "value", scale: true },
    series: [{ ...line("Температура", "gpu_temp", css("--s8")), areaStyle: { color: css("--s8"), opacity: 0.08 } }] }), true);
  c = chart("ch-hist-power");
  if (c) c.setOption(base({ legend: { show: false }, tooltip: tip(" Вт"), grid: { left: 40, right: 12, top: 12, bottom: 28 }, xAxis: axisX,
    yAxis: { ...base().yAxis, type: "value", min: 0, max: 300 },
    series: [{ ...line("Мощность", "gpu_power", css("--s2")), areaStyle: { color: css("--s2"), opacity: 0.08 } }] }), true);
  const per = t.per_series || {};
  const keys = Object.keys(per).filter((x) => per[x] > 0).sort((x, y) => per[y] - per[x]);
  c = chart("ch-hist-series");
  if (c) c.setOption(base({ legend: { show: false }, grid: { left: 130, right: 40, top: 10, bottom: 24 },
    tooltip: { ...base().tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: css("--grid"), opacity: 0.35 } },
      formatter: (ps) => tipRow(css("--accent"), ps[0].value.toLocaleString("ru"), ps[0].axisValue) },
    xAxis: { ...base().yAxis, type: "value" },
    yAxis: { ...base().xAxis, type: "category", data: keys.map((x) => SERIES_LABEL[x] || x), inverse: true },
    series: [{ type: "bar", barMaxWidth: 16, data: keys.map((x) => per[x]), itemStyle: { color: css("--accent"), borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: "right", color: css("--ink-2"), fontSize: 11, formatter: (q) => q.value.toLocaleString("ru") } }] }), true);
}

document.querySelectorAll("#hist-period button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#hist-period button").forEach((x) => x.classList.toggle("active", x === b));
  renderServer();
}));

// ---------------------------------------------------------------- вкладки (#/страница/эксперимент)
const PAGES = ["home", "library", "results", "encoder", "data", "roadmap", "live", "gallery"];
const EXPS = ["tabular", "vision", "q12", "sweep", "ablation", "shots", "barren", "init", "speed"];
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  let page = PAGES.includes(parts[0]) ? parts[0] : "home";
  // старые якоря (#tabular и т.п.) ведут на вкладку эксперимента
  if (EXPS.includes(parts[0])) { page = "results"; parts[1] = parts[0]; }
  const exp = EXPS.includes(parts[1]) ? parts[1] : "tabular";
  PAGES.forEach((p) => $(`page-${p}`)?.classList.toggle("active", p === page));
  document.querySelectorAll("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  document.querySelectorAll("#page-results article.exp").forEach((a) => a.classList.toggle("active", a.id === exp));
  document.querySelectorAll("#exp-nav a").forEach((a) => a.classList.toggle("active", a.dataset.exp === exp));
  $("enc").closest(".enc-switch").style.visibility = ["results", "encoder", "home"].includes(page) ? "visible" : "hidden";
  window.scrollTo({ top: 0, behavior: "instant" });
  document.querySelector("#tabs a.active")?.scrollIntoView({ block: "nearest", inline: "center" });
  if (page === "results") document.querySelector("#exp-nav a.active")?.scrollIntoView({ block: "nearest", inline: "center" });
  syncHeader();
  // графики в скрытых вкладках имели нулевой размер — пересчитываем
  requestAnimationFrame(() => { renderAll(); Object.values(charts).forEach((c) => c.resize()); });
}
window.addEventListener("hashchange", route);

route();
loadAll();
setInterval(loadAll, 30000);
