/* ═══════════════════════════════════════════════════════════════
   app.js — HealthRisk_AI Dashboard Logic (Clinical Theme)
   ═══════════════════════════════════════════════════════════════ */

const API = '';

// ── Chart.js Defaults ──
Chart.defaults.color = '#c0c8cc';
Chart.defaults.borderColor = 'rgba(64,72,75,0.15)';
Chart.defaults.font.family = "'Manrope', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;

// ── Colors ──
const COLORS = {
  secondary: '#40ddbe',
  primary: '#9bcee3',
  error: '#ffb4ab',
  amber: '#f59e0b',
  violet: '#afc6ff',
  surface: '#192123',
};

const RISK_COLORS = { Low: COLORS.secondary, Medium: COLORS.amber, High: COLORS.error };

const CHART_PALETTE = ['#40ddbe', '#9bcee3', '#afc6ff', '#f59e0b', '#64fada', '#7ba5ff', '#ffb4ab', '#ec4899'];

// ── State ──
let chartInstances = {};
let dataCache = {};

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initSmoothScroll();
  loadDataSummary();
  loadEDAData();
});

function initScrollReveal() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.section-reveal').forEach(el => obs.observe(el));
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const t = document.querySelector(a.getAttribute('href'));
      if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ══════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════
async function loadDataSummary() {
  try {
    const res = await fetch(`${API}/api/data-summary`);
    const data = await res.json();
    dataCache.summary = data;
    renderStatCards(data);
    renderDataPreview(data);
    renderFeatureChips(data);
  } catch (err) {
    console.error('Data summary error:', err);
  }
}

async function loadEDAData() {
  try {
    const res = await fetch(`${API}/api/eda`);
    const data = await res.json();
    dataCache.eda = data;
    renderEDACharts(data);
  } catch (err) {
    console.error('EDA error:', err);
  }
}

// ══════════════════════════════════════════════
// STAT CARDS
// ══════════════════════════════════════════════
function renderStatCards(data) {
  animateCounter(document.getElementById('stat-rows'), 0, data.rows, 1000);
  animateCounter(document.getElementById('stat-features'), 0, data.columns, 800);

  const totalMissing = Object.values(data.missing).reduce((a, b) => a + b, 0);
  document.getElementById('stat-missing').textContent = totalMissing;

  const dist = data.target_distribution;
  document.getElementById('stat-classes').textContent = Object.keys(dist).length;

  // Hero stat
  const heroStat = document.getElementById('stat-rows-hero');
  if (heroStat) heroStat.textContent = data.rows;

  renderTargetChart(dist);
}

function animateCounter(el, start, end, dur) {
  if (!el) return;
  const range = end - start;
  const t0 = performance.now();
  function tick(t) {
    const p = Math.min((t - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + range * eased);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ══════════════════════════════════════════════
// FEATURE CHIPS
// ══════════════════════════════════════════════
function renderFeatureChips(data) {
  const c = document.getElementById('feature-chips');
  if (!c) return;
  c.innerHTML = data.features.map(f =>
    `<span class="text-xs font-medium text-on-surface-variant bg-primary-container/40 border border-outline-variant/20 px-3 py-1.5 rounded-full hover:bg-primary-container transition-colors cursor-default">${f}</span>`
  ).join('');
}

// ══════════════════════════════════════════════
// DATA PREVIEW TABLE
// ══════════════════════════════════════════════
function renderDataPreview(data) {
  const c = document.getElementById('data-preview-table');
  if (!c) return;
  const stats = data.stats;
  const keys = Object.keys(stats);
  let html = '<table class="data-tbl"><thead><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th></tr></thead><tbody>';
  keys.forEach(k => {
    const s = stats[k];
    html += `<tr><td class="text-secondary font-semibold">${k}</td><td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.median}</td><td>${s.max}</td></tr>`;
  });
  html += '</tbody></table>';
  c.innerHTML = html;
}

// ══════════════════════════════════════════════
// TARGET CHART
// ══════════════════════════════════════════════
function renderTargetChart(dist) {
  const ctx = document.getElementById('target-chart');
  if (!ctx) return;
  destroyChart('target-chart');
  chartInstances['target-chart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(dist),
      datasets: [{
        data: Object.values(dist),
        backgroundColor: Object.keys(dist).map(k => RISK_COLORS[k] || COLORS.primary),
        borderColor: 'transparent',
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '68%',
      plugins: { legend: { position: 'bottom', labels: { padding: 20 } } },
    },
  });
}

// ══════════════════════════════════════════════
// EDA CHARTS
// ══════════════════════════════════════════════
function renderEDACharts(data) {
  renderCorrelationHeatmap(data.correlation);
  renderDistributions(data.distributions);
  renderBoxPlots(data.box_data);
  renderGenderChart(data.gender_distribution);
}

function renderCorrelationHeatmap(corr) {
  const canvas = document.getElementById('correlation-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const labels = corr.labels, values = corr.values, n = labels.length;
  const cellSize = Math.min(40, (canvas.parentElement.clientWidth - 100) / n);
  const margin = { top: 10, right: 10, bottom: 100, left: 100 };
  const w = cellSize * n + margin.left + margin.right;
  const h = cellSize * n + margin.top + margin.bottom;
  canvas.width = w * 2; canvas.height = h * 2;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  ctx.scale(2, 2);

  function getColor(val) {
    if (val >= 0) {
      const t = val;
      return `rgba(${Math.round(64 + 91 * t)}, ${Math.round(221 - 80 * t)}, ${Math.round(190 - 100 * t)}, ${0.3 + 0.7 * Math.abs(val)})`;
    } else {
      return `rgba(${Math.round(155 + 100 * Math.abs(val))}, ${Math.round(206 - 80 * Math.abs(val))}, ${Math.round(227 - 60 * Math.abs(val))}, ${0.3 + 0.7 * Math.abs(val)})`;
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = margin.left + j * cellSize, y = margin.top + i * cellSize;
      ctx.fillStyle = getColor(values[i][j]);
      ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
      if (cellSize >= 28) {
        ctx.fillStyle = Math.abs(values[i][j]) > 0.5 ? '#fff' : '#c0c8cc';
        ctx.font = `${Math.max(8, cellSize / 4)}px Manrope`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(values[i][j].toFixed(1), x + cellSize / 2, y + cellSize / 2);
      }
    }
  }
  ctx.fillStyle = '#8a9296'; ctx.font = '10px Manrope';
  labels.forEach((l, i) => {
    ctx.save();
    ctx.translate(margin.left + i * cellSize + cellSize / 2, margin.top + n * cellSize + 8);
    ctx.rotate(-Math.PI / 4); ctx.textAlign = 'right'; ctx.fillText(l, 0, 0);
    ctx.restore();
  });
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  labels.forEach((l, i) => ctx.fillText(l, margin.left - 6, margin.top + i * cellSize + cellSize / 2));
}

function renderDistributions(distributions) {
  const c = document.getElementById('distribution-charts');
  if (!c) return;
  c.innerHTML = '';
  const feats = ['age', 'bmi', 'blood_pressure_systolic', 'cholesterol', 'blood_glucose', 'heart_rate'];

  feats.forEach((feat, idx) => {
    if (!distributions[feat]) return;
    const d = distributions[feat];
    const card = document.createElement('div');
    card.className = 'dash-card rounded-xl p-6';
    card.innerHTML = `
      <div class="flex items-center gap-2 mb-4">
        <span class="material-symbols-outlined text-secondary text-lg">bar_chart</span>
        <span class="text-sm font-bold">${formatLabel(feat)}</span>
      </div>
      <div class="chart-box" style="height: 220px;"><canvas id="dist-${feat}"></canvas></div>
    `;
    c.appendChild(card);

    setTimeout(() => {
      const ctx = document.getElementById(`dist-${feat}`);
      const labels = d.bins.slice(0, -1).map((b, i) => `${b.toFixed(0)}-${d.bins[i + 1].toFixed(0)}`);
      chartInstances[`dist-${feat}`] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: d.counts,
            backgroundColor: hexToRgba(CHART_PALETTE[idx % CHART_PALETTE.length], 0.6),
            borderColor: CHART_PALETTE[idx % CHART_PALETTE.length],
            borderWidth: 1, borderRadius: 4,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { ticks: { maxRotation: 45, font: { size: 10 } } }, y: { beginAtZero: true } },
        },
      });
    }, 100 + idx * 50);
  });
}

function renderBoxPlots(boxData) {
  const ctx = document.getElementById('boxplot-chart');
  if (!ctx || !boxData) return;
  const feats = Object.keys(boxData), levels = ['Low', 'Medium', 'High'];
  destroyChart('boxplot-chart');
  chartInstances['boxplot-chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: feats.map(formatLabel),
      datasets: levels.map(level => ({
        label: level,
        data: feats.map(f => boxData[f][level]?.median || 0),
        backgroundColor: hexToRgba(RISK_COLORS[level], 0.7),
        borderColor: RISK_COLORS[level],
        borderWidth: 1, borderRadius: 4,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true } },
    },
  });
}

function renderGenderChart(dist) {
  const ctx = document.getElementById('gender-chart');
  if (!ctx || !dist || Object.keys(dist).length === 0) return;
  destroyChart('gender-chart');
  chartInstances['gender-chart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(dist),
      datasets: [{
        data: Object.values(dist),
        backgroundColor: [COLORS.primary, COLORS.secondary],
        borderWidth: 0, hoverOffset: 6,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom' } } },
  });
}

// ══════════════════════════════════════════════
// PIPELINE
// ══════════════════════════════════════════════
async function runPipeline() {
  const btn = document.getElementById('btn-run-pipeline');
  const overlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  const loadingSub = document.getElementById('loading-sub');
  const heroStatus = document.getElementById('hero-status');

  btn.style.opacity = '0.6'; btn.style.pointerEvents = 'none';
  overlay.classList.add('visible');
  loadingText.textContent = 'Initializing Pipeline...';
  if (heroStatus) heroStatus.textContent = 'RUNNING';

  const pollId = setInterval(async () => {
    try {
      const res = await fetch(`${API}/api/pipeline-status`);
      const s = await res.json();
      loadingText.textContent = s.message || 'Processing...';
      loadingSub.textContent = `Step ${s.step} of ${s.total}`;
      updatePipelineFlow(s.step);
    } catch (e) {}
  }, 1000);

  try {
    const res = await fetch(`${API}/api/run-pipeline`, { method: 'POST' });
    const result = await res.json();
    clearInterval(pollId);
    overlay.classList.remove('visible');
    btn.style.opacity = '1'; btn.style.pointerEvents = 'auto';

    if (result.status === 'success') {
      updatePipelineFlow(5);
      if (heroStatus) heroStatus.textContent = 'COMPLETE';
      showToast('Pipeline completed successfully! 🎉', 'success');
      loadModelResults();
    } else {
      if (heroStatus) heroStatus.textContent = 'ERROR';
      showToast(`Pipeline failed: ${result.error}`, 'error');
    }
  } catch (err) {
    clearInterval(pollId);
    overlay.classList.remove('visible');
    btn.style.opacity = '1'; btn.style.pointerEvents = 'auto';
    if (heroStatus) heroStatus.textContent = 'ERROR';
    showToast(`Error: ${err.message}`, 'error');
  }
}

function updatePipelineFlow(step) {
  document.querySelectorAll('.pipeline-step').forEach((s, i) => {
    s.classList.remove('completed', 'active');
    if (i + 1 < step) s.classList.add('completed');
    else if (i + 1 === step) s.classList.add('active');
  });
  document.querySelectorAll('.pipeline-connector').forEach((c, i) => {
    c.classList.toggle('completed', i + 1 < step);
  });
}

// ══════════════════════════════════════════════
// MODEL RESULTS
// ══════════════════════════════════════════════
async function loadModelResults() {
  try {
    const res = await fetch(`${API}/api/model-results`);
    if (!res.ok) return;
    const data = await res.json();
    dataCache.models = data;

    renderModelTable(data.comparison, data.best_model);
    renderModelBarChart(data.comparison);
    renderFeatureImportance(data.feature_importance);

    document.getElementById('models').style.display = 'block';
    document.getElementById('predict').style.display = 'block';

    // Scroll to results
    setTimeout(() => {
      document.getElementById('models').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 400);
  } catch (err) {
    console.error('Model results error:', err);
  }
}

function renderModelTable(comparison, bestModel) {
  const tbody = document.getElementById('model-table-body');
  if (!tbody) return;
  tbody.innerHTML = comparison.map(m => {
    const isBest = m.Model.toLowerCase() === bestModel?.toLowerCase();
    const accPct = (m.Accuracy * 100).toFixed(1);
    return `
      <tr class="border-b border-outline-variant/10">
        <td class="py-3 px-3">
          <span class="font-bold">${m.Model}</span>
          ${isBest ? '<span class="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-secondary/10 text-secondary text-[10px] font-bold uppercase"><i class="fas fa-crown text-[8px]"></i> Best</span>' : ''}
        </td>
        <td class="py-3 px-3">
          <div class="flex items-center gap-3">
            <div class="accuracy-track"><div class="accuracy-fill" style="width:${accPct}%"></div></div>
            <span class="text-xs font-bold font-mono w-12 text-right">${accPct}%</span>
          </div>
        </td>
        <td class="py-3 px-3 font-bold font-mono text-xs">${(m['F1 (weighted)'] * 100).toFixed(1)}%</td>
      </tr>
    `;
  }).join('');
}

function renderModelBarChart(comparison) {
  const ctx = document.getElementById('model-comparison-chart');
  if (!ctx) return;
  destroyChart('model-comparison-chart');
  chartInstances['model-comparison-chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: comparison.map(m => m.Model),
      datasets: [{
        label: 'Accuracy',
        data: comparison.map(m => +(m.Accuracy * 100).toFixed(1)),
        backgroundColor: comparison.map((_, i) => hexToRgba(CHART_PALETTE[i % CHART_PALETTE.length], 0.7)),
        borderColor: comparison.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        borderWidth: 1.5, borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `Accuracy: ${c.raw}%` } } },
      scales: { x: { beginAtZero: true, max: 100, title: { display: true, text: 'Accuracy (%)' } } },
    },
  });
}

function renderFeatureImportance(fiData) {
  if (!fiData || Object.keys(fiData).length === 0) return;
  const ctx = document.getElementById('feature-importance-chart');
  if (!ctx) return;
  const modelName = Object.keys(fiData)[0];
  const fi = fiData[modelName];
  const pairs = fi.features.map((f, i) => ({ feature: f, importance: fi.importances[i] }));
  pairs.sort((a, b) => b.importance - a.importance);

  destroyChart('feature-importance-chart');
  chartInstances['feature-importance-chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pairs.map(p => formatLabel(p.feature)),
      datasets: [{
        label: 'Importance',
        data: pairs.map(p => p.importance),
        backgroundColor: pairs.map((_, i) => hexToRgba(CHART_PALETTE[i % CHART_PALETTE.length], 0.7)),
        borderColor: pairs.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        borderWidth: 1, borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

// ══════════════════════════════════════════════
// PREDICTION
// ══════════════════════════════════════════════
async function submitPrediction() {
  const form = document.getElementById('predict-form');
  const formData = new FormData(form);
  const data = {};
  for (let [k, v] of formData.entries()) data[k] = parseFloat(v) || 0;
  if (form.querySelector('[name="gender"]')) {
    data.gender = form.querySelector('[name="gender"]').value === 'Male' ? 1 : 0;
  }

  const btn = document.getElementById('btn-predict');
  btn.style.opacity = '0.6'; btn.style.pointerEvents = 'none';

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
    });
    const result = await res.json();
    btn.style.opacity = '1'; btn.style.pointerEvents = 'auto';

    if (result.error) { showToast(result.error, 'error'); return; }
    renderPredictionResult(result);
  } catch (err) {
    btn.style.opacity = '1'; btn.style.pointerEvents = 'auto';
    showToast(`Prediction failed: ${err.message}`, 'error');
  }
}

function renderPredictionResult(result) {
  const c = document.getElementById('prediction-result');
  c.classList.remove('hidden');

  const level = result.prediction;
  const cls = level.toLowerCase();

  const levelEl = document.getElementById('result-level');
  levelEl.textContent = `${level} Risk`;
  levelEl.style.color = RISK_COLORS[level] || '#dce4e6';

  const icon = document.getElementById('result-icon');
  icon.className = cls === 'low' ? 'fas fa-shield-alt text-3xl' :
                   cls === 'medium' ? 'fas fa-exclamation-triangle text-3xl' :
                   'fas fa-heart-pulse text-3xl';
  icon.style.color = RISK_COLORS[level] || '#fff';

  const probC = document.getElementById('prob-bars');
  if (result.probabilities && Object.keys(result.probabilities).length > 0) {
    probC.innerHTML = Object.entries(result.probabilities).map(([cls, prob]) => {
      const pct = (prob * 100).toFixed(1);
      return `
        <div class="flex items-center gap-3">
          <span class="w-16 text-xs font-bold text-on-surface-variant">${cls}</span>
          <div class="prob-track"><div class="prob-fill ${cls.toLowerCase()}" style="width:${pct}%"></div></div>
          <span class="w-12 text-right text-xs font-bold font-mono">${pct}%</span>
        </div>
      `;
    }).join('');
    probC.style.display = 'flex'; probC.style.flexDirection = 'column'; probC.style.gap = '10px';
  } else {
    probC.style.display = 'none';
  }
}

// ══════════════════════════════════════════════
// TABS
// ══════════════════════════════════════════════
function switchTab(group, name) {
  const grp = document.querySelector(`[data-tab-group="${group}"]`);
  if (!grp) return;
  grp.querySelectorAll('.tab-btn-dash').forEach(b => b.classList.remove('active'));
  grp.querySelector(`[data-tab="${name}"]`).classList.add('active');
  document.querySelectorAll(`[data-tab-content-group="${group}"]`).forEach(c => {
    c.classList.toggle('active', c.dataset.tabContent === name);
  });
}

// ══════════════════════════════════════════════
// UTILS
// ══════════════════════════════════════════════
function destroyChart(id) { if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; } }

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function formatLabel(s) { return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.querySelector('.toast-msg').textContent = msg;
  t.className = `toast-box ${type} visible`;
  setTimeout(() => t.classList.remove('visible'), 4000);
}
