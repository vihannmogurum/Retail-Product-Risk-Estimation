/**
 * PROJECT 41 // RETAIL PRODUCT DEMAND ESTIMATION AND STOCKOUT RISK ANALYSIS-III
 * Client Application Logic (Vanilla JS + Chart.js)
 * Clean relative API calls, single-port integration, Swiss/Industrial design
 */

let productsCache = [];
let batchResultsCache = [];
let scenarioChartInstance = null;
let benchmarkChartInstance = null;
let featureChartInstance = null;

// ==================== INITIALIZATION ====================
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupRangeSlider();
  setupScenarioForm();
  setupBatchDropzone();
  setupTableFilters();

  await checkApiHealth();
  await loadProducts();
  await loadBenchmarkMetrics();
  await runDefaultBatch();
});

// ==================== NAVIGATION ====================
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

function setupRangeSlider() {
  const slider = document.getElementById("input-multiplier");
  const display = document.getElementById("multiplier-val");
  slider.addEventListener("input", (e) => {
    display.textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
  });
}

// ==================== API HEALTH ====================
async function checkApiHealth() {
  const t0 = performance.now();
  try {
    const res = await fetch("/health");
    const t1 = performance.now();
    const latency = Math.round(t1 - t0);
    document.getElementById("latency-ms").textContent = `${latency} ms`;

    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("api-status-badge");
      badge.textContent = "ONLINE // HEALTHY";
      badge.className = "status-val pill-live font-mono";
      document.getElementById("active-model-name").textContent = data.best_model_name || "WINNER";
    }
  } catch (err) {
    document.getElementById("api-status-badge").textContent = "DISCONNECTED";
    document.getElementById("api-status-badge").className = "status-val font-mono col-critical";
  }
}

// ==================== PRODUCT CATALOG & SCENARIO ====================
async function loadProducts() {
  try {
    const res = await fetch("/api/products");
    if (!res.ok) return;
    productsCache = await res.json();

    const select = document.getElementById("select-product");
    select.innerHTML = "";

    productsCache.forEach((prod) => {
      const opt = document.createElement("option");
      const pid = prod.product_id || prod.sku;
      opt.value = pid;
      opt.textContent = `${pid} — ${prod.product_name} [${prod.category}]`;
      select.appendChild(opt);
    });

    // Default selection
    if (productsCache.length > 0) {
      populateScenarioDefaults(productsCache[0]);
    }

    select.addEventListener("change", (e) => {
      const selected = productsCache.find((p) => (p.product_id || p.sku) === e.target.value);
      if (selected) populateScenarioDefaults(selected);
    });

    // Run initial scenario prediction
    executeScenarioPrediction();
  } catch (err) {
    console.error("Failed to load products:", err);
  }
}

function populateScenarioDefaults(prod) {
  document.getElementById("input-price").value = prod.base_price.toFixed(2);
  document.getElementById("input-competitor-price").value = (prod.base_price * 1.02).toFixed(2);
  document.getElementById("input-lead-time").value = prod.lead_time_days || 7;
  document.getElementById("input-current-stock").value = prod.recent_inventory || 45;
  document.getElementById("input-date").value = new Date().toISOString().split("T")[0];
}

function setupScenarioForm() {
  const form = document.getElementById("scenario-form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    executeScenarioPrediction();
  });
}

async function executeScenarioPrediction() {
  const btn = document.getElementById("btn-recalculate");
  btn.disabled = true;
  btn.textContent = "COMPUTING RISK...";

  const payload = {
    product_id: document.getElementById("select-product").value,
    date: document.getElementById("input-date").value,
    lead_time_days: parseInt(document.getElementById("input-lead-time").value, 10),
    current_stock: parseInt(document.getElementById("input-current-stock").value, 10),
    demand_multiplier: parseFloat(document.getElementById("input-multiplier").value),
    price: parseFloat(document.getElementById("input-price").value),
    competitor_price: parseFloat(document.getElementById("input-competitor-price").value),
    promotion: document.getElementById("check-promotion").checked ? 1 : 0,
    holiday: document.getElementById("check-holiday").checked ? 1 : 0,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      renderScenarioResults(data);
    }
  } catch (err) {
    console.error("Scenario execution error:", err);
  } finally {
    btn.disabled = false;
    btn.textContent = "RUN PREDICTIVE RISK ENGINE";
  }
}

function renderScenarioResults(data) {
  // Exact Three Tiers Styling: CRITICAL, WARNING / REORDER SOON, HEALTHY
  const banner = document.getElementById("risk-banner");
  const badge = document.getElementById("risk-badge-text");
  const summary = document.getElementById("risk-summary-text");

  banner.className = "risk-banner";
  badge.textContent = data.risk_tier;

  if (data.risk_tier === "CRITICAL") {
    banner.classList.add("tier-critical");
    summary.textContent = `STOCKOUT RISK CRITICAL // Replenishment required within ${data.lead_time_days} days.`;
  } else if (data.risk_tier.includes("WARNING")) {
    banner.classList.add("tier-warning");
    summary.textContent = `REORDER REQUIRED SOON // Stock below Reorder Point (${data.reorder_point} units).`;
  } else {
    banner.classList.add("tier-healthy");
    summary.textContent = `HEALTHY STOCK BUFFER // Operating smoothly above safety thresholds.`;
  }

  // Update KPI Tiles
  document.getElementById("kpi-daily-demand").textContent = data.predicted_daily_demand.toFixed(1);
  document.getElementById("kpi-7d-demand").textContent = data.predicted_7day_demand.toFixed(0);
  document.getElementById("kpi-safety-stock").textContent = data.safety_stock;
  document.getElementById("kpi-reorder-point").textContent = data.reorder_point;
  document.getElementById("kpi-reorder-qty").textContent = data.reorder_qty;
  document.getElementById("kpi-runway").textContent = `${data.runway_days}d`;

  // Rationale
  document.getElementById("rationale-content").textContent = data.rationale;

  // Chart rendering
  renderScenarioChart(data);
}

function renderScenarioChart(data) {
  const ctx = document.getElementById("scenarioComparisonChart").getContext("2d");
  if (scenarioChartInstance) scenarioChartInstance.destroy();

  const currentStock = data.current_stock;
  const ltd = data.lead_time_demand;
  const rop = data.reorder_point;
  const safetyStock = data.safety_stock;

  scenarioChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Current Stock", "Lead Time Demand", "Safety Stock (SS)", "Reorder Point (ROP)"],
      datasets: [
        {
          label: "Inventory Units",
          data: [currentStock, ltd, safetyStock, rop],
          backgroundColor: [
            currentStock < ltd ? "#B91C1C" : currentStock < rop ? "#B45309" : "#15803D",
            "#64748B",
            "#94A3B8",
            "#1D4ED8",
          ],
          borderColor: "#0F172A",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y} units`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: "#E2E8F0" },
          ticks: { color: "#64748B", font: { family: "JetBrains Mono" } },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#0F172A", font: { family: "Inter", weight: 600 } },
        },
      },
    },
  });
}

// ==================== BATCH CSV MATRIX ====================
function setupBatchDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("csv-file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleCsvUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCsvUpload(e.target.files[0]);
    }
  });

  document.getElementById("btn-download-template").addEventListener("click", () => {
    window.location.href = "/api/export/batch-template";
  });

  document.getElementById("btn-export-results").addEventListener("click", exportBatchCsv);
}

async function handleCsvUpload(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/batch-predict", {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const data = await res.json();
      renderBatchResults(data);
    } else {
      alert("Batch CSV processing failed. Check file format.");
    }
  } catch (err) {
    console.error("Batch upload error:", err);
  }
}

async function runDefaultBatch() {
  try {
    const res = await fetch("/api/batch-predict", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      renderBatchResults(data);
    }
  } catch (err) {
    console.error("Default batch error:", err);
  }
}

function renderBatchResults(data) {
  batchResultsCache = data.results || [];
  const s = data.summary || {};

  document.getElementById("batch-total").textContent = s.total_items || batchResultsCache.length;
  document.getElementById("batch-critical").textContent = s.critical_count || 0;
  document.getElementById("batch-warning").textContent = s.warning_count || 0;
  document.getElementById("batch-healthy").textContent = s.healthy_count || 0;
  document.getElementById("batch-reorder-units").textContent = s.total_recommended_reorder_units || 0;

  updateFilterPillCounts(s);
  renderBatchTable(batchResultsCache);
}

function updateFilterPillCounts(s) {
  const total = s.total_items || batchResultsCache.length;
  const crit = s.critical_count || 0;
  const warn = s.warning_count || 0;
  const heal = s.healthy_count || 0;

  document.querySelector('[data-filter="ALL"]').textContent = `ALL (${total})`;
  document.querySelector('[data-filter="CRITICAL"]').textContent = `CRITICAL (${crit})`;
  document.querySelector('[data-filter="WARNING"]').textContent = `WARNING / REORDER SOON (${warn})`;
  document.querySelector('[data-filter="HEALTHY"]').textContent = `HEALTHY (${heal})`;
}

function setupTableFilters() {
  const pills = document.querySelectorAll(".pill-btn");
  pills.forEach((p) => {
    p.addEventListener("click", () => {
      pills.forEach((btn) => btn.classList.remove("active"));
      p.classList.add("active");
      applyTableFiltering();
    });
  });

  const searchInput = document.getElementById("table-search-input");
  searchInput.addEventListener("input", applyTableFiltering);
}

function applyTableFiltering() {
  const activePill = document.querySelector(".pill-btn.active").dataset.filter;
  const query = document.getElementById("table-search-input").value.toLowerCase().trim();

  let filtered = batchResultsCache;

  if (activePill === "CRITICAL") {
    filtered = filtered.filter((r) => r.risk_tier === "CRITICAL");
  } else if (activePill === "WARNING") {
    filtered = filtered.filter((r) => r.risk_tier.includes("WARNING"));
  } else if (activePill === "HEALTHY") {
    filtered = filtered.filter((r) => r.risk_tier === "HEALTHY");
  }

  if (query) {
    filtered = filtered.filter(
      (r) =>
        r.product_id.toLowerCase().includes(query) ||
        r.sku_id.toLowerCase().includes(query) ||
        r.product_name.toLowerCase().includes(query) ||
        r.category.toLowerCase().includes(query) ||
        r.risk_tier.toLowerCase().includes(query)
    );
  }

  renderBatchTable(filtered);
}

function renderBatchTable(rows) {
  const tbody = document.getElementById("batch-table-body");
  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-state">No matching products found.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows
    .map((r) => {
      let badgeClass = "badge-healthy";
      if (r.risk_tier === "CRITICAL") badgeClass = "badge-critical";
      else if (r.risk_tier.includes("WARNING")) badgeClass = "badge-warning";

      return `
      <tr>
        <td><strong>${r.product_id}</strong> <span style="color:#64748B">[${r.sku_id}]</span><br><small>${r.product_name}</small></td>
        <td>${r.category}</td>
        <td><strong>${r.current_stock}</strong></td>
        <td>${r.predicted_daily_demand.toFixed(1)}</td>
        <td>${r.predicted_7day_demand.toFixed(0)}</td>
        <td>${r.safety_stock}</td>
        <td><strong>${r.reorder_point}</strong></td>
        <td style="color:${r.reorder_qty > 0 ? '#B45309' : '#15803D'}; font-weight:700;">${r.reorder_qty}</td>
        <td>${r.runway_days}d</td>
        <td><span class="tier-badge ${badgeClass}">${r.risk_tier}</span></td>
      </tr>
    `;
    })
    .join("");
}

function exportBatchCsv() {
  if (!batchResultsCache || batchResultsCache.length === 0) {
    alert("No batch data to export.");
    return;
  }

  const headers = [
    "product_id",
    "sku_id",
    "product_name",
    "category",
    "date",
    "current_stock",
    "predicted_daily_demand",
    "predicted_7day_demand",
    "safety_stock",
    "reorder_point",
    "reorder_qty",
    "runway_days",
    "risk_tier",
    "rationale",
  ];

  const rows = batchResultsCache.map((r) => [
    r.product_id,
    r.sku_id,
    `"${r.product_name.replace(/"/g, '""')}"`,
    r.category,
    r.date,
    r.current_stock,
    r.predicted_daily_demand,
    r.predicted_7day_demand,
    r.safety_stock,
    r.reorder_point,
    r.reorder_qty,
    r.runway_days,
    `"${r.risk_tier}"`,
    `"${(r.rationale || "").replace(/"/g, '""')}"`,
  ]);

  const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `retail_stockout_risk_evaluation_${new Date().toISOString().split("T")[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==================== 3-MODEL BENCHMARK ====================
async function loadBenchmarkMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();

    renderBenchmarkCards(data);
    renderBenchmarkCharts(data);
  } catch (err) {
    console.error("Benchmark loading error:", err);
  }
}

function renderBenchmarkCards(data) {
  const container = document.getElementById("benchmark-cards");
  container.innerHTML = "";

  const models = data.models || {};
  const winner = data.best_model || "";

  Object.entries(models).forEach(([name, metrics]) => {
    const isWinner = name === winner;
    const card = document.createElement("div");
    card.className = `bench-card ${isWinner ? "winner" : ""}`;

    card.innerHTML = `
      ${isWinner ? '<span class="winner-tag">BEST PERFORMER</span>' : ""}
      <h3 class="bench-title">${name}</h3>
      <div class="bench-metrics">
        <div class="bm-item">
          <span class="bm-lbl">MAE (HOLDOUT)</span>
          <span class="bm-val">${metrics.mae.toFixed(2)}</span>
        </div>
        <div class="bm-item">
          <span class="bm-lbl">RMSE (HOLDOUT)</span>
          <span class="bm-val">${metrics.rmse.toFixed(2)}</span>
        </div>
        <div class="bm-item">
          <span class="bm-lbl">R² SCORE</span>
          <span class="bm-val">${metrics.r2.toFixed(4)}</span>
        </div>
        <div class="bm-item">
          <span class="bm-lbl">WAPE ERROR</span>
          <span class="bm-val">${(metrics.wape * 100).toFixed(2)}%</span>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderBenchmarkCharts(data) {
  const models = data.models || {};
  const names = Object.keys(models);
  const maes = names.map((n) => models[n].mae);
  const rmses = names.map((n) => models[n].rmse);

  // 1. Error comparison
  const ctx1 = document.getElementById("modelComparisonChart").getContext("2d");
  if (benchmarkChartInstance) benchmarkChartInstance.destroy();

  benchmarkChartInstance = new Chart(ctx1, {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        {
          label: "MAE (Units)",
          data: maes,
          backgroundColor: "#1D4ED8",
        },
        {
          label: "RMSE (Units)",
          data: rmses,
          backgroundColor: "#64748B",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { font: { family: "Inter", weight: 600 }, color: "#0F172A" },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: "#E2E8F0" },
          ticks: { color: "#64748B", font: { family: "JetBrains Mono" } },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#0F172A", font: { family: "Inter", weight: 600 } },
        },
      },
    },
  });

  // 2. Feature importances
  const importances = (data.feature_importances || []).slice(0, 8);
  const fLabels = importances.map((f) => f.feature);
  const fValues = importances.map((f) => (f.importance * 100).toFixed(1));

  const ctx2 = document.getElementById("featureImportanceChart").getContext("2d");
  if (featureChartInstance) featureChartInstance.destroy();

  featureChartInstance = new Chart(ctx2, {
    type: "bar",
    indexAxis: "y",
    data: {
      labels: fLabels,
      datasets: [
        {
          label: "Relative Importance (%)",
          data: fValues,
          backgroundColor: "#3B82F6",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: "#E2E8F0" },
          ticks: { color: "#64748B", font: { family: "JetBrains Mono" } },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#0F172A", font: { family: "JetBrains Mono" } },
        },
      },
    },
  });
}
