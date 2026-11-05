const apiBase = (window.SALES_API_BASE || "").replace(/\/$/, "");
const categoryInput = document.querySelector("#category");
const modelInput = document.querySelector("#model");
const horizonInput = document.querySelector("#horizon");
const horizonOutput = document.querySelector("#horizon-value");
const toast = document.querySelector("#toast");
let toastTimer;

const dollars = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);

async function request(path) {
  const response = await fetch(`${apiBase}${path}`);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(
      detail.detail || `Forecast service returned ${response.status}`,
    );
  }
  return response.json();
}

function showError(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 6500);
}

function setLoading(loading) {
  document
    .querySelector("#forecast-chart")
    .classList.toggle("loading", loading);
  if (loading)
    document.querySelector("#forecast-chart").textContent =
      "Updating weekly outlook...";
}

function renderMetrics(metrics) {
  const selectedName =
    modelInput.value === "best" ? metrics.best_model : modelInput.value;
  const selected = metrics.models[selectedName];
  const naiveMae = metrics.models.Naive.mae;
  const improvement = naiveMae
    ? ((naiveMae - selected.mae) / naiveMae) * 100
    : 0;
  document.querySelector("#chart-title").textContent =
    categoryInput.value === "All"
      ? "Overall sales"
      : `${categoryInput.value} sales`;
  document.querySelector("#model-badge").textContent =
    modelInput.value === "best"
      ? `${selectedName} / best on holdout`
      : selectedName;
  document.querySelector("#mae").textContent = dollars(selected.mae);
  document.querySelector("#rmse").textContent = dollars(selected.rmse);
  document.querySelector("#mae-note").textContent =
    `${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)}% vs seasonal naive`;
  document.querySelector("#vs-naive").textContent =
    `${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)}%`;
  document.querySelector("#test-weeks").textContent = metrics.test_weeks;

  const rows = [
    ["Prophet", metrics.models.Prophet],
    ["XGBoost", metrics.models.XGBoost],
    ["Seasonal naive", metrics.models.Naive],
  ];
  document.querySelector("#comparison-body").innerHTML = rows
    .map(([name, value]) => {
      const isBest = name === metrics.best_model;
      const readout =
        name === metrics.best_model ? "Best on holdout" : "Reference";
      return `<tr class="${isBest ? "best-row" : ""}"><td>${name}</td><td>${dollars(value.mae)}</td><td>${dollars(value.rmse)}</td><td>${readout}</td></tr>`;
    })
    .join("");
  return selectedName;
}

function renderChart(payload) {
  const historyX = payload.history.map((point) => point.ds);
  const historyY = payload.history.map((point) => point.y);
  const future = payload.forecast;
  const traces = [
    {
      x: historyX,
      y: historyY,
      type: "scatter",
      mode: "lines",
      name: "Observed sales",
      line: { color: "#176b5b", width: 2 },
    },
  ];
  if (
    payload.model === "Prophet" &&
    future.some((point) => point.yhat_lower !== null)
  ) {
    traces.push({
      x: future.map((point) => point.ds),
      y: future.map((point) => point.yhat_upper),
      type: "scatter",
      mode: "lines",
      line: { width: 0 },
      showlegend: false,
      hoverinfo: "skip",
    });
    traces.push({
      x: future.map((point) => point.ds),
      y: future.map((point) => point.yhat_lower),
      type: "scatter",
      mode: "lines",
      fill: "tonexty",
      fillcolor: "rgba(228,111,76,.17)",
      line: { width: 0 },
      name: "Prophet interval",
      hoverinfo: "skip",
    });
  }
  traces.push({
    x: future.map((point) => point.ds),
    y: future.map((point) => point.yhat),
    type: "scatter",
    mode: "lines+markers",
    name: `${payload.model} forecast`,
    line: { color: "#e46f4c", width: 2.5, dash: "dot" },
    marker: { size: 5 },
  });

  const chart = document.querySelector("#forecast-chart");
  chart.replaceChildren();
  Plotly.react(
    chart,
    traces,
    {
      autosize: true,
      height: 370,
      margin: { l: 62, r: 18, t: 12, b: 45 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#fbfcf9",
      hovermode: "x unified",
      font: { family: "DM Sans, sans-serif", color: "#334039", size: 11 },
      legend: { orientation: "h", y: 1.12, x: 0 },
      xaxis: { showgrid: false, title: null },
      yaxis: {
        gridcolor: "#e3e9e3",
        title: "Weekly sales",
        tickprefix: "$",
        tickformat: ",.0f",
      },
      shapes: [
        {
          type: "line",
          x0: historyX.at(-1),
          x1: historyX.at(-1),
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "#9ba69e", dash: "dash", width: 1 },
        },
      ],
    },
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    },
  );

  document.querySelector("#next-four").textContent = dollars(
    future.slice(0, 4).reduce((sum, point) => sum + point.yhat, 0),
  );
  document.querySelector("#period-label").textContent =
    `History + ${payload.horizon}-week outlook`;
}

async function refreshDashboard() {
  setLoading(true);
  try {
    const category = encodeURIComponent(categoryInput.value);
    const model = encodeURIComponent(modelInput.value);
    const horizon = horizonInput.value;
    const [metrics, forecast] = await Promise.all([
      request(`/api/metrics?category=${category}`),
      request(
        `/api/forecast?category=${category}&model=${model}&horizon=${horizon}`,
      ),
    ]);
    renderMetrics(metrics);
    renderChart(forecast);
  } catch (error) {
    showError(
      `${error.message}. Check that the Render API is deployed and the URL in frontend/config.js is correct.`,
    );
    document.querySelector("#forecast-chart").textContent =
      "Forecast service unavailable.";
  } finally {
    setLoading(false);
  }
}

async function start() {
  try {
    const [health, categoryData] = await Promise.all([
      request("/api/health"),
      request("/api/categories"),
    ]);
    document.querySelector("#service-dot").classList.add("online");
    document.querySelector("#service-label").textContent =
      "Forecast service online";
    categoryInput.replaceChildren(
      ...categoryData.categories.map(
        (category) => new Option(category, category),
      ),
    );
    categoryInput.disabled = false;
    await refreshDashboard();
  } catch (error) {
    document.querySelector("#service-dot").classList.add("offline");
    document.querySelector("#service-label").textContent =
      "Service unavailable";
    showError(
      `${error.message}. Check that the Render API is deployed and the URL in frontend/config.js is correct.`,
    );
  }
}

categoryInput.addEventListener("change", refreshDashboard);
modelInput.addEventListener("change", refreshDashboard);
horizonInput.addEventListener("input", () => {
  horizonOutput.value = `${horizonInput.value} weeks`;
});
horizonInput.addEventListener("change", refreshDashboard);
window.addEventListener("resize", () => Plotly.Plots.resize("forecast-chart"));
start();
