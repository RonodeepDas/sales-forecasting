import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import ROOT, available_categories, weekly_sales


st.set_page_config(page_title="Northstar | Sales Forecast", page_icon="📈", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame]:
    metrics_path = ROOT / "results" / "metrics.json"
    forecasts_path = ROOT / "results" / "forecasts.csv"
    if not metrics_path.exists() or not forecasts_path.exists():
        return {}, pd.DataFrame()
    return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(forecasts_path, parse_dates=["ds"])


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --ink:#17231e; --muted:#6d7770; --line:#dce4dd; --green:#176b5b; --lime:#d7ee63; --orange:#e46f4c; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:#f4f7f2; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#e9efe9; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top:2rem; }
    .block-container { max-width:1420px; padding-top:2.6rem; padding-bottom:3rem; }
    .eyebrow { color:var(--green); font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:1px; }
    .masthead { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:30px; }
    .masthead h1 { font-size:30px; line-height:1.1; margin:5px 0 0; font-weight:600; }
    .live-tag { background:var(--lime); color:#28311f; padding:7px 10px; font:500 10px 'DM Mono',monospace; text-transform:uppercase; }
    .metric-label { color:var(--muted); font:400 11px 'DM Mono',monospace; text-transform:uppercase; }
    [data-testid="stMetric"] { border-top:2px solid var(--green); padding:12px 0 4px; }
    [data-testid="stMetricValue"] { font-size:25px; font-weight:600; }
    .section-label { border-bottom:1px solid var(--line); padding:18px 0 10px; margin:14px 0 10px; color:var(--muted); font:500 11px 'DM Mono',monospace; text-transform:uppercase; }
    .note { border-left:3px solid var(--orange); padding:9px 13px; color:#59635c; font-size:13px; line-height:1.55; }
    .metrics-table { width:100%; border-collapse:collapse; background:#fbfcfa; font-size:14px; }
    .metrics-table th { color:var(--muted); font:500 10px 'DM Mono',monospace; text-transform:uppercase; text-align:left; }
    .metrics-table th, .metrics-table td { border-bottom:1px solid var(--line); padding:11px 14px; }
    .metrics-table td:not(:first-child) { font-variant-numeric:tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

results, forecast_frame = load_results()
if not results or forecast_frame.empty:
    st.error("Forecast artifacts are missing. Run `python -m src.train` locally, then refresh the app.")
    st.stop()

categories = [name for name in available_categories() if name in results["categories"]]
with st.sidebar:
    st.markdown('<div class="eyebrow">Forecast controls</div>', unsafe_allow_html=True)
    category = st.selectbox("Category", categories)
    model_choice = st.selectbox("Model", ["Best", "Prophet", "XGBoost"])
    horizon = st.slider("Forecast horizon", min_value=4, max_value=26, value=12, step=1, format="%d weeks")
    st.divider()
    st.caption(f"Holdout window: last {results['test_weeks']} weeks")
    st.caption("Weekly sales aggregated through Sunday")

category_results = results["categories"][category]
resolved_model = category_results["best_model"] if model_choice == "Best" else model_choice
history = weekly_sales(category).set_index("ds")["y"]
forecast = forecast_frame.loc[
    (forecast_frame["Category"] == category) & (forecast_frame["Model"] == resolved_model)
].sort_values("ds").head(horizon)

st.markdown(
    f'<div class="masthead"><div><div class="eyebrow">Retail intelligence / weekly outlook</div>'
    f'<h1>Sales, in the weeks ahead.</h1></div><div class="live-tag">{resolved_model} forecast</div></div>',
    unsafe_allow_html=True,
)

mae = category_results[resolved_model]["mae"]
rmse = category_results[resolved_model]["rmse"]
baseline_mae = category_results["Naive"]["mae"]
delta = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae else 0
metric_cols = st.columns(4)
metric_cols[0].metric("Next 4 weeks", f"${forecast.head(4)['yhat'].sum():,.0f}")
metric_cols[1].metric("Forecast horizon", f"{horizon} weeks")
metric_cols[2].metric("Holdout MAE", f"${mae:,.0f}", f"{delta:+.1f}% vs naive")
metric_cols[3].metric("Holdout RMSE", f"${rmse:,.0f}")

st.markdown('<div class="section-label">Weekly sales forecast</div>', unsafe_allow_html=True)
figure = go.Figure()
figure.add_trace(go.Scatter(x=history.index, y=history.values, name="Observed sales", mode="lines", line={"color":"#176b5b","width":2}))
if resolved_model == "Prophet" and forecast["yhat_lower"].notna().any():
    figure.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line={"width":0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line={"width":0}, fill="tonexty", fillcolor="rgba(228,111,76,0.16)", name="Prophet interval", hoverinfo="skip"))
figure.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name=f"{resolved_model} forecast", mode="lines+markers", line={"color":"#e46f4c","width":2.5,"dash":"dot"}, marker={"size":5}))
figure.add_vline(x=history.index[-1], line_width=1, line_dash="dash", line_color="#9ba59d")
figure.update_layout(
    height=430, margin={"l":10,"r":18,"t":20,"b":10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfcfa",
    hovermode="x unified", legend={"orientation":"h","y":1.08,"x":0}, font={"family":"DM Sans","color":"#334039"},
    xaxis={"showgrid":False,"title":None}, yaxis={"gridcolor":"#e5ebe5","title":"Sales (USD)","tickprefix":"$","tickformat":",.0f"},
)
st.plotly_chart(figure, use_container_width=True, config={"displaylogo":False})

st.markdown('<div class="section-label">Model comparison / last 12 weeks</div>', unsafe_allow_html=True)
comparison = pd.DataFrame([
    {"Model":"Prophet", "MAE":category_results["Prophet"]["mae"], "RMSE":category_results["Prophet"]["rmse"]},
    {"Model":"XGBoost", "MAE":category_results["XGBoost"]["mae"], "RMSE":category_results["XGBoost"]["rmse"]},
    {"Model":"Seasonal naive", "MAE":baseline_mae, "RMSE":category_results["Naive"]["rmse"]},
])
comparison_html = comparison.to_html(
    index=False,
    border=0,
    classes="metrics-table",
    formatters={"MAE": "${:,.0f}".format, "RMSE": "${:,.0f}".format},
)
st.markdown(comparison_html, unsafe_allow_html=True)
st.markdown('<p class="note"><strong>How to read this:</strong> The dotted line is the selected model forecast. The shaded band shows Prophet\'s estimated interval, not a guarantee. MAE is the average absolute weekly error; lower is better. The naive baseline repeats the same week from the prior year.</p>', unsafe_allow_html=True)
st.caption("Historical data: Kaggle Superstore sample, 2014-2017. Forecasts are illustrative and are not adjusted for promotions, holidays, or structural changes.")