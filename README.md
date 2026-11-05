# Northstar Sales Forecast

Weekly retail sales forecasting with Prophet and an XGBoost lag-feature baseline, tracked in MLflow, displayed on GitHub Pages, and served by a read-only FastAPI service on Render. Streamlit remains available as a local interactive app.

**Live frontend:** [Northstar Sales Outlook](https://ronodeepdas.github.io/sales-forecasting/). **Forecast API:** [northstar-sales-api.onrender.com](https://northstar-sales-api.onrender.com) ([interactive API docs](https://northstar-sales-api.onrender.com/docs)).

## Results

Run the training pipeline to populate the comparison below and `results/metrics.json`. The benchmark uses the final 12 weeks as a chronological holdout; no rows are shuffled. The lowest-MAE configuration per model is compared with a same-week-last-year naive forecast. Since the holdout is also used to select the best configuration, treat the model ranking as exploratory rather than an unbiased final estimate.

| Category        | Prophet MAE | Prophet RMSE | XGBoost MAE | XGBoost RMSE | Naive MAE | Selected |
| --------------- | ----------: | -----------: | ----------: | -----------: | --------: | -------- |
| All             |   $7,742.44 |    $9,227.78 |   $6,312.81 |    $7,911.75 | $8,375.35 | XGBoost  |
| Furniture       |   $2,265.07 |    $3,299.78 |   $2,252.70 |    $3,252.10 | $3,279.62 | XGBoost  |
| Office Supplies |   $2,278.86 |    $2,751.10 |   $2,234.78 |    $2,883.13 | $4,029.37 | XGBoost  |
| Technology      |   $4,414.22 |    $5,939.69 |   $4,883.57 |    $6,035.01 | $4,534.82 | Prophet  |

For overall sales, XGBoost reduced MAE by $1,429.63 (18.5%) versus Prophet and by $2,062.54 (24.6%) versus the seasonal-naive baseline. It won three of the four series; Prophet performed best for Technology.

The notebook creates trend and seasonal-decomposition plots in `reports/`. Each candidate run logs its model type, configuration, train/test sizes, MAE, RMSE, and holdout forecast plot to the local `mlruns/` store.

## Data

`data/superstore.csv` is the Kaggle [Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final), which credits the Tableau [Sample Superstore](https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls). The Kaggle uploader says the data is for educational purposes and does not claim ownership. Keep that attribution with any redistribution and review the upstream terms before using this sample commercially.

The pipeline removes exact duplicate rows, parses order dates, removes rows missing date or sales, converts sales to numeric, and aggregates into continuous Sunday-ending weeks. Missing weeks are zero-filled. Category totals are optional; the app includes the overall series and each category.

## Run locally

Python 3.11 is recommended and pinned for deployment.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m src.train
streamlit run app.py
```

Training makes 16 MLflow runs (4 model configurations for each of the overall series and three categories), writes metrics and 26-week forecasts to `results/`, saves trend/decomposition figures to `reports/`, and logs per-run forecast plots to MLflow. The app reads these prepared artifacts, so category/model/horizon changes do not retrain models.

For the Pages frontend and API locally, start these in separate terminals after training:

```powershell
uvicorn backend.main:app --reload --port 8000
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`. The local frontend automatically connects to the API at port 8000.

To inspect the local experiment UI after training:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Open the notebook at `notebooks/forecasting.ipynb` to reproduce the exploration and seasonal decomposition.

## Deploy

1. The public GitHub repository is `RonodeepDas/sales-forecasting`; GitHub Pages publishes `frontend/` from **GitHub Actions** using `.github/workflows/pages.yml`.
2. The Render Blueprint in `render.yaml` deploys the free `northstar-sales-api` service, installs only `requirements-api.txt`, and checks `/api/health`. Render uses Python 3.11 from `.python-version`.
3. Keep `results/metrics.json`, `results/forecasts.csv`, and `data/superstore.csv` in the repository. Render serves prepared results and does not train models during startup.

Render's free service may sleep while idle; its first request can take longer while it wakes. The frontend reports service status and recovers when the API responds. The API only exposes read-only public forecast data. This small historical sample has no promotion, holiday, or future price signals, so treat outputs as a baseline rather than operational guidance.

## Project layout

```text
frontend/                  GitHub Pages UI
backend/main.py            Render FastAPI service
render.yaml                Render Blueprint
.github/workflows/pages.yml GitHub Pages deployment
app.py                     Optional local Streamlit UI
data/superstore.csv        Source dataset
notebooks/forecasting.ipynb Exploration notebook
src/                       Cleaning, forecasting, evaluation, and training
results/                   Metrics and prepared forecasts
reports/                   Trend and seasonal-decomposition figures
```
