# US AI Patent Intelligence Dashboard

An interactive Streamlit dashboard supporting the descriptive-analysis component of a University of Trier Business Analytics project. It explores 80,566 cleaned US AI patent records and provides filters, KPIs, temporal trends, assignee rankings, technology analysis, grant-time analysis, and a filtered-data download.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app reads the included cleaned analytical dataset from `02_clean_patents.parquet`.

## Deploy on Streamlit Community Cloud

1. Create an app from this GitHub repository.
2. Select the `main` branch.
3. Set the entrypoint to `app.py`.
4. No secrets or environment variables are required.

## Scope

This repository intentionally contains only the files required to run the dashboard. Raw data, notebooks, predictive modelling work, academic reports, and unrelated project artefacts are excluded.

## Data note

The dashboard uses a cleaned analytical extract derived from public US patent records. It is intended for coursework demonstration and descriptive analysis. The 2026 period is partial, so year-over-year growth calculations use completed years through 2025.
