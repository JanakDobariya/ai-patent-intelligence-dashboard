# US AI Patent Intelligence Dashboard

An interactive Streamlit dashboard supporting the descriptive-analysis component of a University of Trier Business Analytics project. It explores 80,566 cleaned US AI patent records and provides filters, KPIs, temporal trends, assignee rankings, technology analysis, grant-time analysis, and a filtered-data download.

## Live demo

[Open the AI Patent Intelligence Dashboard](https://ai-patent-intelligence-dashboard.streamlit.app/)

The hosted app may take a few seconds to wake up after a period of inactivity.

## Run locally

```bash
git clone https://github.com/JanakDobariya/ai-patent-intelligence-dashboard.git
cd ai-patent-intelligence-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` if Streamlit does not open the browser automatically. On Windows, activate the environment with `.venv\Scripts\activate`.

## Offline use

After the Python packages have been installed, the dashboard can run without an internet connection. The cleaned patent dataset is included in the repository, and the app does not require an API key or external service. GitHub links and the hosted live demo still require internet access.

The app reads `02_clean_patents.parquet`, an 18-column deployment extract of
the 23-column cleaned analytical dataset used in the report. The extract keeps
all 80,566 publication records and preserves every shared value exactly; only
five research-only fields that the dashboard does not use were omitted to keep
the public deployment lightweight.

## Deploy on Streamlit Community Cloud

1. Create an app from this GitHub repository.
2. Select the `main` branch.
3. Set the entrypoint to `app.py`.
4. No secrets or environment variables are required.

## Scope

This repository intentionally contains only the files required to run the dashboard. Raw data, notebooks, predictive modelling work, academic reports, and unrelated project artefacts are excluded.

## Data note

The dashboard uses a cleaned analytical extract derived from public US patent records. It is intended for coursework demonstration and descriptive analysis. The 2026 period is partial, so year-over-year growth calculations use completed years through 2025. The patent-characteristics heatmap uses Pearson correlation, matching Part A Figure 4.5.
