# US AI Patent Intelligence Dashboard and Analysis

This repository contains the Streamlit dashboard and the original analysis files used for Parts A and B of a University of Trier Business Analytics project. The work examines 80,566 cleaned US AI patent records through descriptive analysis and predictive modelling.

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

## Project files

- `src/` contains the four Part A notebooks for the raw-data audit, cleaning, exploratory analysis, and visualisations.
- `Part_B/` contains the six Part B notebooks for target creation, preprocessing, model training, evaluation, tuning, and final interpretation.
- `Data/processed/02_clean_patents.parquet` is the complete 23-column cleaned analytical dataset.
- `Data/processed/01_part_b_target_dataset.parquet` is the modelling dataset created for Part B.
- `Data/raw/forward_citations_24m.csv` contains the forward-citation data used to construct the Part B target.
- `Data/raw/Synthetic_Data_Corruption.ipynb` records the synthetic data-quality preparation carried out before the raw-data audit.
- `Outputs/figures/` and `Outputs/tables/` contain the figures and tables produced by the original notebooks.

The root-level `02_clean_patents.parquet` is a smaller 18-column copy used only by the deployed dashboard. The full analytical file is retained under `Data/processed/`.

## Running the notebooks

The notebooks use Python with pandas, NumPy, PyArrow, Matplotlib, and scikit-learn. JupyterLab or Jupyter Notebook can be used to open them. Run the Part A notebooks in numerical order from `src/`, followed by the Part B notebooks in numerical order from `Part_B/`.

The raw annual patent downloads and `Combined_data.parquet` are not stored here because individual files exceed GitHub's normal 100 MB file limit. The duplicate CSV version of the cleaned dataset is also omitted because the equivalent Parquet file is included. As a result, the included processed data supports the exploratory analysis, visualisation, target-construction, and modelling stages, while the earliest raw ingestion and cleaning stages require the original local source files.

## Deploy on Streamlit Community Cloud

1. Create an app from this GitHub repository.
2. Select the `main` branch.
3. Set the entrypoint to `app.py`.
4. No secrets or environment variables are required.

## Scope

The repository contains the dashboard, original coursework notebooks, the smaller data files needed for the main analytical stages, and their direct outputs. Final reports, report-generation files, review notes, temporary files, and later audit or correction artefacts are intentionally excluded.

## Data note

The dashboard uses a cleaned analytical extract derived from public US patent records. It is intended for coursework demonstration and descriptive analysis. The 2026 period is partial, so year-over-year growth calculations use completed years through 2025. The patent-characteristics heatmap uses Pearson correlation, matching Part A Figure 4.5.
