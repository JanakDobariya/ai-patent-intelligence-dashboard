from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="US AI Patent Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# The deployment repository keeps only the app and its cleaned data asset.
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "02_clean_patents.parquet"


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,0.18);
        }

        .hero {
            padding: 1.55rem 1.7rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,0.22);
            background:
                linear-gradient(135deg,
                    rgba(99,102,241,0.15),
                    rgba(14,165,233,0.08),
                    rgba(16,185,129,0.08));
            margin-bottom: 1.1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.15;
        }

        .hero p {
            margin: .55rem 0 0 0;
            opacity: .82;
            font-size: 1.02rem;
        }

        .section-note {
            padding: .85rem 1rem;
            border-radius: 12px;
            border-left: 4px solid #6366f1;
            background: rgba(99,102,241,0.07);
            margin: .35rem 0 1rem 0;
        }

        .insight-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.20);
            min-height: 125px;
            background: rgba(128,128,128,0.04);
        }

        .insight-card h4 {
            margin-top: 0;
            margin-bottom: .4rem;
        }

        .small-muted {
            opacity: .68;
            font-size: .87rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.18);
            padding: .85rem 1rem;
            border-radius: 14px;
            background: rgba(128,128,128,0.025);
        }

        .footer-note {
            text-align: center;
            opacity: .60;
            font-size: .82rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    # Defensive conversions
    date_cols = [
        "publication_date",
        "filing_date",
        "grant_date",
        "priority_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "grant_year" not in df.columns and "grant_date" in df.columns:
        df["grant_year"] = df["grant_date"].dt.year

    if "grant_year" in df.columns:
        df["grant_year"] = pd.to_numeric(
            df["grant_year"], errors="coerce"
        ).astype("Int64")

    # Create convenience field only if not already present
    if (
        "filing_to_grant_years" not in df.columns
        and "filing_to_grant_days" in df.columns
    ):
        df["filing_to_grant_years"] = (
            pd.to_numeric(df["filing_to_grant_days"], errors="coerce")
            / 365.25
        )

    # Normalize common categorical fields without changing the source file
    for col in ["primary_assignee", "assignee_country", "cpc_group"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    return df


def require_columns(df: pd.DataFrame, columns: list[str], section: str) -> bool:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.warning(
            f"{section} is unavailable because these columns are missing: "
            + ", ".join(missing)
        )
        return False
    return True


def format_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(float(x))):,}"


def format_float(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{float(x):,.{digits}f}"


def technology_label(code: str) -> str:
    labels = {
        "G06N3": "G06N3 · Biological / neural models",
        "G06N5": "G06N5 · Knowledge-based models",
        "G06N7": "G06N7 · Mathematical models",
        "G06N10": "G06N10 · Quantum computing",
        "G06N20": "G06N20 · Machine learning",
    }
    return labels.get(str(code), str(code))


def safe_pct(part, whole):
    if whole in (0, None) or pd.isna(whole):
        return np.nan
    return 100 * part / whole


def top_value(series: pd.Series, exclude_unknown=True):
    s = series.dropna().astype(str)
    if exclude_unknown:
        s = s[s.str.lower() != "unknown"]
    if s.empty:
        return "—", 0
    vc = s.value_counts()
    return vc.index[0], int(vc.iloc[0])


def completed_year_growth(filtered_df: pd.DataFrame):
    """Return YoY growth for the latest completed year, excluding partial 2026."""
    if "grant_year" not in filtered_df.columns:
        return np.nan, None, None

    counts = (
        filtered_df.dropna(subset=["grant_year"])
        .groupby("grant_year")
        .size()
        .sort_index()
    )
    completed = counts[counts.index <= 2025]
    if len(completed) < 2:
        return np.nan, None, None

    current_year = int(completed.index[-1])
    previous_year = int(completed.index[-2])
    current = completed.iloc[-1]
    previous = completed.iloc[-2]

    if previous == 0:
        return np.nan, current_year, previous_year

    growth = (current / previous - 1) * 100
    return growth, current_year, previous_year


def empty_state(message="No records match the selected filters."):
    st.info(message)


# ============================================================
# LOAD
# ============================================================
if not DATA_PATH.exists():
    st.error(
        "Cleaned dataset not found.\n\n"
        f"Expected file:\n`{DATA_PATH}`\n\n"
        "Confirm that the deployment data asset is present in the repository."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if df.empty:
    st.error("The cleaned dataset is empty.")
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("🔎 Dashboard Filters")
st.sidebar.caption("All visuals update automatically.")

# Years
if "grant_year" in df.columns:
    available_years = sorted(
        [int(y) for y in df["grant_year"].dropna().unique()]
    )
else:
    available_years = []

if available_years:
    min_year, max_year = min(available_years), max(available_years)
    year_range = st.sidebar.slider(
        "Grant year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
else:
    year_range = None

# Country filter
country_options = []
if "assignee_country" in df.columns:
    country_options = sorted(
        [
            x
            for x in df["assignee_country"].dropna().astype(str).unique()
            if x != ""
        ]
    )

selected_countries = st.sidebar.multiselect(
    "Assignee country",
    options=country_options,
    default=[],
    help="Leave empty to include all countries.",
)

# Technology filter
technology_options = []
if "cpc_group" in df.columns:
    technology_options = sorted(
        [
            x
            for x in df["cpc_group"].dropna().astype(str).unique()
            if x not in ("", "Unknown")
        ]
    )

selected_technologies = st.sidebar.multiselect(
    "AI technology group",
    options=technology_options,
    default=[],
    format_func=technology_label,
    help="Leave empty to include all technology groups.",
)

exclude_unknown = st.sidebar.checkbox(
    "Exclude 'Unknown' from rankings",
    value=True,
)

top_n = st.sidebar.slider(
    "Top-N for ranking charts",
    min_value=5,
    max_value=25,
    value=10,
    step=1,
)

st.sidebar.divider()

search_text = st.sidebar.text_input(
    "Search title / assignee",
    placeholder="e.g. neural network, IBM...",
)

st.sidebar.divider()
st.sidebar.markdown("**Data source**")
st.sidebar.caption("Google Patents Public Data via BigQuery")
st.sidebar.caption("US grants · B1/B2 · G06N scope")


# ============================================================
# APPLY FILTERS
# ============================================================
filtered = df.copy()

if year_range is not None:
    filtered = filtered[
        filtered["grant_year"].between(year_range[0], year_range[1])
    ]

if selected_countries and "assignee_country" in filtered.columns:
    filtered = filtered[
        filtered["assignee_country"].isin(selected_countries)
    ]

if selected_technologies and "cpc_group" in filtered.columns:
    filtered = filtered[
        filtered["cpc_group"].isin(selected_technologies)
    ]

if search_text.strip():
    q = search_text.strip()
    mask = pd.Series(False, index=filtered.index)

    if "title" in filtered.columns:
        mask = mask | filtered["title"].astype("string").str.contains(
            q, case=False, na=False, regex=False
        )

    if "primary_assignee" in filtered.columns:
        mask = mask | filtered["primary_assignee"].astype("string").str.contains(
            q, case=False, na=False, regex=False
        )

    filtered = filtered[mask]

filtered = filtered.copy()


# ============================================================
# HERO
# ============================================================
max_grant_date = (
    df["grant_date"].max()
    if "grant_date" in df.columns and df["grant_date"].notna().any()
    else pd.NaT
)

latest_text = (
    max_grant_date.strftime("%d %B %Y")
    if pd.notna(max_grant_date)
    else "latest available record"
)

st.markdown(
    f"""
    <div class="hero">
        <h1>🧠 US AI Patent Intelligence Dashboard</h1>
        <p>
            Interactive descriptive analysis of US-granted AI-related patents
            using the G06N CPC scope. Explore innovation trends, competitors,
            technology areas and patent characteristics.
        </p>
        <p class="small-muted">
            Observation window: 2021–2026 · 2026 is partial
            (latest grant in the dataset: {latest_text})
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    empty_state()
    st.stop()


# ============================================================
# KPI ROW
# ============================================================
unique_patents = (
    filtered["publication_number"].nunique()
    if "publication_number" in filtered.columns
    else len(filtered)
)

unique_assignees = (
    filtered.loc[
        filtered["primary_assignee"].ne("Unknown"),
        "primary_assignee",
    ].nunique()
    if "primary_assignee" in filtered.columns
    else np.nan
)

unique_countries = (
    filtered.loc[
        filtered["assignee_country"].ne("Unknown"),
        "assignee_country",
    ].nunique()
    if "assignee_country" in filtered.columns
    else np.nan
)

unique_tech = (
    filtered.loc[
        filtered["cpc_group"].ne("Unknown"),
        "cpc_group",
    ].nunique()
    if "cpc_group" in filtered.columns
    else np.nan
)

median_grant_years = (
    filtered["filing_to_grant_years"].median()
    if "filing_to_grant_years" in filtered.columns
    else np.nan
)

median_backward_citations = (
    filtered["backward_citation_count"].median()
    if "backward_citation_count" in filtered.columns
    else np.nan
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Patents", format_int(unique_patents))
k2.metric("Primary assignees", format_int(unique_assignees))
k3.metric("Assignee countries", format_int(unique_countries))
k4.metric("G06N groups", format_int(unique_tech))
k5.metric("Median grant time", f"{format_float(median_grant_years, 1)} yrs")
k6.metric("Median backward citations", format_float(median_backward_citations, 0))


# ============================================================
# TABS
# ============================================================
overview_tab, competition_tab, technology_tab, characteristics_tab, explorer_tab = st.tabs(
    [
        "📊 Executive Overview",
        "🏢 Competitive Intelligence",
        "🧩 Technology Landscape",
        "🔬 Patent Characteristics",
        "🔎 Patent Explorer",
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================
with overview_tab:
    st.subheader("Executive Overview")
    st.markdown(
        """
        <div class="section-note">
            A high-level view of how AI patent activity changes over time,
            who owns the patents, where assignees are located, and which
            G06N technology groups are most represented.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])

    with left:
        if require_columns(
            filtered,
            ["grant_year", "publication_number"],
            "Patent trend",
        ):
            yearly = (
                filtered.dropna(subset=["grant_year"])
                .groupby("grant_year")["publication_number"]
                .nunique()
                .reset_index(name="patent_count")
                .sort_values("grant_year")
            )

            fig = px.line(
                yearly,
                x="grant_year",
                y="patent_count",
                markers=True,
                title="US AI patents granted by year",
                labels={
                    "grant_year": "Grant year",
                    "patent_count": "Patents",
                },
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=9),
                hovertemplate=(
                    "Grant year: %{x}<br>"
                    "Patents: %{y:,}<extra></extra>"
                ),
            )
            fig.update_xaxes(
                dtick=1,
                tickmode="linear",
            )
            fig.update_layout(
                height=410,
                margin=dict(l=10, r=10, t=55, b=10),
                hovermode="x unified",
            )

            if 2026 in yearly["grant_year"].astype(int).tolist():
                fig.add_annotation(
                    x=2026,
                    y=float(
                        yearly.loc[
                            yearly["grant_year"] == 2026,
                            "patent_count",
                        ].iloc[0]
                    ),
                    text="2026 partial",
                    showarrow=True,
                    arrowhead=2,
                    ax=-55,
                    ay=-45,
                )

            st.plotly_chart(fig, width="stretch")

    with right:
        if "cpc_group" in filtered.columns:
            tech = filtered["cpc_group"].astype(str)
            if exclude_unknown:
                tech = tech[tech != "Unknown"]

            tech_counts = (
                tech.value_counts()
                .head(6)
                .rename_axis("cpc_group")
                .reset_index(name="patent_count")
            )

            if not tech_counts.empty:
                tech_counts["technology"] = tech_counts["cpc_group"].map(
                    technology_label
                )

                fig = px.pie(
                    tech_counts,
                    names="technology",
                    values="patent_count",
                    hole=0.56,
                    title="Technology composition",
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "Patents: %{value:,}<br>"
                        "Share: %{percent}<extra></extra>"
                    ),
                )
                fig.update_layout(
                    height=410,
                    margin=dict(l=10, r=10, t=55, b=10),
                    legend_title_text="G06N group",
                )
                st.plotly_chart(fig, width="stretch")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if "primary_assignee" in filtered.columns:
            assignee_series = filtered["primary_assignee"].astype(str)
            if exclude_unknown:
                assignee_series = assignee_series[
                    assignee_series != "Unknown"
                ]

            assignee_counts = (
                assignee_series.value_counts()
                .head(top_n)
                .sort_values()
                .rename_axis("assignee")
                .reset_index(name="patent_count")
            )

            if not assignee_counts.empty:
                fig = px.bar(
                    assignee_counts,
                    x="patent_count",
                    y="assignee",
                    orientation="h",
                    title=f"Top {len(assignee_counts)} primary assignees",
                    labels={
                        "patent_count": "Patents",
                        "assignee": "",
                    },
                    text_auto=",",
                )
                fig.update_layout(
                    height=max(420, 32 * len(assignee_counts)),
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, width="stretch")

    with c2:
        if "assignee_country" in filtered.columns:
            country_series = filtered["assignee_country"].astype(str)
            if exclude_unknown:
                country_series = country_series[
                    country_series != "Unknown"
                ]

            country_counts = (
                country_series.value_counts()
                .head(top_n)
                .sort_values()
                .rename_axis("country")
                .reset_index(name="patent_count")
            )

            if not country_counts.empty:
                fig = px.bar(
                    country_counts,
                    x="patent_count",
                    y="country",
                    orientation="h",
                    title=f"Top {len(country_counts)} assignee countries",
                    labels={
                        "patent_count": "Patents",
                        "country": "",
                    },
                    text_auto=",",
                )
                fig.update_layout(
                    height=max(420, 32 * len(country_counts)),
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, width="stretch")

    # Dynamic executive insights
    st.subheader("Key signals from the current filter")

    top_assignee, top_assignee_count = (
        top_value(filtered["primary_assignee"], exclude_unknown)
        if "primary_assignee" in filtered.columns
        else ("—", 0)
    )
    top_country, top_country_count = (
        top_value(filtered["assignee_country"], exclude_unknown)
        if "assignee_country" in filtered.columns
        else ("—", 0)
    )
    top_tech, top_tech_count = (
        top_value(filtered["cpc_group"], exclude_unknown)
        if "cpc_group" in filtered.columns
        else ("—", 0)
    )

    growth, current_year, previous_year = completed_year_growth(filtered)

    i1, i2, i3, i4 = st.columns(4)

    i1.markdown(
        f"""
        <div class="insight-card">
            <h4>🏢 Leading assignee</h4>
            <b>{top_assignee}</b><br>
            <span class="small-muted">
                {format_int(top_assignee_count)} patents
                ({format_float(safe_pct(top_assignee_count, unique_patents), 1)}%
                of filtered patents)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    i2.markdown(
        f"""
        <div class="insight-card">
            <h4>🌍 Leading assignee country</h4>
            <b>{top_country}</b><br>
            <span class="small-muted">
                {format_int(top_country_count)} patents
                ({format_float(safe_pct(top_country_count, unique_patents), 1)}%
                of filtered patents)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    i3.markdown(
        f"""
        <div class="insight-card">
            <h4>🧩 Largest technology group</h4>
            <b>{technology_label(top_tech)}</b><br>
            <span class="small-muted">
                {format_int(top_tech_count)} patents
                ({format_float(safe_pct(top_tech_count, unique_patents), 1)}%
                of filtered patents)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    growth_text = (
        f"{growth:+.1f}% from {previous_year} to {current_year}"
        if pd.notna(growth)
        else "Not enough completed-year data in current filter"
    )

    i4.markdown(
        f"""
        <div class="insight-card">
            <h4>📈 Latest completed-year growth</h4>
            <b>{growth_text}</b><br>
            <span class="small-muted">
                2026 is excluded because it is a partial year.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 — COMPETITIVE INTELLIGENCE
# ============================================================
with competition_tab:
    st.subheader("Competitive Intelligence")
    st.caption(
        "Explore the primary assignees represented in the selected patent set."
    )

    if require_columns(
        filtered,
        ["primary_assignee", "grant_year", "publication_number"],
        "Competitive intelligence",
    ):
        comp_df = filtered[filtered["primary_assignee"].notna()].copy()

        if exclude_unknown:
            comp_df = comp_df[
                comp_df["primary_assignee"] != "Unknown"
            ]

        top_assignees = (
            comp_df["primary_assignee"]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )

        if not top_assignees:
            empty_state("No assignee records match the selected filters.")
        else:
            trend = (
                comp_df[
                    comp_df["primary_assignee"].isin(top_assignees)
                ]
                .groupby(["grant_year", "primary_assignee"])[
                    "publication_number"
                ]
                .nunique()
                .reset_index(name="patent_count")
            )

            fig = px.line(
                trend,
                x="grant_year",
                y="patent_count",
                color="primary_assignee",
                markers=True,
                title="Patent activity of leading primary assignees",
                labels={
                    "grant_year": "Grant year",
                    "patent_count": "Patents",
                    "primary_assignee": "Primary assignee",
                },
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=55, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.20,
                    xanchor="left",
                    x=0,
                ),
            )
            st.plotly_chart(fig, width="stretch")

            left, right = st.columns([1.2, 1])

            with left:
                ranking = (
                    comp_df[
                        comp_df["primary_assignee"].isin(top_assignees)
                    ]
                    .groupby("primary_assignee")
                    .agg(
                        patents=("publication_number", "nunique"),
                        median_inventors=("inventor_count", "median")
                        if "inventor_count" in comp_df.columns
                        else ("publication_number", "size"),
                        median_cpc_count=("cpc_count", "median")
                        if "cpc_count" in comp_df.columns
                        else ("publication_number", "size"),
                        median_backward_citations=(
                            "backward_citation_count",
                            "median",
                        )
                        if "backward_citation_count" in comp_df.columns
                        else ("publication_number", "size"),
                    )
                    .sort_values("patents", ascending=False)
                    .reset_index()
                )

                st.markdown("#### Assignee benchmark table")
                st.dataframe(
                    ranking,
                    width="stretch",
                    hide_index=True,
                )

            with right:
                selected_company = st.selectbox(
                    "Inspect a primary assignee",
                    options=top_assignees,
                    index=0,
                )

                company = comp_df[
                    comp_df["primary_assignee"] == selected_company
                ]

                st.markdown(f"#### {selected_company}")

                cc1, cc2 = st.columns(2)
                cc1.metric(
                    "Patents",
                    format_int(company["publication_number"].nunique()),
                )

                if "assignee_country" in company.columns:
                    company_country, _ = top_value(
                        company["assignee_country"],
                        exclude_unknown=True,
                    )
                    cc2.metric("Main country", company_country)

                if "cpc_group" in company.columns:
                    tech_mix = (
                        company.loc[
                            company["cpc_group"] != "Unknown",
                            "cpc_group",
                        ]
                        .value_counts()
                        .head(8)
                        .rename_axis("cpc_group")
                        .reset_index(name="patent_count")
                    )
                    if not tech_mix.empty:
                        tech_mix["technology"] = tech_mix["cpc_group"].map(
                            technology_label
                        )
                        fig = px.bar(
                            tech_mix.sort_values("patent_count"),
                            x="patent_count",
                            y="technology",
                            orientation="h",
                            title="Technology mix",
                            labels={
                                "patent_count": "Patents",
                                "technology": "",
                            },
                        )
                        fig.update_layout(
                            height=350,
                            margin=dict(l=10, r=10, t=50, b=10),
                        )
                        st.plotly_chart(fig, width="stretch")

        # Company × technology heatmap
        if (
            "cpc_group" in comp_df.columns
            and top_assignees
        ):
            heat = comp_df[
                comp_df["primary_assignee"].isin(top_assignees)
                & comp_df["cpc_group"].ne("Unknown")
            ]

            if not heat.empty:
                top_techs = (
                    heat["cpc_group"].value_counts().head(8).index
                )
                heat = heat[heat["cpc_group"].isin(top_techs)]

                pivot = pd.crosstab(
                    heat["primary_assignee"],
                    heat["cpc_group"],
                    normalize="index",
                ) * 100

                pivot = pivot.reindex(index=top_assignees).dropna(
                    how="all"
                )
                pivot.columns = [
                    technology_label(c) for c in pivot.columns
                ]

                fig = px.imshow(
                    pivot,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(
                        x="Technology group",
                        y="Primary assignee",
                        color="Share %",
                    ),
                    title=(
                        "Technology mix across selected major G06N groups "
                        "(row-normalized)"
                    ),
                )
                fig.update_layout(
                    height=max(450, 38 * len(pivot)),
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 3 — TECHNOLOGY LANDSCAPE
# ============================================================
with technology_tab:
    st.subheader("Technology Landscape")
    st.caption(
        "Compare the selected primary G06N technology groups across time, "
        "countries and patent characteristics."
    )

    if require_columns(
        filtered,
        ["cpc_group", "grant_year", "publication_number"],
        "Technology landscape",
    ):
        tech_df = filtered[filtered["cpc_group"].notna()].copy()
        if exclude_unknown:
            tech_df = tech_df[tech_df["cpc_group"] != "Unknown"]

        if tech_df.empty:
            empty_state()
        else:
            top_techs = (
                tech_df["cpc_group"]
                .value_counts()
                .head(min(top_n, 10))
                .index
                .tolist()
            )

            trend = (
                tech_df[tech_df["cpc_group"].isin(top_techs)]
                .groupby(["grant_year", "cpc_group"])[
                    "publication_number"
                ]
                .nunique()
                .reset_index(name="patent_count")
            )
            trend["technology"] = trend["cpc_group"].map(
                technology_label
            )

            fig = px.line(
                trend,
                x="grant_year",
                y="patent_count",
                color="technology",
                markers=True,
                title="Technology groups over time",
                labels={
                    "grant_year": "Grant year",
                    "patent_count": "Patents",
                    "technology": "Technology group",
                },
            )
            fig.update_xaxes(dtick=1)
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=55, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.20,
                    xanchor="left",
                    x=0,
                ),
            )
            st.plotly_chart(fig, width="stretch")

            left, right = st.columns(2)

            with left:
                # Technology share by year
                share = trend.copy()
                share["year_total"] = share.groupby(
                    "grant_year"
                )["patent_count"].transform("sum")
                share["share_percent"] = (
                    share["patent_count"] / share["year_total"] * 100
                )

                fig = px.area(
                    share,
                    x="grant_year",
                    y="share_percent",
                    color="technology",
                    title="Share within selected major technology groups",
                    labels={
                        "grant_year": "Grant year",
                        "share_percent": "Share (%)",
                        "technology": "Technology group",
                    },
                    groupnorm=None,
                )
                fig.update_xaxes(dtick=1)
                fig.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, width="stretch")

            with right:
                # Median grant time by technology
                if "filing_to_grant_years" in tech_df.columns:
                    grant_time = (
                        tech_df[
                            tech_df["cpc_group"].isin(top_techs)
                        ]
                        .groupby("cpc_group")[
                            "filing_to_grant_years"
                        ]
                        .median()
                        .dropna()
                        .sort_values()
                        .reset_index(name="median_years")
                    )
                    grant_time["technology"] = grant_time[
                        "cpc_group"
                    ].map(technology_label)

                    fig = px.bar(
                        grant_time,
                        x="median_years",
                        y="technology",
                        orientation="h",
                        title="Median filing-to-grant time",
                        labels={
                            "median_years": "Median years",
                            "technology": "",
                        },
                        text_auto=".1f",
                    )
                    fig.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=55, b=10),
                    )
                    st.plotly_chart(fig, width="stretch")

            # Country × technology heatmap
            if "assignee_country" in tech_df.columns:
                country_tech = tech_df[
                    tech_df["assignee_country"].ne("Unknown")
                    & tech_df["cpc_group"].isin(top_techs)
                ]

                top_countries = (
                    country_tech["assignee_country"]
                    .value_counts()
                    .head(10)
                    .index
                )
                country_tech = country_tech[
                    country_tech["assignee_country"].isin(top_countries)
                ]

                if not country_tech.empty:
                    pivot = pd.crosstab(
                        country_tech["assignee_country"],
                        country_tech["cpc_group"],
                        normalize="index",
                    ) * 100

                    pivot = pivot.reindex(index=top_countries).dropna(
                        how="all"
                    )
                    pivot.columns = [
                        technology_label(c) for c in pivot.columns
                    ]

                    fig = px.imshow(
                        pivot,
                        text_auto=".1f",
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title="Technology profile of leading assignee countries",
                        labels=dict(
                            x="Technology group",
                            y="Assignee country",
                            color="Share %",
                        ),
                    )
                    fig.update_layout(
                        height=500,
                        margin=dict(l=10, r=10, t=60, b=10),
                    )
                    st.plotly_chart(fig, width="stretch")

            # Technology benchmark table
            agg_map = {
                "patents": ("publication_number", "nunique"),
            }

            optional_aggs = {
                "median_inventors": ("inventor_count", "median"),
                "median_cpc_count": ("cpc_count", "median"),
                "median_backward_citations": (
                    "backward_citation_count",
                    "median",
                ),
                "median_claim_words": ("claims_word_count", "median"),
                "median_grant_years": (
                    "filing_to_grant_years",
                    "median",
                ),
            }

            for output_name, spec in optional_aggs.items():
                if spec[0] in tech_df.columns:
                    agg_map[output_name] = spec

            benchmark = (
                tech_df.groupby("cpc_group")
                .agg(**agg_map)
                .sort_values("patents", ascending=False)
                .reset_index()
            )
            benchmark.insert(
                1,
                "technology",
                benchmark["cpc_group"].map(technology_label),
            )

            st.markdown("#### Technology benchmark table")
            st.dataframe(
                benchmark,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# TAB 4 — PATENT CHARACTERISTICS
# ============================================================
with characteristics_tab:
    st.subheader("Patent Characteristics")
    st.caption(
        "Explore distributions and relationships between selected patent attributes."
    )

    numeric_candidates = [
        "inventor_count",
        "assignee_count",
        "cpc_count",
        "backward_citation_count",
        "title_word_count",
        "abstract_word_count",
        "claims_word_count",
        "filing_to_grant_years",
    ]
    numeric_cols = [
        c for c in numeric_candidates if c in filtered.columns
    ]

    if not numeric_cols:
        empty_state("No numerical patent-characteristic columns are available.")
    else:
        left, right = st.columns(2)

        with left:
            selected_metric = st.selectbox(
                "Distribution variable",
                options=numeric_cols,
                index=(
                    numeric_cols.index("backward_citation_count")
                    if "backward_citation_count" in numeric_cols
                    else 0
                ),
            )

            series = pd.to_numeric(
                filtered[selected_metric], errors="coerce"
            ).dropna()

            if series.empty:
                empty_state("No values available for this variable.")
            else:
                use_log = st.checkbox(
                    "Use log1p scale for this distribution",
                    value=selected_metric
                    in {
                        "backward_citation_count",
                        "claims_word_count",
                    },
                )

                plot_series = (
                    np.log1p(series.clip(lower=0))
                    if use_log
                    else series
                )

                hist_df = pd.DataFrame(
                    {
                        "value": plot_series,
                    }
                )

                title_suffix = " (log1p)" if use_log else ""

                fig = px.histogram(
                    hist_df,
                    x="value",
                    nbins=50,
                    title=f"{selected_metric}{title_suffix}",
                    labels={"value": selected_metric},
                )
                fig.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=55, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("#### Distribution summary")
            summary = (
                filtered[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .describe()
                .T[
                    ["count", "mean", "50%", "std", "min", "max"]
                ]
                .rename(columns={"50%": "median"})
                .round(2)
            )
            st.dataframe(
                summary,
                width="stretch",
            )

        st.divider()

        # Spearman correlation
        if len(numeric_cols) >= 2:
            corr = (
                filtered[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .corr(method="spearman")
            )

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Spearman correlation between patent characteristics",
                labels=dict(
                    color="Correlation",
                    x="",
                    y="",
                ),
            )
            fig.update_layout(
                height=620,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        # Scatter relationship
        scatter_candidates = [
            c
            for c in [
                "cpc_count",
                "backward_citation_count",
                "inventor_count",
                "claims_word_count",
                "filing_to_grant_years",
            ]
            if c in filtered.columns
        ]

        if len(scatter_candidates) >= 2:
            st.markdown("#### Explore a relationship")
            sc1, sc2, sc3 = st.columns([1, 1, 1])

            x_var = sc1.selectbox(
                "X variable",
                scatter_candidates,
                index=0,
                key="scatter_x",
            )
            default_y = (
                scatter_candidates.index("backward_citation_count")
                if "backward_citation_count" in scatter_candidates
                and scatter_candidates.index("backward_citation_count") != 0
                else min(1, len(scatter_candidates) - 1)
            )
            y_var = sc2.selectbox(
                "Y variable",
                scatter_candidates,
                index=default_y,
                key="scatter_y",
            )
            color_var = sc3.selectbox(
                "Color by",
                ["None"]
                + (
                    ["cpc_group"]
                    if "cpc_group" in filtered.columns
                    else []
                )
                + (
                    ["assignee_country"]
                    if "assignee_country" in filtered.columns
                    else []
                ),
            )

            scatter_cols = [x_var, y_var]
            if color_var != "None":
                scatter_cols.append(color_var)

            scatter_df = filtered[scatter_cols].copy()
            scatter_df[x_var] = pd.to_numeric(
                scatter_df[x_var], errors="coerce"
            )
            scatter_df[y_var] = pd.to_numeric(
                scatter_df[y_var], errors="coerce"
            )
            scatter_df = scatter_df.dropna(subset=[x_var, y_var])

            # Keep dashboard responsive
            if len(scatter_df) > 10000:
                scatter_df = scatter_df.sample(
                    10000,
                    random_state=42,
                )

            if scatter_df.empty:
                empty_state("No data available for this relationship.")
            else:
                fig = px.scatter(
                    scatter_df,
                    x=x_var,
                    y=y_var,
                    color=(
                        color_var
                        if color_var != "None"
                        else None
                    ),
                    opacity=0.45,
                    title=f"{x_var} vs {y_var}",
                    render_mode="webgl",
                )
                fig.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 5 — PATENT EXPLORER
# ============================================================
with explorer_tab:
    st.subheader("Patent Explorer")
    st.caption(
        "Browse individual patents after applying the filters in the sidebar."
    )

    preferred_columns = [
        "publication_number",
        "grant_date",
        "grant_year",
        "title",
        "primary_assignee",
        "assignee_country",
        "cpc_group",
        "inventor_count",
        "cpc_count",
        "backward_citation_count",
        "filing_to_grant_days",
    ]
    display_cols = [
        c for c in preferred_columns if c in filtered.columns
    ]

    explorer = filtered[display_cols].copy()

    if "grant_date" in explorer.columns:
        explorer = explorer.sort_values(
            "grant_date",
            ascending=False,
        )

    st.markdown(
        f"**Showing {len(explorer):,} filtered patent records**"
    )

    st.dataframe(
        explorer,
        width="stretch",
        hide_index=True,
        height=620,
    )

    csv_bytes = explorer.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download filtered patent table (CSV)",
        data=csv_bytes,
        file_name="filtered_ai_patents.csv",
        mime="text/csv",
    )


# ============================================================
# METHODOLOGY / NOTES
# ============================================================
st.divider()

with st.expander("ℹ️ Scope, definitions and interpretation notes"):
    st.markdown(
        f"""
        **Source:** Google Patents Public Data accessed through BigQuery.

        **Patent scope:** US-granted utility patents (`B1` / `B2`) containing
        at least one CPC classification within `G06N`.

        **Observation period:** 2021–2026. The 2026 data are incomplete;
        the latest grant currently present in the dataset is
        **{latest_text}**.

        **Primary assignee:** The dashboard uses the `primary_assignee`
        variable created during preprocessing. This is useful for a compact
        competitive view, but jointly owned patents can contain additional
        assignees.

        **Technology group:** The dashboard uses the cleaned `cpc_group`
        variable from the preprocessing pipeline.

        **Backward citations:** These describe prior patent documents cited
        by the patent; they are not forward citations or a direct measure of
        future patent impact.

        **Interpretation:** Patent counts indicate patenting activity within
        this dataset. They should not be interpreted as company revenue,
        market share or direct proof of technological quality.
        """
    )

st.markdown(
    """
    <div class="footer-note">
        Business Analytics · Part A Descriptive Analysis · AI Patent Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
