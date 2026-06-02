import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Community Market Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #3b82f6;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background: #fefce8;
        border-left: 4px solid #eab308;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #854d0e;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    # Try multiple paths so the app works wherever it is run from
    candidate_paths = [
        "../data/processed/survey_cleaned.csv",
        "data/processed/survey_cleaned.csv",
        os.path.join(os.path.dirname(__file__),
                     "../data/processed/survey_cleaned.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    # Fallback: generate synthetic demo data so the app still runs for demo purposes
    np.random.seed(42)
    n = 500
    regions = ["Addis Ababa", "Oromia", "Amhara", "SNNPR", "Tigray"]
    occupations = ["Professional", "Business Owner",
                   "Student", "Government Employee", "Trader"]
    categories = ["Electronics", "Clothing",
                  "Food & Beverage", "Personal Care", "Home Goods"]
    channels = ["Physical Shop", "Word of Mouth",
                "Online", "Social Media", "Other"]
    educations = ["Secondary", "Diploma", "Bachelor's", "Master's+", "Primary"]
    genders = ["Male", "Female"]
    challenges = ["Price too high", "Product quality",
                  "Poor service", "Limited variety", "Unknown"]

    df = pd.DataFrame({
        "region":                  np.random.choice(regions,    n, p=[0.55, 0.18, 0.12, 0.10, 0.05]),
        "occupation":              np.random.choice(occupations, n),
        "preferred_product_category": np.random.choice(categories, n, p=[0.40, 0.25, 0.15, 0.12, 0.08]),
        "purchase_channel":        np.random.choice(channels,    n, p=[0.50, 0.25, 0.12, 0.08, 0.05]),
        "education_level":         np.random.choice(educations,  n),
        "gender":                  np.random.choice(genders,     n),
        "top_challenge":           np.random.choice(challenges,  n, p=[0.45, 0.25, 0.15, 0.10, 0.05]),
        "age":                     np.random.randint(18, 65, n),
        "monthly_income_birr":     np.random.exponential(8000, n).clip(1500, 60000),
        "visits_per_month":        np.random.randint(1, 20, n),
        "avg_spend_per_visit_birr": np.random.exponential(250, n).clip(50, 3000),
        "satisfaction_score":      np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.15, 0.42, 0.25, 0.10]),
    })
    df["monthly_spend_birr"] = df["visits_per_month"] * \
        df["avg_spend_per_visit_birr"]
    spend_max = df["monthly_spend_birr"].max()
    sat_max = df["satisfaction_score"].max()
    df["spend_norm"] = df["monthly_spend_birr"] / spend_max
    df["sat_norm"] = df["satisfaction_score"] / sat_max
    df["customer_value_score"] = 0.7 * df["spend_norm"] + 0.3 * df["sat_norm"]
    return df


df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    st.markdown("---")

    all_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.multiselect(
        "Region", all_regions, default=all_regions)

    all_occupations = sorted(df["occupation"].unique().tolist())
    selected_occupations = st.multiselect(
        "Occupation", all_occupations, default=all_occupations)

    income_min = int(df["monthly_income_birr"].min())
    income_max = int(df["monthly_income_birr"].max())
    income_range = st.slider(
        "Monthly Income (Birr)",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max),
        step=500,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af;'>Community Market Research Dashboard<br>Addis Ababa — 2024</div>",
        unsafe_allow_html=True,
    )

# Apply filters
mask = (
    df["region"].isin(selected_regions) &
    df["occupation"].isin(selected_occupations) &
    df["monthly_income_birr"].between(income_range[0], income_range[1])
)
dff = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Community Market Research</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Addis Ababa Consumer Survey — Exploratory & Segmentation Analysis</div>',
            unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)


def kpi(col, value, label):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


kpi(k1, f"{len(dff):,}", "Respondents")
kpi(k2, f"{dff['monthly_income_birr'].median():,.0f} Birr", "Median Income")
kpi(k3, f"{dff['avg_spend_per_visit_birr'].mean():,.0f} Birr",
    "Avg Spend/Visit")
kpi(k4, f"{dff['satisfaction_score'].mean():.1f} / 5", "Avg Satisfaction")
kpi(k5, f"{dff['customer_value_score'].mean():.2f}", "Avg Value Score")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🛒 Behaviour & Challenges", "🗂️ Customer Segments", "📋 Data Explorer"])

sns.set_theme(style="whitegrid", palette="muted")
CHART_COLOR = "#3b82f6"
ACCENT = "#f59e0b"

# ─────────────────── TAB 1: OVERVIEW ─────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    # Chart 1 — Respondents by Region
    with col_a:
        st.markdown(
            '<div class="section-header">Respondents by Region</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        region_counts = dff["region"].value_counts()
        bars = ax.barh(region_counts.index, region_counts.values,
                       color=CHART_COLOR, edgecolor="none")
        ax.bar_label(bars, padding=4, fontsize=9, color="#374151")
        ax.set_xlabel("Count", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ Addis Ababa dominates the sample — findings reflect urban consumers most strongly.</div>', unsafe_allow_html=True)

    # Chart 2 — Income Distribution
    with col_b:
        st.markdown(
            '<div class="section-header">Income Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(dff["monthly_income_birr"], bins=30,
                color=CHART_COLOR, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Monthly Income (Birr)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(
            '<div class="insight-box">⚡ Right-skewed — most respondents are in lower-to-middle income bands.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    # Chart 3 — Avg Spend by Occupation
    with col_c:
        st.markdown(
            '<div class="section-header">Average Spend by Occupation</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        occ_spend = (
            dff.groupby("occupation")["avg_spend_per_visit_birr"]
            .mean()
            .sort_values(ascending=True)
        )
        colors = [ACCENT if v == occ_spend.max(
        ) else CHART_COLOR for v in occ_spend.values]
        bars = ax.barh(occ_spend.index, occ_spend.values,
                       color=colors, edgecolor="none")
        ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9, color="#374151")
        ax.set_xlabel("Avg Spend / Visit (Birr)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(
            '<div class="insight-box">⚡ Professionals and business owners spend the most per visit.</div>', unsafe_allow_html=True)

    # Chart 4 — Preferred Product Categories
    with col_d:
        st.markdown(
            '<div class="section-header">Preferred Product Categories</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        cat_counts = dff["preferred_product_category"].value_counts()
        colors = [ACCENT if v == cat_counts.max(
        ) else CHART_COLOR for v in cat_counts.values]
        bars = ax.bar(cat_counts.index, cat_counts.values,
                      color=colors, edgecolor="none")
        ax.bar_label(bars, padding=4, fontsize=9, color="#374151")
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ One category dominates demand — inventory and promotions should focus here first.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_e, col_f = st.columns(2)

    # Chart 5 — Purchase Channels
    with col_e:
        st.markdown(
            '<div class="section-header">Purchase Channels</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ch_counts = dff["purchase_channel"].value_counts()
        palette = [CHART_COLOR, ACCENT, "#10b981",
                   "#8b5cf6", "#f43f5e"][:len(ch_counts)]
        wedges, texts, autotexts = ax.pie(
            ch_counts.values,
            labels=ch_counts.index,
            autopct="%1.1f%%",
            colors=palette,
            startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for t in texts:
            t.set_fontsize(9)
        for a in autotexts:
            a.set_fontsize(9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ Physical shop is dominant — digital channels are an untapped growth opportunity.</div>', unsafe_allow_html=True)

    # Chart 6 — Satisfaction Scores
    with col_f:
        st.markdown(
            '<div class="section-header">Satisfaction Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sat_counts = dff["satisfaction_score"].value_counts().sort_index()
        bar_colors = [CHART_COLOR if s >= 4 else (
            ACCENT if s == 3 else "#f43f5e") for s in sat_counts.index]
        bars = ax.bar(sat_counts.index.astype(str), sat_counts.values,
                      color=bar_colors, edgecolor="none", width=0.6)
        ax.bar_label(bars, padding=4, fontsize=9, color="#374151")
        ax.set_xlabel(
            "Score (1 = Very Dissatisfied, 5 = Very Satisfied)", fontsize=8)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="warning-box">⚠️ Most customers are neutral (3). A large minority score 1–2 — churn risk is real.</div>', unsafe_allow_html=True)

# ─────────────────── TAB 2: BEHAVIOUR & CHALLENGES ───────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    # Chart 7 — Correlation Heatmap
    with col_a:
        st.markdown(
            '<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        numeric_cols = ["age", "monthly_income_birr", "visits_per_month",
                        "avg_spend_per_visit_birr", "satisfaction_score",
                        "monthly_spend_birr", "customer_value_score"]
        corr = dff[[c for c in numeric_cols if c in dff.columns]].corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax,
            annot_kws={"size": 8},
            square=True,
            cbar_kws={"shrink": 0.75},
        )
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ Monthly spend & value score are tightly linked. Income has almost no correlation with spend.</div>', unsafe_allow_html=True)

    # Chart 8 — Income vs Monthly Spend scatter
    with col_b:
        st.markdown(
            '<div class="section-header">Income vs Monthly Spend</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        if "monthly_spend_birr" in dff.columns:
            ax.scatter(
                dff["monthly_income_birr"],
                dff["monthly_spend_birr"],
                alpha=0.35, s=18, color=CHART_COLOR, edgecolors="none",
            )
            # Regression line
            m, b = np.polyfit(dff["monthly_income_birr"],
                              dff["monthly_spend_birr"], 1)
            x_line = np.linspace(dff["monthly_income_birr"].min(
            ), dff["monthly_income_birr"].max(), 100)
            ax.plot(x_line, m * x_line + b, color=ACCENT, linewidth=2)
        ax.set_xlabel("Monthly Income (Birr)", fontsize=9)
        ax.set_ylabel("Monthly Spend (Birr)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ Weak upward trend only. Low-income customers often outspend high-income ones.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    # Chart 9 — Top Challenges
    with col_c:
        st.markdown(
            '<div class="section-header">Top Customer Challenges</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ch_counts = dff["top_challenge"].value_counts().head(8).sort_values()
        colors = [ACCENT if v == ch_counts.max(
        ) else CHART_COLOR for v in ch_counts.values]
        bars = ax.barh(ch_counts.index, ch_counts.values,
                       color=colors, edgecolor="none")
        ax.bar_label(bars, padding=4, fontsize=9, color="#374151")
        ax.set_xlabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="warning-box">⚠️ "Price too high" is the #1 barrier by a large margin — price sensitivity must drive strategy.</div>', unsafe_allow_html=True)

    # Chart 10 — Customer Value by Education
    with col_d:
        st.markdown(
            '<div class="section-header">Customer Value Score by Education</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        edu_order = ["Primary", "Secondary",
                     "Diploma", "Bachelor's", "Master's+"]
        edu_order_present = [
            e for e in edu_order if e in dff["education_level"].unique()]

        # Defensive: if no education categories or no value score, show a message instead of plotting
        if dff.empty or "customer_value_score" not in dff.columns or len(edu_order_present) == 0:
            ax.text(
                0.5,
                0.5,
                "No education or value-score data available for current filters",
                ha="center",
                va="center",
                fontsize=10,
            )
        else:
            try:
                sns.boxplot(
                    data=dff,
                    x="education_level",
                    y="customer_value_score",
                    order=edu_order_present,
                    palette="Blues",
                    ax=ax,
                    width=0.5,
                    fliersize=3,
                )
            except Exception as err:
                ax.text(
                    0.5,
                    0.5,
                    f"Plot error: {err}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        ax.set_xlabel("Education Level", fontsize=9)
        ax.set_ylabel("Customer Value Score", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">⚡ Higher education = higher customer value. Bachelor\'s+ groups show the most valuable customers.</div>', unsafe_allow_html=True)

# ─────────────────── TAB 3: CUSTOMER SEGMENTS ────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">K-Means Customer Segmentation</div>',
                unsafe_allow_html=True)

    feature_cols = [c for c in ["age", "monthly_income_birr", "visits_per_month",
                                "avg_spend_per_visit_birr", "satisfaction_score",
                                "customer_value_score"] if c in dff.columns]

    if len(dff) >= 20 and len(feature_cols) >= 4:
        X = dff[feature_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow + Silhouette
        inertias, sil_scores, k_range = [], [], range(
            2, min(9, len(X) // 10 + 2))
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))

        best_k = list(k_range)[int(np.argmax(sil_scores))]

        col_el, col_info = st.columns([2, 1])
        with col_el:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.plot(list(k_range), inertias, marker="o",
                     color=CHART_COLOR, linewidth=2)
            ax1.set_title("Elbow Curve", fontsize=10)
            ax1.set_xlabel("K", fontsize=9)
            ax1.set_ylabel("Inertia", fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.spines[["top", "right"]].set_visible(False)

            ax2.plot(list(k_range), sil_scores,
                     marker="o", color=ACCENT, linewidth=2)
            ax2.axvline(x=best_k, color="#f43f5e",
                        linestyle="--", linewidth=1.5, alpha=0.7)
            ax2.set_title("Silhouette Score", fontsize=10)
            ax2.set_xlabel("K", fontsize=9)
            ax2.set_ylabel("Score", fontsize=9)
            ax2.tick_params(labelsize=8)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_info:
            st.metric("Best K (by silhouette)", best_k)
            st.metric("Silhouette Score", f"{max(sil_scores):.3f}")
            st.metric("Total Respondents Clustered", f"{len(X):,}")

        # Fit best K
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        dff = dff.loc[X.index].copy()
        dff["cluster"] = km_final.fit_predict(X_scaled)

        # Cluster profile table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)

        profile_cols = [c for c in ["age", "monthly_income_birr", "visits_per_month",
                                    "avg_spend_per_visit_birr", "satisfaction_score",
                                    "customer_value_score"] if c in dff.columns]
        profile = (
            dff.groupby("cluster")[profile_cols]
            .mean()
            .round(1)
            .rename(columns={
                "age": "Avg Age",
                "monthly_income_birr": "Avg Income (Birr)",
                "visits_per_month": "Visits/Month",
                "avg_spend_per_visit_birr": "Spend/Visit (Birr)",
                "satisfaction_score": "Satisfaction",
                "customer_value_score": "Value Score",
            })
        )
        profile.insert(0, "Count", dff.groupby("cluster").size())
        st.dataframe(profile.style.background_gradient(
            cmap="Blues", subset=["Value Score"]), use_container_width=True)

        # Scatter
        st.markdown("<br>", unsafe_allow_html=True)
        col_sc, col_cat = st.columns(2)
        with col_sc:
            st.markdown(
                '<div class="section-header">Income vs Monthly Spend by Segment</div>', unsafe_allow_html=True)
            if "monthly_spend_birr" in dff.columns:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                palette = sns.color_palette("tab10", n_colors=best_k)
                for clust_id in range(best_k):
                    sub = dff[dff["cluster"] == clust_id]
                    ax.scatter(
                        sub["monthly_income_birr"], sub["monthly_spend_birr"],
                        label=f"Segment {clust_id}", alpha=0.55, s=20,
                        color=palette[clust_id], edgecolors="none",
                    )
                ax.set_xlabel("Monthly Income (Birr)", fontsize=9)
                ax.set_ylabel("Monthly Spend (Birr)", fontsize=9)
                ax.legend(fontsize=8, framealpha=0.5)
                ax.tick_params(labelsize=9)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with col_cat:
            st.markdown(
                '<div class="section-header">Segment Categorical Modes</div>', unsafe_allow_html=True)
            cat_cols = [c for c in ["region", "occupation",
                                    "preferred_product_category"] if c in dff.columns]
            cat_profile = (
                dff.groupby("cluster")[cat_cols]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "—")
            )
            st.dataframe(cat_profile, use_container_width=True)
            st.markdown(
                '<div class="insight-box">⚡ Each segment has a dominant profile — use this to tailor messaging and product targeting.</div>', unsafe_allow_html=True)

    else:
        st.warning(
            "Not enough data to run clustering with current filter selection. Please widen your filters.")

# ─────────────────── TAB 4: DATA EXPLORER ────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Filtered Dataset</div>',
                unsafe_allow_html=True)
    st.caption(f"{len(dff):,} rows shown based on current sidebar filters.")
    st.dataframe(dff.reset_index(drop=True),
                 use_container_width=True, height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    col_s, col_d2 = st.columns(2)

    with col_s:
        st.markdown(
            '<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        num_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(dff[num_cols].describe().round(
            2).T, use_container_width=True)

    with col_d2:
        st.markdown(
            '<div class="section-header">Download Filtered Data</div>', unsafe_allow_html=True)
        csv_bytes = dff.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download as CSV",
            data=csv_bytes,
            file_name="survey_filtered.csv",
            mime="text/csv",
        )
        st.markdown(
            f"<br><div style='font-size:0.85rem;color:#6b7280;'>"
            f"<b>{len(dff):,}</b> rows · <b>{len(dff.columns)}</b> columns</div>",
            unsafe_allow_html=True,
        )
