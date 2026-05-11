"""
HR Workforce Analytics — Streamlit App
=======================================
Run with:  streamlit run hr_streamlit_app.py

Requirements:
    pip install streamlit pandas matplotlib seaborn scikit-learn joblib plotly

Before running, execute ALL cells in HR_Analysis_code_updated.ipynb first.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Workforce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CLEAN LIGHT THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main background — clean white */
    .stApp { background-color: #f5f7fa; color: #1a1f36; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1e3a5f 0%, #16294a 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #cdd9f0 !important; }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #ffffff !important; font-size: 18px !important; }
    [data-testid="stSidebar"] label { color: #a8c0e0 !important; font-size: 12px !important; font-weight: 600 !important; letter-spacing: 0.5px; }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1);
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 8px;
        font-size: 13px;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2);
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e9f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { color: #1a1f36 !important; font-size: 26px !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { color: #10b981 !important; }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6b7280;
        border-bottom: 2px solid #e4e9f0;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* Chart cards */
    .chart-card {
        background: #ffffff;
        border: 1px solid #e4e9f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 10px;
        box-shadow: 0 4px 16px rgba(37,99,235,0.25);
    }
    .pred-value { font-size: 34px; font-weight: 800; color: #ffffff; }
    .pred-label { font-size: 12px; color: rgba(255,255,255,0.75); margin-bottom: 4px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    .pred-seg   { font-size: 22px; font-weight: 700; color: #fbbf24; }

    /* Info banner */
    .info-banner {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #1e40af;
        margin-top: 12px;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e4e9f0;
        padding: 4px;
        gap: 2px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 7px;
        font-weight: 600;
        font-size: 13px;
        color: #6b7280;
        padding: 8px 18px;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #ffffff !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Hide branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Slider */
    [data-testid="stSlider"] > div > div > div { background: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR = "models"
DEPT_COLORS = {
    "Finance":    "#2563eb",
    "HR":         "#10b981",
    "IT Support": "#8b5cf6",
    "Operations": "#f59e0b",
    "Marketing":  "#ec4899",
    "Sales":      "#ef4444",
    "Unknown":    "#94a3b8",
}
SEG_COLORS = {"Junior": "#38bdf8", "Mid-Level": "#a78bfa", "Senior": "#fb923c"}
TEMPLATE   = "plotly_white"

# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        sal_model  = joblib.load(os.path.join(MODELS_DIR, "salary_model.joblib"))
        seg_model  = joblib.load(os.path.join(MODELS_DIR, "segment_model.joblib"))
        le_dept    = joblib.load(os.path.join(MODELS_DIR, "le_dept.joblib"))
        le_mgr     = joblib.load(os.path.join(MODELS_DIR, "le_mgr.joblib"))
        le_seg     = joblib.load(os.path.join(MODELS_DIR, "le_seg.joblib"))
        df_cleaned = joblib.load(os.path.join(MODELS_DIR, "df_cleaned.joblib"))
        return sal_model, seg_model, le_dept, le_mgr, le_seg, df_cleaned, None
    except FileNotFoundError as e:
        return None, None, None, None, None, None, str(e)

sal_model, seg_model, le_dept, le_mgr, le_seg, df_raw, load_err = load_models()

if load_err:
    st.error(f"⚠️ **Model files not found!** Run all cells in `HR_Analysis_code_updated.ipynb` first.\n\nMissing: `{load_err}`")
    st.stop()

df = df_raw.copy()

# Auto-create missing columns
if "Exp Band" not in df.columns:
    def _exp_band(e): return "0-4" if e < 4 else ("4-8" if e <= 8 else "8+")
    df["Exp Band"] = df["Experience"].apply(_exp_band)

if "Segment" not in df.columns:
    def _segment(e): return "Junior" if e < 4 else ("Mid-Level" if e <= 8 else "Senior")
    df["Segment"] = df["Experience"].apply(_segment)

# ─────────────────────────────────────────────
# HELPER — manual trendline (no statsmodels)
# ─────────────────────────────────────────────
def add_trendline(fig, x_data, y_data, color="#1e3a5f"):
    if len(x_data) < 2:
        return fig
    xs = np.array(x_data, dtype=float)
    ys = np.array(y_data, dtype=float)
    m, b = np.polyfit(xs, ys, 1)
    x0, x1 = xs.min(), xs.max()
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[m*x0+b, m*x1+b],
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name="Trend", showlegend=True
    ))
    return fig

# ─────────────────────────────────────────────
# SIDEBAR — SLICERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 HR Analytics")
    st.markdown("<div style='font-size:11px;color:#8aadd4;margin-top:-8px;margin-bottom:16px'>Filter the dashboard</div>", unsafe_allow_html=True)
    st.markdown("---")

    all_depts = sorted(df["Department"].dropna().unique().tolist())
    sel_depts = st.multiselect("🏢 Department",  all_depts, default=all_depts)

    sel_segs  = st.multiselect("🎯 Segment", ["Junior","Mid-Level","Senior"],
                               default=["Junior","Mid-Level","Senior"])

    all_mgrs  = sorted(df["Manager"].dropna().unique().tolist())
    sel_mgrs  = st.multiselect("👤 Manager", all_mgrs, default=all_mgrs)

    all_bands = ["0-4","4-8","8+"]
    sel_bands = st.multiselect("📈 Exp Band", all_bands, default=all_bands)

    sal_min, sal_max = int(df["Salary"].min()), int(df["Salary"].max())
    sal_range = st.slider("💰 Salary Range (₹)", sal_min, sal_max,
                          (sal_min, sal_max), step=1000, format="₹%d")

    st.markdown("---")
    if st.button("↺  Reset All Filters"):
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:#8aadd4;text-align:center'>"
        f"⚡ Powered by <b style='color:#fff'>joblib</b><br>"
        f"📦 {len(os.listdir(MODELS_DIR))} model artifacts cached</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
filtered = df[
    df["Department"].isin(sel_depts) &
    df["Segment"].isin(sel_segs) &
    df["Manager"].isin(sel_mgrs) &
    df["Exp Band"].isin(sel_bands) &
    df["Salary"].between(sal_range[0], sal_range[1])
]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:14px;"
    "padding:20px 28px;margin-bottom:20px;display:flex;align-items:center;gap:16px'>"
    "<div style='font-size:40px'>📊</div>"
    "<div>"
    "<div style='font-size:22px;font-weight:800;color:#fff'>HR Workforce Analytics Dashboard</div>"
    "<div style='font-size:12px;color:rgba(255,255,255,0.7);margin-top:2px;font-family:monospace'>"
    "50 EMPLOYEES · MULTI-DEPARTMENT · JOBLIB-POWERED ML</div>"
    "</div>"
    f"<div style='margin-left:auto;background:rgba(255,255,255,0.15);border-radius:20px;"
    f"padding:8px 18px;font-size:13px;color:#fff;font-weight:600'>"
    f"🟢 {len(filtered)} / {len(df)} employees</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋  Overview", "💰  Salary Analysis", "🔬  Advanced Analytics", "🤖  ML Predictor"]
)

# ══════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════
with tab1:

    # KPIs
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    total      = len(filtered)
    avg_sal    = filtered["Salary"].mean()     if total else 0
    avg_exp    = filtered["Experience"].mean() if total else 0
    miss_mgr   = (filtered["Manager"] == "Unassigned").sum()
    top_dept   = filtered.groupby("Department")["Salary"].mean().idxmax() if total else "—"
    top_dept_v = filtered.groupby("Department")["Salary"].mean().max()   if total else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("👥 Total Employees",   total)
    k2.metric("₹  Avg Salary",       f"₹{avg_sal:,.0f}")
    k3.metric("⏱  Avg Experience",    f"{avg_exp:.1f} yrs")
    k4.metric("⚠️  Missing Managers", miss_mgr)
    k5.metric("🏆 Highest Paid Dept", top_dept, f"₹{top_dept_v:,.0f} avg")

    st.markdown("<br>", unsafe_allow_html=True)

    # Headcount + Donut
    col_hc, col_donut = st.columns([3, 2])

    with col_hc:
        st.markdown('<div class="section-header">Department Headcount</div>', unsafe_allow_html=True)
        hc = filtered["Department"].value_counts().reset_index()
        hc.columns = ["Department","Count"]
        fig_hc = px.bar(hc, x="Count", y="Department", orientation="h",
                        color="Department", color_discrete_map=DEPT_COLORS,
                        text="Count", template=TEMPLATE)
        fig_hc.update_traces(textposition="outside", marker_line_width=0)
        fig_hc.update_layout(showlegend=False, height=270,
                              margin=dict(l=0,r=20,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"),
                              plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_hc, use_container_width=True)

    with col_donut:
        st.markdown('<div class="section-header">Employee Segments</div>', unsafe_allow_html=True)
        sc = filtered["Segment"].value_counts().reset_index()
        sc.columns = ["Segment","Count"]
        fig_d = px.pie(sc, names="Segment", values="Count",
                       color="Segment", color_discrete_map=SEG_COLORS,
                       hole=0.62, template=TEMPLATE)
        fig_d.update_traces(textinfo="percent+label", pull=[0.03]*3,
                            marker=dict(line=dict(color="#fff",width=2)))
        fig_d.update_layout(showlegend=True, height=270,
                             margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(font=dict(size=11)),
                             plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_d, use_container_width=True)

    # Stacked bar
    st.markdown('<div class="section-header">Segment Mix by Department</div>', unsafe_allow_html=True)
    stack = filtered.groupby(["Department","Segment"]).size().reset_index(name="Count")
    fig_st = px.bar(stack, x="Department", y="Count", color="Segment",
                    barmode="stack", color_discrete_map=SEG_COLORS,
                    text="Count", template=TEMPLATE)
    fig_st.update_traces(textposition="inside", marker_line_width=0)
    fig_st.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(font=dict(size=11)),
                          plot_bgcolor="#fff", paper_bgcolor="#fff")
    st.plotly_chart(fig_st, use_container_width=True)


# ══════════════════════════════════
# TAB 2 — SALARY
# ══════════════════════════════════
with tab2:
    s1, s2 = st.columns(2)

    with s1:
        st.markdown('<div class="section-header">Avg Salary by Department</div>', unsafe_allow_html=True)
        asd = (filtered.groupby("Department")["Salary"]
               .mean().round(0).reset_index()
               .sort_values("Salary", ascending=False))
        asd.columns = ["Department","Avg Salary"]
        fig_sal = px.bar(asd, x="Department", y="Avg Salary",
                         color="Department", color_discrete_map=DEPT_COLORS,
                         text=asd["Avg Salary"].apply(lambda v: f"₹{v:,.0f}"),
                         template=TEMPLATE)
        fig_sal.update_traces(textposition="outside", marker_line_width=0)
        fig_sal.update_layout(showlegend=False, height=300,
                               margin=dict(l=0,r=0,t=10,b=0),
                               yaxis=dict(tickprefix="₹"),
                               plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_sal, use_container_width=True)

    with s2:
        st.markdown('<div class="section-header">Salary Distribution (Box Plot)</div>', unsafe_allow_html=True)
        fig_box = px.box(filtered, x="Department", y="Salary",
                         color="Department", color_discrete_map=DEPT_COLORS,
                         points="outliers", template=TEMPLATE)
        fig_box.update_layout(showlegend=False, height=300,
                               margin=dict(l=0,r=0,t=10,b=0),
                               plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_box, use_container_width=True)

    # Scatter + manual trendline (no statsmodels)
    st.markdown('<div class="section-header">Salary vs Experience (with Trendline)</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for dept in filtered["Department"].unique():
        sub = filtered[filtered["Department"] == dept]
        fig_sc.add_trace(go.Scatter(
            x=sub["Experience"], y=sub["Salary"],
            mode="markers", name=dept,
            marker=dict(color=DEPT_COLORS.get(dept,"#94a3b8"), size=9, opacity=0.85,
                        line=dict(width=1, color="#fff")),
            hovertemplate=(f"<b>{dept}</b><br>Exp: %{{x}}y<br>"
                           "Salary: ₹%{y:,.0f}<extra></extra>")
        ))
    fig_sc = add_trendline(fig_sc, filtered["Experience"], filtered["Salary"], "#1e3a5f")
    fig_sc.update_layout(template=TEMPLATE, height=360,
                          margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Experience (years)", yaxis_title="Salary (₹)",
                          legend=dict(font=dict(size=11)),
                          plot_bgcolor="#fff", paper_bgcolor="#fff")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Department Salary Summary</div>', unsafe_allow_html=True)
    summary = (filtered.groupby("Department")["Salary"]
               .agg(Count="count", Min="min", Avg="mean", Max="max")
               .round(0).reset_index().sort_values("Avg", ascending=False))
    for c in ["Min","Avg","Max"]:
        summary[c] = summary[c].apply(lambda v: f"₹{v:,.0f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════
# TAB 3 — ADVANCED
# ══════════════════════════════════
with tab3:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="section-header">Department Bubble Chart</div>', unsafe_allow_html=True)
        bub = (filtered.groupby("Department")
               .agg(Avg_Salary=("Salary","mean"),
                    Avg_Exp=("Experience","mean"),
                    Count=("Employee ID","count")).reset_index())
        fig_bub = px.scatter(bub, x="Avg_Exp", y="Avg_Salary",
                             size="Count", color="Department",
                             color_discrete_map=DEPT_COLORS, text="Department",
                             template=TEMPLATE, size_max=55)
        fig_bub.update_traces(textposition="top center",
                               marker=dict(opacity=0.85, line=dict(width=1,color="#fff")))
        fig_bub.update_layout(showlegend=False, height=320,
                               margin=dict(l=0,r=0,t=10,b=0),
                               xaxis_title="Avg Experience (yrs)",
                               yaxis_title="Avg Salary (₹)",
                               plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_bub, use_container_width=True)

    with a2:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = filtered[["Salary","Experience"]].corr()
        fig_hm, ax = plt.subplots(figsize=(4, 3.2))
        fig_hm.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues",
                    linewidths=2, linecolor="#e4e9f0", ax=ax,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"color": "#1a1f36", "size": 15, "weight": "bold"})
        ax.tick_params(colors="#6b7280", labelsize=10)
        plt.title("Salary vs Experience Correlation",
                  color="#1a1f36", pad=10, fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_hm)

    # Outlier Detection
    st.markdown('<div class="section-header">Salary Outlier Detection (IQR Method)</div>', unsafe_allow_html=True)
    Q1  = filtered["Salary"].quantile(0.25)
    Q3  = filtered["Salary"].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    out_df = filtered[(filtered["Salary"] < lo) | (filtered["Salary"] > hi)]

    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Q1 (25th pct)",  f"₹{Q1:,.0f}")
    o2.metric("Q3 (75th pct)",  f"₹{Q3:,.0f}")
    o3.metric("IQR",            f"₹{IQR:,.0f}")
    o4.metric("Outliers Found", len(out_df))

    if len(out_df):
        st.dataframe(
            out_df[["Employee ID","Department","Salary","Experience","Segment"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.success("✅ No salary outliers in the current selection.")

    # Manager Assignment
    st.markdown('<div class="section-header">Manager Assignment Status</div>', unsafe_allow_html=True)
    mgr_s = filtered["Manager"].apply(lambda m: "Unassigned" if m=="Unassigned" else "Assigned")
    mgr_c = mgr_s.value_counts().reset_index()
    mgr_c.columns = ["Status","Count"]
    fig_mgr = px.pie(mgr_c, names="Status", values="Count", hole=0.58,
                     color="Status",
                     color_discrete_map={"Assigned":"#10b981","Unassigned":"#ef4444"},
                     template=TEMPLATE)
    fig_mgr.update_traces(marker=dict(line=dict(color="#fff",width=2)))
    fig_mgr.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                           legend=dict(font=dict(size=12)),
                           plot_bgcolor="#fff", paper_bgcolor="#fff")
    st.plotly_chart(fig_mgr, use_container_width=True)


# ══════════════════════════════════
# TAB 4 — ML PREDICTOR
# ══════════════════════════════════
with tab4:
    st.markdown(
        "<div class='info-banner'>🤖 Uses <b>joblib-loaded models</b>: "
        "<b>LinearRegression</b> (salary predictor) + "
        "<b>RandomForestClassifier</b> (segment classifier), "
        "trained in the notebook and loaded via <code>joblib.load()</code>.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_inp, col_res = st.columns(2)

    with col_inp:
        st.markdown('<div class="section-header">Input Employee Profile</div>', unsafe_allow_html=True)
        pred_dept = st.selectbox("🏢 Department", list(le_dept.classes_))
        pred_exp  = st.slider("⏱ Years of Experience", 0.0, 20.0, 5.0, 0.1)
        st.selectbox("👤 Manager (reference only)",
                     ["— not used in prediction —"] + list(le_mgr.classes_))

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Predict Salary & Segment",
                                use_container_width=True, type="primary")

    with col_res:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        if predict_btn:
            try:
                dept_enc    = le_dept.transform([pred_dept])[0]
                pred_salary = sal_model.predict([[dept_enc, pred_exp]])[0]
                seg_enc_p   = seg_model.predict([[dept_enc, pred_salary, pred_exp]])[0]
                pred_seg    = le_seg.inverse_transform([seg_enc_p])[0]
                probs       = seg_model.predict_proba([[dept_enc, pred_salary, pred_exp]])[0]
                seg_classes = le_seg.inverse_transform(range(len(probs)))

                st.markdown(
                    f"<div class='pred-box'>"
                    f"<div class='pred-label'>Predicted Annual Salary</div>"
                    f"<div class='pred-value'>₹{pred_salary:,.0f}</div>"
                    f"<div style='height:14px'></div>"
                    f"<div class='pred-label'>Predicted Segment</div>"
                    f"<div class='pred-seg'>{pred_seg}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("**Segment Probability Breakdown**")
                prob_df = pd.DataFrame({"Segment": seg_classes, "Probability": probs})
                fig_prob = px.bar(
                    prob_df.sort_values("Probability", ascending=True),
                    x="Probability", y="Segment", orientation="h",
                    color="Segment", color_discrete_map=SEG_COLORS,
                    text=prob_df.sort_values("Probability", ascending=True)
                        ["Probability"].apply(lambda v: f"{v:.1%}"),
                    template=TEMPLATE, range_x=[0, 1]
                )
                fig_prob.update_traces(textposition="outside", marker_line_width=0)
                fig_prob.update_layout(showlegend=False, height=190,
                                        margin=dict(l=0,r=30,t=5,b=0),
                                        xaxis=dict(tickformat=".0%"),
                                        plot_bgcolor="#fff", paper_bgcolor="#fff")
                st.plotly_chart(fig_prob, use_container_width=True)

                dept_avg = df[df["Department"] == pred_dept]["Salary"].mean()
                diff     = pred_salary - dept_avg
                direction = "above" if diff > 0 else "below"
                color     = "#10b981" if diff > 0 else "#ef4444"
                st.markdown(
                    f"<div class='info-banner'>"
                    f"📊 <b>{pred_dept}</b> dept avg: <b>₹{dept_avg:,.0f}</b> &nbsp;|&nbsp; "
                    f"Prediction is <b style='color:{color}'>{direction}</b> avg by "
                    f"<b>₹{abs(diff):,.0f}</b> ({abs(diff/dept_avg)*100:.1f}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:13px;margin-top:60px;text-align:center'>"
                "👈 Fill in the employee profile and click <b>Predict</b></div>",
                unsafe_allow_html=True,
            )

# FOOTER
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:11px;padding:16px 0 8px'>"
    "HR Workforce Analytics · Streamlit + joblib · "
    "LinearRegression (salary) + RandomForestClassifier (segment)"
    "</div>",
    unsafe_allow_html=True,
)
