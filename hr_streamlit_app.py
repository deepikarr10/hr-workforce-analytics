"""
HR Workforce Analytics — Streamlit App
=======================================
Run with:  streamlit run hr_streamlit_app.py

Requirements:
    pip install streamlit pandas matplotlib seaborn scikit-learn joblib plotly

Before running, execute ALL cells in HR_Analysis_code_updated.ipynb so that
the /models/ folder is created with the saved .joblib files.
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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Light theme base ── */
    .stApp { background-color: #f4f6fb; color: #1a1f36; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4ef;
    }

    /* ── KPI cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e4ef;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"]  { color: #5a6380 !important; font-size: 12px !important; }
    [data-testid="stMetricValue"]  { color: #1a1f36 !important; font-size: 22px !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]  { color: #16a34a !important; }

    /* ── Section headers ── */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #5a6380;
        border-bottom: 2px solid #e0e4ef;
        padding-bottom: 6px;
        margin-bottom: 14px;
        margin-top: 10px;
    }

    /* ── Prediction box ── */
    .pred-box {
        background: #ffffff;
        border: 1px solid #e0e4ef;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .pred-value { font-size: 32px; font-weight: 800; color: #2563eb; }
    .pred-label { font-size: 13px; color: #5a6380; margin-bottom: 6px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 8px 8px 0 0;
        border: 1px solid #e0e4ef;
        color: #5a6380;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR = "models"
DEPT_COLORS = {
    "Finance":    "#3b82f6",
    "HR":         "#10b981",
    "IT Support": "#8b5cf6",
    "Operations": "#f59e0b",
    "Marketing":  "#ec4899",
    "Sales":      "#ef4444",
    "Unknown":    "#94a3b8",
}
SEG_COLORS = {"Junior": "#3b82f6", "Mid-Level": "#a855f7", "Senior": "#f97316"}

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
    st.error(f"""
    ⚠️ **Model files not found!**

    Run all cells in `HR_Analysis_code_updated.ipynb` first to generate `/models/`.

    Missing: `{load_err}`
    """)
    st.stop()

df = df_raw.copy()

# Create 'Exp Band' column if missing from older joblib saves
if "Exp Band" not in df.columns:
    def assign_exp_band(exp):
        if exp < 4:   return "0-4"
        elif exp <= 8: return "4-8"
        else:          return "8+"
    df["Exp Band"] = df["Experience"].apply(assign_exp_band)

# Create 'Segment' column if missing
if "Segment" not in df.columns:
    def assign_segment(exp):
        if exp < 4:   return "Junior"
        elif exp <= 8: return "Mid-Level"
        else:          return "Senior"
    df["Segment"] = df["Experience"].apply(assign_segment)

# ─────────────────────────────────────────────
# SIDEBAR — SLICERS (RIGHT side via layout)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Slicers")
    st.markdown("---")

    all_depts = sorted(df["Department"].dropna().unique().tolist())
    sel_depts = st.multiselect("🏢 Department",  all_depts, default=all_depts)

    sel_segs  = st.multiselect("🎯 Segment",
                               ["Junior", "Mid-Level", "Senior"],
                               default=["Junior", "Mid-Level", "Senior"])

    all_mgrs  = sorted(df["Manager"].dropna().unique().tolist())
    sel_mgrs  = st.multiselect("👤 Manager",     all_mgrs,  default=all_mgrs)

    all_bands = ["0-4", "4-8", "8+"]
    sel_bands = st.multiselect("📈 Exp Band",    all_bands, default=all_bands)

    sal_min, sal_max = int(df["Salary"].min()), int(df["Salary"].max())
    sal_range = st.slider("💰 Salary Range (₹)", sal_min, sal_max,
                          (sal_min, sal_max), step=1000, format="₹%d")

    st.markdown("---")
    if st.button("↺ Reset All Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:#5a6380;text-align:center'>"
        f"Models loaded via <b>joblib</b><br>"
        f"📦 {len(os.listdir(MODELS_DIR))} artifacts cached</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# FILTER DATA
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
c1, c2, c3 = st.columns([0.06, 0.7, 0.24])
with c1:
    st.markdown("<div style='font-size:36px;margin-top:4px'>📊</div>", unsafe_allow_html=True)
with c2:
    st.markdown(
        "<h1 style='font-size:24px;font-weight:800;margin:0;color:#1a1f36'>"
        "HR Workforce <span style='color:#2563eb'>Analytics</span> Dashboard</h1>"
        "<div style='font-size:11px;color:#5a6380;margin-top:2px;font-family:monospace'>"
        "50 EMPLOYEES · MULTI-DEPARTMENT · JOBLIB-POWERED ML</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;"
        f"padding:8px 14px;font-size:12px;color:#5a6380;margin-top:8px'>"
        f"🟢 Showing <b style='color:#1a1f36'>{len(filtered)}</b> / {len(df)} employees</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Overview", "💰 Salary Analysis", "🔬 Advanced Analytics", "🤖 ML Predictor"]
)

# ══════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════
with tab1:
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    total       = len(filtered)
    avg_sal     = filtered["Salary"].mean()     if total else 0
    avg_exp     = filtered["Experience"].mean() if total else 0
    missing_mgr = (filtered["Manager"] == "Unassigned").sum()
    top_dept    = filtered.groupby("Department")["Salary"].mean().idxmax() if total else "—"
    top_dept_v  = filtered.groupby("Department")["Salary"].mean().max()   if total else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Employees",   total)
    k2.metric("₹ Avg Salary",        f"₹{avg_sal:,.0f}")
    k3.metric("⏱ Avg Experience",     f"{avg_exp:.1f} yrs")
    k4.metric("⚠️ Missing Managers",  missing_mgr)
    k5.metric("🏆 Highest Paid Dept", top_dept, f"₹{top_dept_v:,.0f} avg")

    st.markdown("")

    col_hc, col_donut = st.columns([2, 1])

    with col_hc:
        st.markdown('<div class="section-header">Department Headcount</div>', unsafe_allow_html=True)
        hc = filtered["Department"].value_counts().reset_index()
        hc.columns = ["Department", "Count"]
        fig_hc = px.bar(hc, x="Count", y="Department", orientation="h",
                        color="Department", color_discrete_map=DEPT_COLORS,
                        text="Count", template="plotly_white")
        fig_hc.update_traces(textposition="outside", marker_line_width=0)
        fig_hc.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=0,r=20,t=10,b=0),
                              yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_hc, use_container_width=True)

    with col_donut:
        st.markdown('<div class="section-header">Employee Segments</div>', unsafe_allow_html=True)
        sc = filtered["Segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Count"]
        fig_d = px.pie(sc, names="Segment", values="Count",
                       color="Segment", color_discrete_map=SEG_COLORS,
                       hole=0.6, template="plotly_white")
        fig_d.update_traces(textinfo="percent+label", pull=[0.03]*3)
        fig_d.update_layout(showlegend=True,
                             legend=dict(font=dict(color="#5a6380", size=11)),
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=280,
                             margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<div class="section-header">Segment Mix by Department</div>', unsafe_allow_html=True)
    stack = filtered.groupby(["Department","Segment"]).size().reset_index(name="Count")
    fig_stack = px.bar(stack, x="Department", y="Count", color="Segment",
                       barmode="stack", color_discrete_map=SEG_COLORS,
                       text="Count", template="plotly_white")
    fig_stack.update_traces(textposition="inside", marker_line_width=0)
    fig_stack.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=300, margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(font=dict(color="#5a6380")))
    st.plotly_chart(fig_stack, use_container_width=True)


# ══════════════════════
# TAB 2 — SALARY
# ══════════════════════
with tab2:
    s1, s2 = st.columns(2)

    with s1:
        st.markdown('<div class="section-header">Avg Salary by Department</div>', unsafe_allow_html=True)
        asd = filtered.groupby("Department")["Salary"].mean().round(0).reset_index().sort_values("Salary", ascending=False)
        asd.columns = ["Department", "Avg Salary"]
        fig_sal = px.bar(asd, x="Department", y="Avg Salary",
                         color="Department", color_discrete_map=DEPT_COLORS,
                         text=asd["Avg Salary"].apply(lambda v: f"₹{v:,.0f}"),
                         template="plotly_white")
        fig_sal.update_traces(textposition="outside", marker_line_width=0)
        fig_sal.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=300,
                               margin=dict(l=0,r=0,t=10,b=0),
                               yaxis=dict(tickprefix="₹"))
        st.plotly_chart(fig_sal, use_container_width=True)

    with s2:
        st.markdown('<div class="section-header">Salary Distribution (Box Plot)</div>', unsafe_allow_html=True)
        fig_box = px.box(filtered, x="Department", y="Salary",
                         color="Department", color_discrete_map=DEPT_COLORS,
                         points="outliers", template="plotly_white")
        fig_box.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=300,
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-header">Salary vs Experience</div>', unsafe_allow_html=True)
    fig_sc = px.scatter(filtered, x="Experience", y="Salary",
                        color="Department", color_discrete_map=DEPT_COLORS,
                        hover_data=["Employee ID", "Manager", "Segment"],
                        template="plotly_white")
    # Add manual linear trendline (numpy only, no statsmodels dependency)
    if len(filtered) >= 2:
        x_vals = filtered["Experience"].values
        y_vals = filtered["Salary"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="#1a1f36", width=2, dash="dash"),
            showlegend=False
        ))
    fig_sc.update_traces(marker=dict(size=9, opacity=0.8))
    fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=350, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(font=dict(color="#5a6380", size=11)))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="section-header">Department Salary Summary</div>', unsafe_allow_html=True)
    summary = (filtered.groupby("Department")["Salary"]
               .agg(Count="count", Min="min", Avg="mean", Max="max")
               .round(0).reset_index().sort_values("Avg", ascending=False))
    for col in ["Min","Avg","Max"]:
        summary[col] = summary[col].apply(lambda v: f"₹{v:,.0f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════
# TAB 3 — ADVANCED
# ══════════════════════
with tab3:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="section-header">Department Bubble Chart</div>', unsafe_allow_html=True)
        bub = (filtered.groupby("Department")
               .agg(Avg_Salary=("Salary","mean"), Avg_Exp=("Experience","mean"),
                    Count=("Employee ID","count")).reset_index())
        fig_bub = px.scatter(bub, x="Avg_Exp", y="Avg_Salary",
                             size="Count", color="Department",
                             color_discrete_map=DEPT_COLORS, text="Department",
                             template="plotly_white", size_max=55)
        fig_bub.update_traces(textposition="top center", marker=dict(opacity=0.8))
        fig_bub.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=320,
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_bub, use_container_width=True)

    with a2:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = filtered[["Salary","Experience"]].corr()
        fig_hm, ax = plt.subplots(figsize=(4,3))
        fig_hm.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues",
                    linewidths=1, linecolor="#e0e4ef", ax=ax,
                    cbar_kws={"shrink":0.8}, annot_kws={"color":"#1a1f36","size":14})
        ax.tick_params(colors="#5a6380")
        plt.title("Salary vs Experience", color="#1a1f36", pad=8, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_hm)

    st.markdown('<div class="section-header">Salary Outlier Detection (IQR)</div>', unsafe_allow_html=True)
    Q1 = filtered["Salary"].quantile(0.25)
    Q3 = filtered["Salary"].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    out_df = filtered[(filtered["Salary"] < lo) | (filtered["Salary"] > hi)]

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Q1 (25th pct)", f"₹{Q1:,.0f}")
    o2.metric("Q3 (75th pct)", f"₹{Q3:,.0f}")
    o3.metric("IQR",           f"₹{IQR:,.0f}")
    o4.metric("Outliers",      len(out_df))

    if len(out_df):
        st.dataframe(out_df[["Employee ID","Department","Salary","Experience","Segment"]],
                     use_container_width=True, hide_index=True)
    else:
        st.success("✅ No salary outliers in current selection.")

    st.markdown('<div class="section-header">Manager Assignment Status</div>', unsafe_allow_html=True)
    mgr_s = filtered["Manager"].apply(lambda m: "Unassigned" if m=="Unassigned" else "Assigned")
    mgr_c = mgr_s.value_counts().reset_index()
    mgr_c.columns = ["Status","Count"]
    fig_mgr = px.pie(mgr_c, names="Status", values="Count", hole=0.55,
                     color="Status",
                     color_discrete_map={"Assigned":"#30c67c","Unassigned":"#ef5350"},
                     template="plotly_white")
    fig_mgr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=250, margin=dict(l=0,r=0,t=10,b=0),
                           legend=dict(font=dict(color="#5a6380")))
    st.plotly_chart(fig_mgr, use_container_width=True)


# ══════════════════════
# TAB 4 — ML PREDICTOR
# ══════════════════════
with tab4:
    st.markdown(
        "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;"
        "padding:14px 18px;margin-bottom:18px;font-size:13px;color:#5a6380'>"
        "🤖 Uses <b style='color:#2563eb'>joblib-loaded models</b>: "
        "<b>LinearRegression</b> (salary) + <b>RandomForestClassifier</b> (segment), "
        "trained in the notebook and loaded via <code>joblib.load()</code>.</div>",
        unsafe_allow_html=True,
    )

    col_inp, col_res = st.columns(2)

    with col_inp:
        st.markdown('<div class="section-header">Input Employee Profile</div>', unsafe_allow_html=True)
        pred_dept = st.selectbox("🏢 Department", list(le_dept.classes_))
        pred_exp  = st.slider("⏱ Years of Experience", 0.0, 20.0, 5.0, 0.1)
        st.selectbox("👤 Manager (reference only)", ["— not used in prediction —"] + list(le_mgr.classes_))
        predict_btn = st.button("🔮 Predict Salary & Segment", use_container_width=True, type="primary")

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
                seg_color   = SEG_COLORS.get(pred_seg, "#2563eb")

                st.markdown(
                    f"<div class='pred-box'>"
                    f"<div class='pred-label'>Predicted Annual Salary</div>"
                    f"<div class='pred-value'>₹{pred_salary:,.0f}</div>"
                    f"<hr style='border-color:#e0e4ef;margin:14px 0'>"
                    f"<div class='pred-label'>Predicted Segment</div>"
                    f"<div style='font-size:22px;font-weight:700;color:{seg_color}'>{pred_seg}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**Segment Probability Breakdown**")
                prob_df = pd.DataFrame({"Segment":seg_classes,"Probability":probs})
                fig_prob = px.bar(prob_df.sort_values("Probability", ascending=True),
                                  x="Probability", y="Segment", orientation="h",
                                  color="Segment", color_discrete_map=SEG_COLORS,
                                  text=prob_df.sort_values("Probability", ascending=True)
                                      ["Probability"].apply(lambda v: f"{v:.1%}"),
                                  template="plotly_white", range_x=[0,1])
                fig_prob.update_traces(textposition="outside", marker_line_width=0)
                fig_prob.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)", height=180,
                                        margin=dict(l=0,r=0,t=5,b=0),
                                        xaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_prob, use_container_width=True)

                dept_avg = df[df["Department"]==pred_dept]["Salary"].mean()
                diff     = pred_salary - dept_avg
                st.info(
                    f"📊 **{pred_dept}** dept avg: **₹{dept_avg:,.0f}** | "
                    f"Your prediction is **{'above' if diff>0 else 'below'}** avg "
                    f"by **₹{abs(diff):,.0f}** ({abs(diff/dept_avg)*100:.1f}%)"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown(
                "<div style='color:#5a6380;font-size:13px;margin-top:40px;text-align:center'>"
                "👈 Fill in the profile and click <b>Predict</b></div>",
                unsafe_allow_html=True,
            )

# FOOTER
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#5a6380;font-size:11px;padding:10px 0'>"
    "HR Workforce Analytics · Streamlit + joblib · "
    "LinearRegression (salary) + RandomForestClassifier (segment)</div>",
    unsafe_allow_html=True,
)
