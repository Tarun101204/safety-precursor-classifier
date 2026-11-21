"""
Production-grade Streamlit application for SIF Precursor Screening.
Implements single narrative triage with in-text highlighting, high-throughput batch ranking,
dynamic metrics visualization from reports/metrics.json, and qualitative error analysis.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import config
from src.predict import score_narrative, score_batch_dataframe

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SIF Precursor Screening System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .narrative-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.6;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Load Dynamic Metrics Payload
@st.cache_data
def load_system_metrics():
    if not config.METRICS_JSON_PATH.exists():
        return None
    with open(config.METRICS_JSON_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_cv_curve():
    if not config.CV_CURVES_PATH.exists():
        return None
    return pd.read_csv(config.CV_CURVES_PATH)

metrics_payload = load_system_metrics()
cv_curve_df = load_cv_curve()

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("Operational Controls")

active_model_name = st.sidebar.selectbox(
    "Active Inference Engine",
    ["Logistic Regression (Calibrated Baseline)", "Random Forest (Tree Ensemble)"]
)

operating_threshold = st.sidebar.slider(
    "Decision Cutoff Threshold",
    min_value=0.20,
    max_value=0.70,
    value=config.DEFAULT_OPERATING_THRESHOLD,
    step=0.02,
    help="Cutoff above which an incident triggers an urgent safety committee manual review."
)

st.sidebar.markdown(f"**Current Cutoff:** `≥ {operating_threshold:.2f}` flags for inspection.")
st.sidebar.markdown("---")

# Quick Stats in Sidebar (Loaded Dynamically)
if metrics_payload:
    ds = metrics_payload['dataset_summary']
    st.sidebar.markdown("##### System Data Lineage")
    st.sidebar.caption(f"**Records Analyzed:** {ds['total_samples']} (deduplicated)")
    st.sidebar.caption(f"**SIF Baseline Prevalence:** {ds['baseline_prevalence'] * 100:.1f}%")
    st.sidebar.caption(f"**Validation:** 5-Fold Repeated CV (3 seeds)")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Advisory Notice:** Operates strictly as a prioritization decision-support tool "
    "under NIOSH safety guidelines. Does not substitute for formal statutory incident investigations."
)

# -----------------------------------------------------------------------------
# 3. HEADER & TABS
# -----------------------------------------------------------------------------
st.title("SIF Precursor Screening System")
st.caption("AI-driven prioritization of serious injury and fatality precursors in free-text safety narratives")

tab_screen, tab_batch, tab_perf, tab_errors, tab_about = st.tabs([
    "Single Incident Screening",
    "Batch Triage Queue",
    "Model Performance",
    "Error Analysis",
    "About & Methodology"
])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE INCIDENT SCREENING
# -----------------------------------------------------------------------------
with tab_screen:
    st.subheader("Field Incident Triage")

    # Session State Preset Handlers
    if "narrative_input" not in st.session_state:
        st.session_state["narrative_input"] = (
            "While adjusting the primary feeder line, worker contacted 1200v distribution busbar, "
            "triggering an arc flash and electrical burns."
        )

    def set_preset(text):
        st.session_state["narrative_input"] = text

    st.markdown("##### Standard Preset Test Narratives")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("1200V Electrical Hazard", use_container_width=True):
            set_preset("Worker opened electrical panel and was hit by current of 1200v during maintenance, sustaining severe arc flash exposure.")
    with col_p2:
        if st.button("5-Ton Crane Hoist Snap", use_container_width=True):
            set_preset("Lifting sling snapped while hoisting 5 tons precast girder over the lower processing deck.")
    with col_p3:
        if st.button("Routine Paper Cut (Safe)", use_container_width=True):
            set_preset("Employee sustained minor paper cut to index finger while filing logistics paperwork. First aid antiseptic applied.")

    narrative_text = st.text_area(
        "Incident Narrative / Shift Log:",
        value=st.session_state["narrative_input"],
        height=110,
        placeholder="Paste industrial incident narrative here..."
    )

    if st.button("Execute Safety Screening", type="primary"):
        if narrative_text.strip():
            result = score_narrative(narrative_text, active_model_name, operating_threshold)

            st.markdown("---")
            st.markdown("##### Screening Assessment")

            c_met1, c_met2, c_met3 = st.columns([1, 1.2, 1.5])
            with c_met1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated SIF Probability</div>
                    <div class="metric-value">{result['risk_probability'] * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c_met2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {result['badge_color']};">
                    <div class="metric-label">Priority Classification</div>
                    <div class="metric-value" style="color: {result['badge_color']}; font-size: 19px;">{result['priority_tier']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_met3:
                review_str = "Immediate Manual Review Prioritized" if result['is_flagged'] else "Standard Weekly Log Audit"
                border_c = "#EF4444" if result['is_flagged'] else "#22C55E"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {border_c};">
                    <div class="metric-label">Recommended Action</div>
                    <div class="metric-value" style="font-size: 17px;">{review_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(result['risk_probability'])

            st.markdown("##### Explainable Feature Attribution")
            st.markdown("**Incident narrative with detected high-energy precursors highlighted in place:**")
            st.markdown(f'<div class="narrative-box">{result["highlighted_html"]}</div>', unsafe_allow_html=True)

            if result['top_phrases']:
                st.markdown("<br>**Top Statistically Contributing Tokens (Model Weights):**", unsafe_allow_html=True)
                chips = [f"`{feat}` (+{weight:.2f})" for feat, weight in result['top_phrases']]
                st.write(" • ".join(chips))

# -----------------------------------------------------------------------------
# TAB 2: BATCH TRIAGE QUEUE
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Incident Processing")
    st.write("Upload a CSV file containing unreviewed incident reports to generate an actionable, risk-ranked triage queue.")

    col_upload, col_action = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Incident Records (CSV)", type=["csv"])
    with col_action:
        load_demo_batch = st.button("Load Pre-saved Evaluation Batch", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_demo_batch or config.RAW_DATA_PATH.exists():
        # Load sample from original verified dataset
        batch_df = pd.read_csv(config.RAW_DATA_PATH).head(30)

    if batch_df is not None:
        st.markdown(f"**Loaded {len(batch_df)} incident records.**")
        
        # Select narrative column
        possible_cols = [c for c in batch_df.columns if 'desc' in c.lower() or 'narrative' in c.lower() or 'text' in c.lower()]
        default_col = possible_cols[0] if possible_cols else batch_df.columns[0]
        text_col = st.selectbox("Select Text / Narrative Column:", batch_df.columns, index=batch_df.columns.get_loc(default_col))

        if st.button("Score & Rank Batch Queue", type="primary"):
            with st.spinner("Processing through vectorizer and inference engine..."):
                scored_df = score_batch_dataframe(batch_df, text_col, active_model_name, operating_threshold)

            n_flagged = int(scored_df['Flagged_For_Review'].sum())
            flag_pct = (n_flagged / len(scored_df)) * 100

            st.markdown(f"##### Triage Summary: **{n_flagged} of {len(scored_df)} cases ({flag_pct:.1f}%)** prioritized for manual safety committee inspection.")

            # Display ranked table
            cols_to_show = ['SIF_Risk_Score', 'Priority_Tier', text_col]
            if 'Critical Risk' in scored_df.columns:
                cols_to_show.insert(2, 'Critical Risk')
            if 'Industry Sector' in scored_df.columns:
                cols_to_show.insert(2, 'Industry Sector')

            st.dataframe(
                scored_df[cols_to_show],
                column_config={
                    "SIF_Risk_Score": st.column_config.ProgressColumn(
                        "Risk Score", format="%.2f", min_value=0.0, max_value=1.0
                    ),
                    text_col: st.column_config.TextColumn("Incident Narrative", width="large")
                },
                use_container_width=True,
                height=380
            )

            # Export Button
            csv_export = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Prioritized Review Queue (CSV)",
                data=csv_export,
                file_name="prioritized_sif_triage_queue.csv",
                mime="text/csv"
            )

# -----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & CURVES
# -----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Empirical Model Performance & Cross-Validation")

    if metrics_payload:
        m_lr = metrics_payload['models']['logistic_regression_text']
        m_rf = metrics_payload['models']['random_forest_text']
        m_meta = metrics_payload['models']['logistic_regression_metadata']

        st.markdown("##### 5-Fold Repeated Cross-Validation Benchmark (Mean ± Std)")
        benchmark_table = [
            {
                "Model Pipeline": "Logistic Regression (Text Only)",
                "ROC-AUC": f"{m_lr['roc_auc_mean']:.3f} ± {m_lr['roc_auc_std']:.3f}",
                "PR-AUC": f"{m_lr['pr_auc_mean']:.3f} ± {m_lr['pr_auc_std']:.3f}",
                "Architecture": "Linear TF-IDF with L2 Penalty"
            },
            {
                "Model Pipeline": "Random Forest (Text Only)",
                "ROC-AUC": f"{m_rf['roc_auc_mean']:.3f} ± {m_rf['roc_auc_std']:.3f}",
                "PR-AUC": f"{m_rf['pr_auc_mean']:.3f} ± {m_rf['pr_auc_std']:.3f}",
                "Architecture": "100 Ensembled Decision Trees"
            },
            {
                "Model Pipeline": "Logistic Regression (Text + Sector/Risk Metadata)",
                "ROC-AUC": f"{m_meta['roc_auc_mean']:.3f} ± {m_meta['roc_auc_std']:.3f}",
                "PR-AUC": f"{m_meta['pr_auc_mean']:.3f} ± {m_meta['pr_auc_std']:.3f}",
                "Architecture": "TF-IDF + One-Hot Encoded Categoricals"
            }
        ]
        st.table(pd.DataFrame(benchmark_table))
        st.caption(f"Random chance baselines: ROC-AUC = 0.500 | PR-AUC = {metrics_payload['dataset_summary']['baseline_prevalence']:.3f}")

        # Interactive Curves
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### ROC Curve (Cross-Validated)")
            pts = metrics_payload['curve_points']
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=pts['roc_fpr'], y=pts['roc_tpr'], mode='lines', name='Logistic Regression', line=dict(color='#2563EB', width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(color='#64748B', dash='dash')))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=320, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_c2:
            st.markdown("##### Precision-Recall Curve")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=pts['pr_recall'], y=pts['pr_precision'], mode='lines', name='Logistic Regression', line=dict(color='#10B981', width=2.5)))
            fig_pr.add_hline(y=metrics_payload['dataset_summary']['baseline_prevalence'], line_dash="dash", line_color="#64748B", annotation_text="Prevalence Floor")
            fig_pr.update_layout(xaxis_title="Recall (Catch Rate)", yaxis_title="Precision (Reliability)", height=320, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
            st.plotly_chart(fig_pr, use_container_width=True)

        # Dynamic Threshold Trade-off Curve
        if cv_curve_df is not None:
            st.markdown("##### Interactive Operating Threshold Trade-off")
            st.caption("Inspect how Precision, Recall, and Queue Review Volume shift as the cutoff threshold is adjusted:")
            
            fig_tradeoff = go.Figure()
            fig_tradeoff.add_trace(go.Scatter(x=cv_curve_df['threshold'], y=cv_curve_df['recall'], mode='lines+markers', name='Recall (Catch Rate)', line=dict(color='#EF4444', width=2)))
            fig_tradeoff.add_trace(go.Scatter(x=cv_curve_df['threshold'], y=cv_curve_df['precision'], mode='lines+markers', name='Precision', line=dict(color='#3B82F6', width=2)))
            fig_tradeoff.add_trace(go.Scatter(x=cv_curve_df['threshold'], y=cv_curve_df['flag_rate'], mode='lines+markers', name='% of Queue Flagged', line=dict(color='#A855F7', width=2, dash='dot')))
            fig_tradeoff.add_vline(x=operating_threshold, line_width=2, line_dash="dash", line_color="#F59E0B", annotation_text=f"Selected: {operating_threshold:.2f}")
            fig_tradeoff.update_layout(xaxis_title="Threshold", yaxis_title="Metric Value (0.0 to 1.0)", height=340, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_tradeoff, use_container_width=True)
    else:
        st.warning("Performance artifacts not found. Run 'python src/train.py' to generate metrics.json.")

# -----------------------------------------------------------------------------
# TAB 4: ERROR ANALYSIS
# -----------------------------------------------------------------------------
with tab_errors:
    st.subheader("Qualitative Failure Mode Analysis")
    st.write("Inspection of out-of-fold classification discrepancies at the calibrated operating threshold:")

    if config.ERROR_ANALYSIS_PATH.exists():
        err_df = pd.read_csv(config.ERROR_ANALYSIS_PATH)

        col_fn, col_fp = st.columns(2)
        with col_fn:
            st.markdown("##### False Negatives (Missed Precursors)")
            st.caption("Ground truth was SIF (Level IV/V), but model predicted Routine. Highest consequence error:")
            fn_sub = err_df[err_df['error_type'].str.contains('False Negative')].head(6)
            for _, r in fn_sub.iterrows():
                with st.expander(f"Risk Score: {r['oof_risk_prob']:.2f} | Potential Level: {r['Potential Accident Level']}"):
                    st.write(f"**Narrative:** {r['Description']}")
                    st.caption(f"**Root Cause:** {r['root_cause']}")

        with col_fp:
            st.markdown("##### False Positives (Over-flagged Routine)")
            st.caption("Ground truth was Routine (Level I/II), but model raised an alert. Safe triage bias:")
            fp_sub = err_df[err_df['error_type'].str.contains('False Positive')].head(6)
            for _, r in fp_sub.iterrows():
                with st.expander(f"Risk Score: {r['oof_risk_prob']:.2f} | Actual Level: {r['Accident Level']}"):
                    st.write(f"**Narrative:** {r['Description']}")
                    st.caption(f"**Root Cause:** {r['root_cause']}")
    else:
        st.warning("Error analysis artifact not found. Run 'python src/train.py' first.")

# -----------------------------------------------------------------------------
# TAB 5: ABOUT & METHODOLOGY
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("System Architecture & Governance")
    st.markdown("""
    ### Problem Statement (SIH26165)
    In high-hazard industries (mining, manufacturing, metallurgical refining), incident logs are heavily dominated 
    by routine first-aid events. However, approximately 20-40% of non-injury events contain **SIF Precursors** 
    (near-misses involving high-energy mechanical, electrical, or gravitational hazards). 
    This system screens incident narratives to surface hidden precursor patterns.

    ### Methodology
    1. **Data Lineage:** Industrial Safety & Health Database (411 deduplicated incident narratives).
    2. **Ground Truth Definition:** SIF Precursor = Potential Accident Severity Level IV, V, or VI.
    3. **Leak-Free Preprocessing:** Canonical regex normalizer converts physical hazard quantities into generic semantic tokens (`__voltage_hazard__`, `__heavy_load__`).
    4. **Model Architecture:** Regularized Logistic Regression with balanced loss weighting evaluated against a 100-tree Random Forest benchmark via Repeated 5-Fold Cross-Validation.
    5. **Operating Calibration:** Threshold optimized at ~0.38 to satisfy high-recall safety operational constraints.

    ### Academic & Institutional Reference
    * **Course Assignment:** Supervised Learning (INT395)
    * **Platform Reference:** Smart India Hackathon (SIH 2026 — SIH26165)
    * **Governance Boundary:** Decision-support research prototype. Not an autonomous safety regulator.
    """)
