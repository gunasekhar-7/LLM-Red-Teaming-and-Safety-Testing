"""Interactive Streamlit dashboard for LLM Red-Teaming."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from test_harness import AttackResult, ComparativeHarness, TestHarness
from scripts.run_baseline import create_baseline_probes, run_baseline
from existing_rag_adapter import MockExistingRAGAdapter
from llm_provider import ProviderFactory, LLMSettings


st.set_page_config(
    page_title="LLM Red-Teaming Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.9em;
    }
    .critical { color: #dc2626; }
    .high { color: #ea580c; }
    .medium { color: #ca8a04; }
    .low { color: #16a34a; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(file_path: Path) -> list[AttackResult]:
    """Load results from JSONL file."""
    results = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            # Convert datetime string back to datetime
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            results.append(AttackResult(**data))
    return results


@st.cache_data
def results_to_df(results: list[AttackResult]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    data = []
    for r in results:
        d = r.to_dict()
        d["timestamp"] = r.timestamp
        data.append(d)
    return pd.DataFrame(data)


def render_metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal"):
    """Render a metric card."""
    st.metric(label, value, delta=delta, delta_color=delta_color)


def render_sidebar():
    """Render sidebar with configuration."""
    st.sidebar.title("🔴 LLM Red-Teaming")
    
    # Results file selector
    results_dir = Path("results")
    if results_dir.exists():
        jsonl_files = list(results_dir.rglob("*.jsonl"))
        if jsonl_files:
            selected = st.sidebar.selectbox(
                "Load Results",
                options=jsonl_files,
                format_func=lambda p: p.name,
            )
            if selected:
                st.session_state.results_file = selected
    
    # Target selection
    st.sidebar.subheader("Targets")
    targets = st.sidebar.multiselect(
        "Select Targets",
        ["reference-rag", "existing-rag"],
        default=["reference-rag", "existing-rag"],
    )
    st.session_state.selected_targets = targets
    
    # Provider selection
    st.sidebar.subheader("Providers")
    providers = st.sidebar.multiselect(
        "Select Providers",
        ["openai-gpt4o-mini", "openai-gpt4o", "anthropic-haiku", "anthropic-sonnet", "ollama-llama31"],
        default=[],
    )
    st.session_state.selected_providers = providers
    
    # Attack categories
    st.sidebar.subheader("Attack Categories")
    categories = [c.value for c in AttackCategory]
    selected_cats = st.sidebar.multiselect(
        "Categories",
        categories,
        default=categories,
    )
    st.session_state.selected_categories = selected_cats


def render_overview_tab(df: pd.DataFrame):
    """Render overview tab with key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    passed = int(df["passed"].sum())
    failed = total - passed
    failure_rate = failed / total if total > 0 else 0
    
    with col1:
        render_metric_card("Total Tests", f"{total:,}")
    with col2:
        render_metric_card("Blocked", f"{passed:,}", f"{passed/total:.1%}" if total else None)
    with col3:
        render_metric_card("Succeeded", f"{failed:,}", f"{failure_rate:.1%}" if total else None, "inverse")
    with col4:
        render_metric_card("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")
    
    # Failure rate by category
    st.subheader("Failure Rate by Category")
    cat_stats = df.groupby("category").agg(
        total=("passed", "count"),
        failed=("passed", lambda x: (~x).sum()),
    ).reset_index()
    cat_stats["failure_rate"] = cat_stats["failed"] / cat_stats["total"]
    
    fig = px.bar(
        cat_stats,
        x="category",
        y="failure_rate",
        color="failure_rate",
        color_continuous_scale="RdYlGn_r",
        labels={"failure_rate": "Failure Rate", "category": "Category"},
        text_auto=".1%",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Failure rate by target
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Target")
        target_stats = df.groupby("target_name").agg(
            total=("passed", "count"),
            failed=("passed", lambda x: (~x).sum()),
        ).reset_index()
        target_stats["failure_rate"] = target_stats["failed"] / target_stats["total"]
        
        fig = px.pie(
            target_stats,
            values="failed",
            names="target_name",
            title="Failure Distribution by Target",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("By Provider")
        if "provider_name" in df.columns and df["provider_name"].nunique() > 1:
            prov_stats = df.groupby("provider_name").agg(
                total=("passed", "count"),
                failed=("passed", lambda x: (~x).sum()),
            ).reset_index()
            prov_stats["failure_rate"] = prov_stats["failed"] / prov_stats["total"]
            
            fig = px.bar(
                prov_stats,
                x="provider_name",
                y="failure_rate",
                color="failure_rate",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Single provider or provider data not available")


def render_findings_tab(df: pd.DataFrame):
    """Render findings tab with detailed results."""
    st.subheader("Attack Findings")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect("Category", df["category"].unique(), default=df["category"].unique())
    with col2:
        sev_filter = st.multiselect("Severity", df["severity"].unique(), default=df["severity"].unique())
    with col3:
        target_filter = st.multiselect("Target", df["target_name"].unique(), default=df["target_name"].unique())
    
    # Apply filters
    filtered = df[
        (df["category"].isin(cat_filter)) &
        (df["severity"].isin(sev_filter)) &
        (df["target_name"].isin(target_filter))
    ]
    
    # Show only failures by default
    show_passed = st.checkbox("Show blocked attacks", value=False)
    if not show_passed:
        filtered = filtered[~filtered["passed"]]
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    filtered["severity_rank"] = filtered["severity"].map(severity_order)
    filtered = filtered.sort_values(["severity_rank", "latency_ms"], ascending=[True, False])
    
    # Display as interactive grid
    gb = GridOptionsBuilder.from_dataframe(
        filtered[["id", "target_name", "provider_name", "probe_name", "category", "severity", "passed", "prompt", "response", "latency_ms"]]
    )
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("prompt", maxWidth=300)
    gb.configure_column("response", maxWidth=300)
    
    AgGrid(
        filtered,
        gridOptions=gb.build(),
        height=600,
        theme="streamlit",
    )
    
    # Export
    if st.button("Export Filtered Results"):
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )


def render_comparison_tab():
    """Render comparison tab for baseline vs mitigated."""
    st.subheader("Baseline vs Mitigated Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        baseline_file = st.file_uploader("Baseline Results (JSONL)", type=["jsonl"])
    with col2:
        mitigated_file = st.file_uploader("Mitigated Results (JSONL)", type=["jsonl"])
    
    if baseline_file and mitigated_file:
        baseline_results = []
        for line in baseline_file:
            baseline_results.append(AttackResult(**json.loads(line)))
        
        mitigated_results = []
        for line in mitigated_file:
            mitigated_results.append(AttackResult(**json.loads(line))
        
        comparator = ComparativeHarness(baseline_results, mitigated_results)
        comparison = comparator.compare()
        
        # Overall metrics
        overall = comparison["overall"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Baseline Failure Rate", f"{overall['baseline_failure_rate']:.1%}")
        with col2:
            render_metric_card("Mitigated Failure Rate", f"{overall['mitigated_failure_rate']:.1%}")
        with col3:
            render_metric_card("Absolute Reduction", f"{overall['absolute_risk_reduction']:.1%}", delta_color="normal" if overall['absolute_risk_reduction'] > 0 else "inverse")
        with col4:
            render_metric_card("Relative Reduction", f"{overall['relative_risk_reduction']:.1%}")
        
        # Significance
        if overall["significant"]:
            st.success(f"✅ Statistically significant (p={overall['p_value']:.6f}, Cohen's h={overall['effect_size']:.3f})")
        else:
            st.warning(f"⚠️ Not statistically significant (p={overall['p_value']:.6f})")
        
        # Category comparison
        st.subheader("Category-Level Comparison")
        cat_data = []
        for cat, data in comparison["by_category"].items():
            cat_data.append({
                "Category": cat.replace("_", " ").title(),
                "Baseline": data["baseline_failure_rate"],
                "Mitigated": data["mitigated_failure_rate"],
                "Abs. Reduction": data["absolute_reduction"],
                "Rel. Reduction": data["relative_reduction"],
            })
        
        cat_df = pd.DataFrame(cat_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=cat_df["Category"], y=cat_df["Baseline"], marker_color="#ef4444"))
        fig.add_trace(go.Bar(name="Mitigated", x=cat_df["Category"], y=cat_df["Mitigated"], marker_color="#22c55e"))
        fig.update_layout(barmode="group", title="Failure Rate by Category", yaxis_title="Failure Rate", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Regressions
        if comparison["regressions"]:
            st.subheader("⚠️ Regressions Detected")
            reg_df = pd.DataFrame(comparison["regressions"])
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.success("✅ No regressions detected")


def render_live_test_tab():
    """Render live testing tab."""
    st.subheader("Live Attack Testing")
    
    # Probe selection
    probes = create_baseline_probes()
    probe_names = [p.name for p in probes]
    selected_probes = st.multiselect("Select Probes", probe_names, default=probe_names[:3])
    
    # Target selection
    target_names = st.multiselect("Select Targets", ["reference-rag", "existing-rag"], default=["reference-rag"])
    
    if st.button("Run Selected Probes", type="primary"):
        if not selected_probes or not target_names:
            st.error("Please select at least one probe and one target")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        async def run_tests():
            selected_probe_objs = [p for p in probes if p.name in selected_probes]
            targets = [MockExistingRAGAdapter(name) for name in target_names]
            
            harness = TestHarness(
                targets=targets,
                probes=selected_probe_objs,
                output_dir=Path("results/live"),
            )
            
            results = await harness.run_all(max_concurrent=3, progress=False)
            return results
        
        # Run async in streamlit
        import asyncio
        results = asyncio.run(run_tests())
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        # Show results
        df = results_to_df(results)
        st.subheader("Results")
        st.dataframe(df[["id", "target_name", "probe_name", "category", "severity", "passed", "latency_ms"]], use_container_width=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Total", len(results))
        with col2:
            render_metric_card("Blocked", int(df["passed"].sum()))
        with col3:
            render_metric_card("Failed", len(results) - int(df["passed"].sum()))


def main():
    """Main dashboard entry point."""
    render_sidebar()
    
    # Load results if file selected
    df = None
    if "results_file" in st.session_state:
        try:
            results = load_results(st.session_state.results_file)
            df = results_to_df(results)
        except Exception as e:
            st.error(f"Error loading results: {e}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Findings", "⚖️ Comparison", "🧪 Live Test"])
    
    with tab1:
        if df is not None:
            render_overview_tab(df)
        else:
            st.info("Select a results file from the sidebar to view overview")
    
    with tab2:
        if df is not None:
            render_findings_tab(df)
        else:
            st.info("Select a results file from the sidebar to view findings")
    
    with tab3:
        render_comparison_tab()
    
    with tab4:
        render_live_test_tab()


if __name__ == "__main__":
    main()