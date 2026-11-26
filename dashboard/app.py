"""Interactive Streamlit dashboard for LLM Red-Teaming."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from existing_rag_adapter import MockExistingRAGAdapter
from st_aggrid import AgGrid, GridOptionsBuilder
from test_harness import (
    AttackCategory,
    AttackResult,
    AttackSeverity,
    ComparativeHarness,
    TestHarness,
    create_baseline_probes,
)

st.set_page_config(
    page_title="LLM Red-Teaming Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _parse_jsonl_lines(lines: Any) -> list[AttackResult]:
    """Parse JSONL lines into AttackResult objects."""
    results: list[AttackResult] = []
    for raw_line in lines:
        text = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        text = text.strip()
        if not text:
            continue
        data = json.loads(text)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["category"] = AttackCategory(data["category"])
        data["severity"] = AttackSeverity(data["severity"])
        results.append(AttackResult(**data))
    return results


@st.cache_data
def load_results(file_path: Path) -> list[AttackResult]:
    """Load results from JSONL file."""
    with open(file_path, encoding="utf-8") as f:
        return _parse_jsonl_lines(f)


@st.cache_data
def results_to_df(results: list[AttackResult]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    data = []
    for r in results:
        d = r.to_dict()
        d["timestamp"] = r.timestamp
        data.append(d)
    return pd.DataFrame(data)


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
) -> None:
    """Render a metric card."""
    st.metric(label, value, delta=delta, delta_color=delta_color)


def render_sidebar() -> None:
    """Render sidebar with configuration."""
    st.sidebar.title("LLM Red-Teaming")

    results_dir = Path("results")
    if results_dir.exists():
        jsonl_files = list(results_dir.rglob("*.jsonl"))
        if jsonl_files:
            selected = st.sidebar.selectbox(
                "Load Results", options=jsonl_files, format_func=lambda p: p.name
            )
            if selected:
                st.session_state.results_file = selected

    st.sidebar.subheader("Targets")
    st.session_state.selected_targets = st.sidebar.multiselect(
        "Select Targets",
        ["reference-rag", "existing-rag"],
        default=["reference-rag", "existing-rag"],
    )

    st.sidebar.subheader("Providers")
    st.session_state.selected_providers = st.sidebar.multiselect(
        "Select Providers",
        [
            "openai-gpt4o-mini",
            "openai-gpt4o",
            "anthropic-haiku",
            "anthropic-sonnet",
            "ollama-llama31",
        ],
        default=[],
    )

    st.sidebar.subheader("Attack Categories")
    categories = [c.value for c in AttackCategory]
    st.session_state.selected_categories = st.sidebar.multiselect(
        "Categories", categories, default=categories
    )


def render_overview_tab(df: pd.DataFrame) -> None:
    """Render overview tab with key metrics."""
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    passed = int(df["passed"].sum())
    failed = total - passed
    failure_rate = failed / total if total > 0 else 0.0

    with col1:
        render_metric_card("Total Tests", f"{total:,}")
    with col2:
        render_metric_card("Blocked", f"{passed:,}", f"{passed / total:.1%}" if total else None)
    with col3:
        render_metric_card(
            "Succeeded", f"{failed:,}", f"{failure_rate:.1%}" if total else None, "inverse"
        )
    with col4:
        render_metric_card("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")

    st.subheader("Failure Rate by Category")
    cat_stats = (
        df.groupby("category")
        .agg(total=("passed", "count"), failed=("passed", lambda x: (~x).sum()))
        .reset_index()
    )
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Target")
        target_stats = (
            df.groupby("target_name")
            .agg(total=("passed", "count"), failed=("passed", lambda x: (~x).sum()))
            .reset_index()
        )
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
            prov_stats = (
                df.groupby("provider_name")
                .agg(total=("passed", "count"), failed=("passed", lambda x: (~x).sum()))
                .reset_index()
            )
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


def render_findings_tab(df: pd.DataFrame) -> None:
    """Render findings tab with detailed results."""
    st.subheader("Attack Findings")

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect(
            "Category", df["category"].unique(), default=df["category"].unique()
        )
    with col2:
        sev_filter = st.multiselect(
            "Severity", df["severity"].unique(), default=df["severity"].unique()
        )
    with col3:
        target_filter = st.multiselect(
            "Target", df["target_name"].unique(), default=df["target_name"].unique()
        )

    filtered = df[
        (df["category"].isin(cat_filter))
        & (df["severity"].isin(sev_filter))
        & (df["target_name"].isin(target_filter))
    ].copy()

    show_passed = st.checkbox("Show blocked attacks", value=False)
    if not show_passed:
        filtered = filtered[~filtered["passed"]].copy()

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    filtered["severity_rank"] = filtered["severity"].map(severity_order)
    filtered = filtered.sort_values(["severity_rank", "latency_ms"], ascending=[True, False])

    gb = GridOptionsBuilder.from_dataframe(
        filtered[
            [
                "id",
                "target_name",
                "provider_name",
                "probe_name",
                "category",
                "severity",
                "passed",
                "prompt",
                "response",
                "latency_ms",
            ]
        ]
    )
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(wrapText=True, autoHeight=True)
    gb.configure_column("prompt", maxWidth=300)
    gb.configure_column("response", maxWidth=300)

    AgGrid(filtered, gridOptions=gb.build(), height=600, theme="streamlit")

    if st.button("Export Filtered Results"):
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )


def render_comparison_tab() -> None:
    """Render comparison tab for baseline vs mitigated."""
    st.subheader("Baseline vs Mitigated Comparison")

    col1, col2 = st.columns(2)
    with col1:
        baseline_file = st.file_uploader("Baseline Results (JSONL)", type=["jsonl"])
    with col2:
        mitigated_file = st.file_uploader("Mitigated Results (JSONL)", type=["jsonl"])

    if baseline_file and mitigated_file:
        baseline_results = _parse_jsonl_lines(baseline_file)
        mitigated_results = _parse_jsonl_lines(mitigated_file)

        comparator = ComparativeHarness(baseline_results, mitigated_results)
        comparison = comparator.compare()

        overall = comparison["overall"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Baseline Failure Rate", f"{overall['baseline_failure_rate']:.1%}")
        with col2:
            render_metric_card("Mitigated Failure Rate", f"{overall['mitigated_failure_rate']:.1%}")
        with col3:
            render_metric_card(
                "Absolute Reduction",
                f"{overall['absolute_risk_reduction']:.1%}",
                delta_color="normal" if overall["absolute_risk_reduction"] > 0 else "inverse",
            )
        with col4:
            render_metric_card("Relative Reduction", f"{overall['relative_risk_reduction']:.1%}")

        if overall["significant"]:
            st.success(
                f"Statistically significant (p={overall['p_value']:.6f}, Cohen's h={overall['effect_size']:.3f})"
            )
        else:
            st.warning(f"Not statistically significant (p={overall['p_value']:.6f})")

        st.subheader("Category-Level Comparison")
        cat_data = [
            {
                "Category": cat.replace("_", " ").title(),
                "Baseline": data["baseline_failure_rate"],
                "Mitigated": data["mitigated_failure_rate"],
                "Abs. Reduction": data["absolute_reduction"],
                "Rel. Reduction": data["relative_reduction"],
            }
            for cat, data in comparison["by_category"].items()
        ]
        cat_df = pd.DataFrame(cat_data)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Baseline", x=cat_df["Category"], y=cat_df["Baseline"], marker_color="#ef4444"
            )
        )
        fig.add_trace(
            go.Bar(
                name="Mitigated",
                x=cat_df["Category"],
                y=cat_df["Mitigated"],
                marker_color="#22c55e",
            )
        )
        fig.update_layout(
            barmode="group",
            title="Failure Rate by Category",
            yaxis_title="Failure Rate",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

        if comparison["regressions"]:
            st.subheader("Regressions Detected")
            reg_df = pd.DataFrame(comparison["regressions"])
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.success("No regressions detected")


def render_live_test_tab() -> None:
    """Render live testing tab."""
    st.subheader("Live Attack Testing")

    probes = create_baseline_probes()
    probe_names = [p.name for p in probes]
    selected_probes = st.multiselect("Select Probes", probe_names, default=probe_names[:3])
    target_names = st.multiselect(
        "Select Targets", ["reference-rag", "existing-rag"], default=["reference-rag"]
    )

    if st.button("Run Selected Probes", type="primary"):
        if not selected_probes or not target_names:
            st.error("Please select at least one probe and one target")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        async def run_tests() -> list[AttackResult]:
            selected_probe_objs = [p for p in probes if p.name in selected_probes]
            targets = [MockExistingRAGAdapter(name) for name in target_names]
            harness = TestHarness(
                targets=targets, probes=selected_probe_objs, output_dir=Path("results/live")
            )
            return await harness.run_all(max_concurrent=3, progress=False)

        results = asyncio.run(run_tests())
        progress_bar.progress(1.0)
        status_text.text("Complete!")

        df = results_to_df(results)
        st.subheader("Results")
        st.dataframe(
            df[["id", "target_name", "probe_name", "category", "severity", "passed", "latency_ms"]],
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Total", f"{len(results)}")
        with col2:
            render_metric_card("Blocked", f"{int(df['passed'].sum())}")
        with col3:
            render_metric_card("Failed", f"{len(results) - int(df['passed'].sum())}")


def main() -> None:
    """Main dashboard entry point."""
    render_sidebar()

    df = None
    if "results_file" in st.session_state:
        try:
            results = load_results(st.session_state.results_file)
            df = results_to_df(results)
        except Exception as e:
            st.error(f"Error loading results: {e}")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Findings", "Comparison", "Live Test"])

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
