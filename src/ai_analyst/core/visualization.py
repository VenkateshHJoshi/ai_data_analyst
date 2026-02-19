"""
visualization.py
─────────────────────────────────────────────────────────────────────────────
Generates the RIGHT chart for the RIGHT query intent.

Intent → Chart mapping
  outlier / box / spread          → Box plot  (full column + outliers highlighted)
  correlation / multicollinearity → Heatmap   (annotated correlation matrix)
  trend / time / over time        → Line chart
  distribution / histogram        → Histogram with KDE overlay
  cluster / segment               → Scatter plot coloured by cluster
  compare / group / by category   → Grouped bar chart
  summary / describe              → Horizontal bar (mean values)
  default                         → Smart fallback (bar / scatter / line)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Optional
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)

# ── colour palette (dark-theme friendly) ─────────────────────────────────────
CLR_NORMAL   = "#4C9BE8"   # blue  – normal data points
CLR_OUTLIER  = "#FF4B4B"   # red   – outliers
CLR_MEDIAN   = "#F5A623"   # orange – median / reference lines
CLR_BG       = "rgba(0,0,0,0)"
FONT_COLOR   = "#FFFFFF"

LAYOUT_BASE = dict(
    paper_bgcolor=CLR_BG,
    plot_bgcolor ="rgba(14,17,23,1)",
    font         =dict(color=FONT_COLOR, size=13),
    margin       =dict(l=60, r=40, t=70, b=80),
    xaxis        =dict(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
    yaxis        =dict(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
)


# ── intent detection ──────────────────────────────────────────────────────────
def _intent(query: str) -> str:
    q = query.lower()
    if re.search(r"outlier|anomal|box|spread|iqr|extreme", q):
        return "outlier"
    if re.search(r"correlat|multicollinear|heatmap|vif|relationship between", q):
        return "correlation"
    if re.search(r"trend|over time|time series|monthly|daily|weekly|yearly", q):
        return "trend"
    if re.search(r"distribut|histogram|frequency|spread of", q):
        return "histogram"
    if re.search(r"cluster|segment|group by cluster|pca", q):
        return "cluster"
    if re.search(r"compar|group by|by categor|by gender|by class|by type|average.*by|mean.*by", q):
        return "groupbar"
    if re.search(r"summar|descri|overview|statistic", q):
        return "summary"
    return "auto"


# ── helpers ───────────────────────────────────────────────────────────────────
def _numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _cat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]

def _date_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def _apply_layout(fig, title: str, xlabel: str = None, ylabel: str = None) -> go.Figure:
    updates = dict(**LAYOUT_BASE, title=dict(text=title, font=dict(size=18, color=FONT_COLOR)))
    if xlabel:
        updates["xaxis"] = {**LAYOUT_BASE.get("xaxis", {}), "title": xlabel}
    if ylabel:
        updates["yaxis"] = {**LAYOUT_BASE.get("yaxis", {}), "title": ylabel}
    fig.update_layout(**updates)
    return fig


# ── 1. OUTLIER – Box plot with full distribution + highlighted outlier points ─
def _chart_outlier(result_df: pd.DataFrame, original_query: str) -> go.Figure:
    """
    result_df  : the filtered outlier rows returned by execute_code
    We reconstruct the full box from result_df stats if we only have outlier rows,
    or plot all rows when the result contains the full dataset column.
    """
    num_cols = _numeric_cols(result_df)
    if not num_cols:
        return None

    # Pick the column most likely asked about from the query
    col = num_cols[0]
    for c in num_cols:
        if c.lower() in original_query.lower():
            col = c
            break

    vals = result_df[col].dropna()

    # Compute IQR bounds from available data
    Q1   = vals.quantile(0.25)
    Q3   = vals.quantile(0.75)
    IQR  = Q3 - Q1
    lo   = Q1 - 1.5 * IQR
    hi   = Q3 + 1.5 * IQR

    fig = go.Figure()

    # Box trace – shows full distribution shape
    fig.add_trace(go.Box(
        y=vals,
        name=col,
        boxpoints="all",        # show every point
        jitter=0.4,
        pointpos=0,
        marker=dict(
            color=[CLR_OUTLIER if (v < lo or v > hi) else CLR_NORMAL for v in vals],
            size=6,
            line=dict(width=1, color="#ffffff"),
        ),
        line=dict(color=CLR_NORMAL, width=2),
        fillcolor="rgba(76,155,232,0.15)",
        whiskerwidth=0.5,
        hovertemplate="<b>%{y}</b><extra></extra>",
    ))

    # IQR boundary reference lines
    for y_val, label, dash in [
        (hi, f"Upper fence  ({hi:.2f})", "dash"),
        (Q3, f"Q3  ({Q3:.2f})",          "dot"),
        (Q1, f"Q1  ({Q1:.2f})",          "dot"),
        (lo, f"Lower fence  ({lo:.2f})", "dash"),
    ]:
        fig.add_hline(
            y=y_val,
            line=dict(color=CLR_MEDIAN, width=1, dash=dash),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=CLR_MEDIAN, size=11),
        )

    n_out = int(((vals < lo) | (vals > hi)).sum())
    title = (
        f"Outlier Analysis — <b>{col}</b><br>"
        f"<sup style='color:{CLR_OUTLIER}'>{n_out} outlier(s) detected</sup>  "
        f"<sup style='color:#aaa'>IQR = {IQR:.2f}  |  fences [{lo:.2f}, {hi:.2f}]</sup>"
    )
    fig = _apply_layout(fig, title, ylabel=col)
    fig.update_layout(showlegend=False)
    return fig


# ── 2. CORRELATION HEATMAP ────────────────────────────────────────────────────
def _chart_heatmap(result_df: pd.DataFrame) -> go.Figure:
    """
    result_df is expected to be a correlation matrix (square, numeric).
    Renders an annotated heatmap with diverging colour scale.
    """
    num_cols = _numeric_cols(result_df)
    if not num_cols:
        return None

    # If result_df is already a corr matrix (square + same index as columns)
    corr = result_df[num_cols]
    labels = list(corr.columns)
    z = corr.values.round(2)

    # Text annotations on each cell
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale="RdBu_r",   # red = positive, blue = negative
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0<br>(negative)", "-0.5", "0<br>(none)", "0.5", "1.0<br>(positive)"],
            tickfont=dict(color=FONT_COLOR, size=10),
        ),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
    ))

    n = len(labels)
    title = (
        f"Correlation Heatmap — {n} variables<br>"
        "<sup style='color:#aaa'>Red = strong positive  |  Blue = strong negative  |  "
        "White = no correlation  |  |r| > 0.7 indicates multicollinearity risk</sup>"
    )
    fig = _apply_layout(fig, title)
    fig.update_layout(
        xaxis=dict(tickangle=-40, side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=max(400, n * 55),
    )
    return fig


# ── 3. TREND – Line chart ─────────────────────────────────────────────────────
def _chart_trend(result_df: pd.DataFrame, query: str) -> go.Figure:
    date_c = _date_cols(result_df)
    num_c  = _numeric_cols(result_df)
    cat_c  = _cat_cols(result_df)

    # x-axis: datetime > string that looks like date > first string col > index
    if date_c:
        x_col = date_c[0]
    elif cat_c:
        x_col = cat_c[0]
    else:
        result_df = result_df.reset_index()
        x_col = "index"

    if not num_c:
        return None

    y_col = num_c[0]
    fig = px.line(
        result_df, x=x_col, y=y_col,
        markers=True,
        color_discrete_sequence=[CLR_NORMAL],
    )
    fig.update_traces(line_width=2, marker_size=6)
    fig = _apply_layout(fig, f"Trend of <b>{y_col}</b> over <b>{x_col}</b>",
                        xlabel=x_col, ylabel=y_col)
    return fig


# ── 4. HISTOGRAM ──────────────────────────────────────────────────────────────
def _chart_histogram(result_df: pd.DataFrame, query: str) -> go.Figure:
    num_c = _numeric_cols(result_df)
    if not num_c:
        return None

    col = num_c[0]
    for c in num_c:
        if c.lower() in query.lower():
            col = c
            break

    vals = result_df[col].dropna()
    mean_v   = vals.mean()
    median_v = vals.median()
    std_v    = vals.std()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=40,
        name=col,
        marker_color=CLR_NORMAL,
        opacity=0.8,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    # Mean & median reference lines
    for val, label, color in [
        (mean_v,   f"Mean {mean_v:.2f}",     CLR_OUTLIER),
        (median_v, f"Median {median_v:.2f}", CLR_MEDIAN),
    ]:
        fig.add_vline(
            x=val,
            line=dict(color=color, width=2, dash="dash"),
            annotation_text=label,
            annotation_position="top",
            annotation_font=dict(color=color, size=11),
        )

    title = (
        f"Distribution of <b>{col}</b><br>"
        f"<sup style='color:#aaa'>Mean={mean_v:.2f}  |  Median={median_v:.2f}  "
        f"|  Std={std_v:.2f}  |  N={len(vals)}</sup>"
    )
    fig = _apply_layout(fig, title, xlabel=col, ylabel="Count")
    return fig


# ── 5. CLUSTER SCATTER ────────────────────────────────────────────────────────
def _chart_cluster(result_df: pd.DataFrame) -> go.Figure:
    num_c = _numeric_cols(result_df)
    # Look for a cluster column
    cluster_col = next(
        (c for c in result_df.columns if "cluster" in c.lower()),
        None
    )
    if len(num_c) < 2:
        return None

    x_col = num_c[0]
    y_col = num_c[1]

    fig = px.scatter(
        result_df,
        x=x_col,
        y=y_col,
        color=cluster_col if cluster_col else None,
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75,
    )
    fig.update_traces(marker_size=7)
    fig = _apply_layout(
        fig,
        f"Cluster Scatter — <b>{x_col}</b> vs <b>{y_col}</b>",
        xlabel=x_col,
        ylabel=y_col,
    )
    return fig


# ── 6. GROUPED BAR ────────────────────────────────────────────────────────────
def _chart_groupbar(result_df: pd.DataFrame, query: str) -> go.Figure:
    cat_c = _cat_cols(result_df)
    num_c = _numeric_cols(result_df)
    if not num_c:
        return None

    x_col  = cat_c[0] if cat_c else result_df.columns[0]
    y_cols = num_c[:3]   # max 3 numeric series for readability

    fig = go.Figure()
    colors = [CLR_NORMAL, CLR_OUTLIER, CLR_MEDIAN]
    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Bar(
            x=result_df[x_col],
            y=result_df[y_col],
            name=y_col,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(barmode="group")
    title = f"<b>{y_cols[0]}</b> by <b>{x_col}</b>"
    if len(y_cols) > 1:
        title = "Grouped Comparison by <b>" + x_col + "</b>"
    fig = _apply_layout(fig, title, xlabel=x_col, ylabel="Value")
    return fig


# ── 7. SUMMARY BAR (describe output) ─────────────────────────────────────────
def _chart_summary(result_df: pd.DataFrame) -> go.Figure:
    num_c = _numeric_cols(result_df)
    if not num_c:
        return None

    # result_df from df.describe().T has stats as columns
    stat_cols = [c for c in result_df.columns if c in ("mean", "50%", "std", "min", "max")]
    if not stat_cols:
        # Plain numeric result — fallback to horizontal bar of first numeric col
        col = num_c[0]
        fig = px.bar(result_df, y=result_df.index, x=col, orientation="h",
                     color_discrete_sequence=[CLR_NORMAL])
        fig = _apply_layout(fig, f"Summary — {col}", xlabel=col)
        return fig

    means = result_df["mean"] if "mean" in result_df.columns else result_df[stat_cols[0]]
    stds  = result_df["std"]  if "std"  in result_df.columns else None

    fig = go.Figure(go.Bar(
        x=means.index,
        y=means.values,
        error_y=dict(type="data", array=stds.values, visible=True) if stds is not None else None,
        marker_color=CLR_NORMAL,
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}<extra></extra>",
    ))
    fig = _apply_layout(fig, "Dataset Summary — Mean ± Std per Column",
                        xlabel="Column", ylabel="Mean Value")
    fig.update_layout(xaxis_tickangle=-35)
    return fig


# ── AUTO FALLBACK ─────────────────────────────────────────────────────────────
def _chart_auto(result_df: pd.DataFrame, query: str) -> go.Figure:
    num_c  = _numeric_cols(result_df)
    cat_c  = _cat_cols(result_df)
    date_c = _date_cols(result_df)

    if not num_c:
        return None

    # Date + numeric → line
    if date_c:
        return _chart_trend(result_df, query)

    # One categorical + one numeric → bar
    if cat_c and len(num_c) >= 1:
        return _chart_groupbar(result_df, query)

    # Two numeric, no category → scatter
    if len(num_c) >= 2 and not cat_c:
        x_col, y_col = num_c[0], num_c[1]
        fig = px.scatter(result_df, x=x_col, y=y_col,
                         color_discrete_sequence=[CLR_NORMAL], opacity=0.7)
        fig.update_traces(marker_size=7)
        fig = _apply_layout(fig, f"<b>{x_col}</b> vs <b>{y_col}</b>",
                            xlabel=x_col, ylabel=y_col)
        return fig

    # Single numeric → histogram
    return _chart_histogram(result_df, query)


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def generate_plotly_json(result: Any, query: str = "") -> Optional[str]:
    """
    Inspect the query intent and result shape, then generate the most
    appropriate Plotly chart. Returns JSON string or None.
    """
    logger.info(f"Generating visualization for intent: '{query[:60]}'")

    if not isinstance(result, pd.DataFrame) or result.empty:
        logger.info("Result is not a non-empty DataFrame — skipping chart.")
        return None

    df    = result.copy()
    intent = _intent(query)
    fig    = None

    try:
        if intent == "outlier":
            fig = _chart_outlier(df, query)

        elif intent == "correlation":
            # result_df from df.corr() is a square correlation matrix
            # Validate: square + all numeric
            num_c = _numeric_cols(df)
            is_corr_matrix = (
                len(num_c) > 1
                and df.shape[0] == df.shape[1]
                and set(df.index.astype(str)) == set(df.columns.astype(str))
            )
            # Also handle case where index holds feature names (from corr())
            if not is_corr_matrix and len(num_c) >= 2:
                # Try building corr from numeric columns
                corr = df[num_c].corr()
                fig  = _chart_heatmap(corr)
            else:
                fig = _chart_heatmap(df)

        elif intent == "trend":
            fig = _chart_trend(df, query)

        elif intent == "histogram":
            fig = _chart_histogram(df, query)

        elif intent == "cluster":
            fig = _chart_cluster(df)

        elif intent == "groupbar":
            fig = _chart_groupbar(df, query)

        elif intent == "summary":
            fig = _chart_summary(df)

        else:
            fig = _chart_auto(df, query)

        if fig is None:
            logger.warning("Chart function returned None — no visualization generated.")
            return None

        logger.info(f"Visualization generated successfully (intent={intent}).")
        return fig.to_json()

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        return None