"""
visualization.py
─────────────────────────────────────────────────────────────────────────────
Generates the RIGHT chart for the RIGHT query intent.

Intent → Chart mapping
  outlier / anomal / iqr          → Box plot  (all points, outliers red, IQR fences)
  correlation / multicollinear    → Annotated heatmap (diverging RdBu, r values on cells)
  explain + previous was corr     → Re-uses heatmap
  trend / over time               → Line chart with markers
  distribution / histogram        → Histogram with mean+median lines
  cluster / segment / pca         → Scatter coloured by cluster label
  compare / group by / average by → Grouped bar chart
  summary / describe / overview   → Horizontal mean±std bar
  explain (generic)               → Passes through to auto based on data shape
  default                         → Smart auto (bar / scatter / line / histogram)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Optional
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)

# ── colour palette ────────────────────────────────────────────────────────────
CLR_NORMAL  = "#4C9BE8"
CLR_OUTLIER = "#FF4B4B"
CLR_MEDIAN  = "#F5A623"
CLR_BG      = "rgba(0,0,0,0)"
FONT_COLOR  = "#FFFFFF"

LAYOUT_BASE = dict(
    paper_bgcolor=CLR_BG,
    plot_bgcolor ="rgba(14,17,23,1)",
    font         =dict(color=FONT_COLOR, size=13),
    margin       =dict(l=60, r=40, t=80, b=90),
    xaxis        =dict(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
    yaxis        =dict(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
)


# ── intent detection ──────────────────────────────────────────────────────────
def _intent(query: str) -> str:
    q = query.lower()
    if re.search(r"outlier|anomal|box|iqr|extreme", q):             return "outlier"
    if re.search(r"correlat|multicollinear|heatmap|vif|relationship between", q): return "correlation"
    if re.search(r"trend|over time|time series|monthly|daily|weekly|yearly", q):  return "trend"
    if re.search(r"distribut|histogram|frequency|spread of", q):    return "histogram"
    if re.search(r"cluster|segment|kmeans|pca", q):                 return "cluster"
    if re.search(r"compar|group by|by categor|by gender|by class|average.*by|mean.*by", q): return "groupbar"
    if re.search(r"summar|descri|overview|statistic", q):           return "summary"
    if re.search(r"explain|elaborate|tell me more|interpret|what does|what do|understand", q): return "explain"
    return "auto"


# ── helpers ───────────────────────────────────────────────────────────────────
def _num(df):  return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def _cat(df):  return [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
def _date(df): return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def _is_corr_matrix(df: pd.DataFrame) -> bool:
    nc = _num(df)
    if len(nc) < 2 or df.shape[0] != df.shape[1]:
        return False
    vals = df[nc].values
    return float(vals.min()) >= -1.01 and float(vals.max()) <= 1.01

def _apply_layout(fig, title: str, xlabel: str = None, ylabel: str = None) -> go.Figure:
    upd = dict(**LAYOUT_BASE, title=dict(text=title, font=dict(size=17, color=FONT_COLOR)))
    if xlabel:
        upd["xaxis"] = {**LAYOUT_BASE.get("xaxis", {}), "title": dict(text=xlabel, font=dict(size=13))}
    if ylabel:
        upd["yaxis"] = {**LAYOUT_BASE.get("yaxis", {}), "title": dict(text=ylabel, font=dict(size=13))}
    fig.update_layout(**upd)
    return fig


# ── 1. OUTLIER BOX PLOT ───────────────────────────────────────────────────────
def _chart_outlier(df: pd.DataFrame, query: str) -> go.Figure:
    nc = _num(df)
    if not nc:
        return None

    # Prefer column mentioned in query
    col = nc[0]
    for c in nc:
        if c.lower() in query.lower():
            col = c
            break

    vals = df[col].dropna()
    Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = int(((vals < lo) | (vals > hi)).sum())

    point_colors = [CLR_OUTLIER if (v < lo or v > hi) else CLR_NORMAL for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=vals, name=col,
        boxpoints="all", jitter=0.35, pointpos=0,
        marker=dict(color=point_colors, size=5, line=dict(width=0.8, color="#ffffff")),
        line=dict(color=CLR_NORMAL, width=2),
        fillcolor="rgba(76,155,232,0.12)",
        whiskerwidth=0.5,
        hovertemplate="<b>%{y:.3f}</b><extra></extra>",
    ))

    for val, label, dash in [
        (hi, f"Upper fence = {hi:.2f}",  "dash"),
        (Q3, f"Q3 = {Q3:.2f}",           "dot"),
        (vals.median(), f"Median = {vals.median():.2f}", "solid"),
        (Q1, f"Q1 = {Q1:.2f}",           "dot"),
        (lo, f"Lower fence = {lo:.2f}",  "dash"),
    ]:
        fig.add_hline(
            y=val,
            line=dict(color=CLR_MEDIAN, width=1.2, dash=dash),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=CLR_MEDIAN, size=11),
        )

    title = (
        f"📦 Outlier Analysis — <b>{col}</b><br>"
        f"<sup>"
        f"<span style='color:{CLR_OUTLIER}'>● {n_out} outlier(s) detected</span>"
        f"  |  IQR = {IQR:.2f}  |  Normal range [{lo:.2f}, {hi:.2f}]"
        f"  |  Total points = {len(vals)}"
        f"</sup>"
    )
    fig = _apply_layout(fig, title, ylabel=col)
    fig.update_layout(showlegend=False)
    return fig


# ── 2. CORRELATION HEATMAP ────────────────────────────────────────────────────
def _chart_heatmap(df: pd.DataFrame) -> go.Figure:
    nc = _num(df)
    if len(nc) < 2:
        return None

    # If already a correlation matrix use it; otherwise compute
    if _is_corr_matrix(df):
        corr   = df[nc]
        labels = nc
    else:
        corr   = df[nc].corr()
        labels = nc

    z    = corr.values.round(2)
    text = [[f"{v:.2f}" for v in row] for row in z]
    n    = len(labels)

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=max(8, min(12, 120 // n)), color="white"),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text="r value", font=dict(color=FONT_COLOR)),
            tickvals=[-1, -0.7, -0.5, 0, 0.5, 0.7, 1],
            ticktext=["-1.0<br>perfect neg.", "-0.7<br>strong neg.", "-0.5<br>mod. neg.",
                      "0<br>none", "+0.5<br>mod. pos.", "+0.7<br>strong pos.", "+1.0<br>perfect pos."],
            tickfont=dict(color=FONT_COLOR, size=10),
            len=0.85,
        ),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    # Highlight cells with |r| > 0.7 with a border-like shape
    for i in range(n):
        for j in range(n):
            if i != j and abs(z[i, j]) >= 0.7:
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    line=dict(color="yellow", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )

    title = (
        f"🔥 Correlation Heatmap — {n} variables<br>"
        "<sup style='color:#aaa'>"
        "🟥 Red = positive  🟦 Blue = negative  ⬜ White = no correlation  "
        "🟨 Yellow border = multicollinearity risk (|r| ≥ 0.7)"
        "</sup>"
    )
    fig = _apply_layout(fig, title)
    fig.update_layout(
        xaxis=dict(tickangle=-40, side="bottom", gridcolor="#2a2f3a"),
        yaxis=dict(autorange="reversed", gridcolor="#2a2f3a"),
        height=max(420, n * 52),
    )
    return fig


# ── 3. TREND LINE ─────────────────────────────────────────────────────────────
def _chart_trend(df: pd.DataFrame, query: str) -> go.Figure:
    dc = _date(df); nc = _num(df); cc = _cat(df)
    x_col = dc[0] if dc else (cc[0] if cc else None)
    if not x_col:
        df = df.reset_index(); x_col = "index"
    if not nc:
        return None
    y_col = nc[0]
    fig = px.line(df, x=x_col, y=y_col, markers=True,
                  color_discrete_sequence=[CLR_NORMAL])
    fig.update_traces(line_width=2.5, marker_size=7)
    fig = _apply_layout(fig, f"📈 Trend of <b>{y_col}</b> over <b>{x_col}</b>",
                        xlabel=x_col, ylabel=y_col)
    return fig


# ── 4. HISTOGRAM ──────────────────────────────────────────────────────────────
def _chart_histogram(df: pd.DataFrame, query: str) -> go.Figure:
    nc = _num(df)
    if not nc:
        return None
    col = nc[0]
    for c in nc:
        if c.lower() in query.lower():
            col = c; break
    vals    = df[col].dropna()
    mean_v  = float(vals.mean())
    med_v   = float(vals.median())
    std_v   = float(vals.std())
    skew_v  = float(vals.skew())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=40, name=col,
        marker_color=CLR_NORMAL, opacity=0.82,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    for val, label, color in [
        (mean_v, f"Mean = {mean_v:.2f}",    CLR_OUTLIER),
        (med_v,  f"Median = {med_v:.2f}",   CLR_MEDIAN),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(color=color, size=12))

    title = (
        f"📊 Distribution of <b>{col}</b><br>"
        f"<sup style='color:#aaa'>"
        f"Mean={mean_v:.2f}  |  Median={med_v:.2f}  |  Std={std_v:.2f}  "
        f"|  Skewness={skew_v:.2f}  |  N={len(vals)}"
        f"</sup>"
    )
    fig = _apply_layout(fig, title, xlabel=col, ylabel="Frequency (count)")
    return fig


# ── 5. CLUSTER SCATTER ────────────────────────────────────────────────────────
def _chart_cluster(df: pd.DataFrame) -> go.Figure:
    nc = _num(df)
    cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    if len(nc) < 2:
        return None
    x_col, y_col = nc[0], nc[1]
    fig = px.scatter(df, x=x_col, y=y_col,
                     color=cluster_col,
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     opacity=0.75)
    fig.update_traces(marker_size=7)
    fig = _apply_layout(fig, f"🔵 Cluster Analysis — <b>{x_col}</b> vs <b>{y_col}</b>",
                        xlabel=x_col, ylabel=y_col)
    return fig


# ── 6. GROUPED BAR ────────────────────────────────────────────────────────────
def _chart_groupbar(df: pd.DataFrame, query: str) -> go.Figure:
    cc = _cat(df); nc = _num(df)
    if not nc:
        return None
    x_col  = cc[0] if cc else df.columns[0]
    y_cols = nc[:3]
    colors = [CLR_NORMAL, CLR_OUTLIER, CLR_MEDIAN]
    fig = go.Figure()
    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Bar(
            x=df[x_col], y=df[y_col], name=y_col,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    title = (f"📊 <b>{y_cols[0]}</b> by <b>{x_col}</b>" if len(y_cols) == 1
             else f"📊 Grouped Comparison by <b>{x_col}</b>")
    fig = _apply_layout(fig, title, xlabel=x_col, ylabel="Value")
    return fig


# ── 7. SUMMARY BAR ───────────────────────────────────────────────────────────
def _chart_summary(df: pd.DataFrame) -> go.Figure:
    nc = _num(df)
    if not nc:
        return None
    stat_cols = [c for c in df.columns if c in ("mean", "50%", "std", "min", "max")]
    if stat_cols:
        means = df["mean"] if "mean" in df.columns else df[stat_cols[0]]
        stds  = df["std"]  if "std"  in df.columns else None
        fig   = go.Figure(go.Bar(
            x=means.index, y=means.values,
            error_y=dict(type="data", array=stds.values, visible=True) if stds is not None else None,
            marker_color=CLR_NORMAL,
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>",
        ))
        fig = _apply_layout(fig, "📋 Dataset Summary — Mean ± Std per Column",
                            xlabel="Column", ylabel="Mean Value")
        fig.update_layout(xaxis_tickangle=-35)
        return fig
    # plain fallback
    col = nc[0]
    fig = px.bar(df, y=df.index, x=col, orientation="h",
                 color_discrete_sequence=[CLR_NORMAL])
    fig = _apply_layout(fig, f"📋 Summary — {col}", xlabel=col)
    return fig


# ── AUTO FALLBACK ─────────────────────────────────────────────────────────────
def _chart_auto(df: pd.DataFrame, query: str) -> go.Figure:
    nc = _num(df); cc = _cat(df); dc = _date(df)
    if not nc:
        return None
    if dc:
        return _chart_trend(df, query)
    if cc and len(nc) >= 1:
        return _chart_groupbar(df, query)
    if len(nc) >= 2 and not cc:
        x_col, y_col = nc[0], nc[1]
        fig = px.scatter(df, x=x_col, y=y_col,
                         color_discrete_sequence=[CLR_NORMAL], opacity=0.7)
        fig.update_traces(marker_size=6)
        fig = _apply_layout(fig, f"<b>{x_col}</b> vs <b>{y_col}</b>",
                            xlabel=x_col, ylabel=y_col)
        return fig
    return _chart_histogram(df, query)


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def generate_plotly_json(
    result: Any,
    query: str = "",
    previous_intent: str = None,
) -> Optional[str]:
    """
    Generate the most appropriate Plotly chart for the query + data combination.

    Args:
        result          : DataFrame returned by execute_code
        query           : User's natural language question
        previous_intent : Intent of the immediately preceding turn (for "explain" queries)

    Returns:
        Plotly figure JSON string, or None if no chart is appropriate.
    """
    logger.info(f"Generating viz — query: '{query[:70]}' | prev_intent: {previous_intent}")

    if not isinstance(result, pd.DataFrame) or result.empty:
        return None

    df     = result.copy()
    intent = _intent(query)

    # "Explain me the above heatmap" → re-use previous intent if available
    if intent == "explain":
        intent = previous_intent if previous_intent else "auto"
        logger.info(f"Explain query — resolved intent to: {intent}")

    fig = None
    try:
        if intent == "outlier":
            fig = _chart_outlier(df, query)

        elif intent == "correlation":
            if _is_corr_matrix(df):
                fig = _chart_heatmap(df)
            else:
                nc = _num(df)
                if len(nc) >= 2:
                    corr = df[nc].corr()
                    fig  = _chart_heatmap(corr)

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
            logger.warning("Chart function returned None.")
            return None

        logger.info(f"Chart generated (intent={intent}).")
        return fig.to_json()

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        return None