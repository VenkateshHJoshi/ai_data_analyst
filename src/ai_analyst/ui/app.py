import streamlit as st
import sys
import os
import pandas as pd
import json
import re


from dotenv import load_dotenv
load_dotenv()

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.ai_analyst.core.ingestion import ingest_file
from src.ai_analyst.core.query_understanding import generate_code, regenerate_code_with_error
from src.ai_analyst.core.execution_engine import execute_code
from src.ai_analyst.core.insights import generate_insight
from src.ai_analyst.core.visualization import generate_plotly_json
from src.ai_analyst.utils.exceptions import AppException, CodeExecutionError
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.insight-box {
    background: rgba(76,155,232,0.08);
    border-left: 3px solid #4C9BE8;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 10px;
    line-height: 1.75;
}
.restore-banner {
    background: rgba(255,75,75,0.12);
    border: 1px solid rgba(255,75,75,0.4);
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
for key, default in [
    ("messages",         []),
    ("df_context",       None),
    ("original_df",      None),   # snapshot taken on first upload — never mutated
    ("loaded_filename",  None),
    ("last_intent",      None),   # intent of the most recent successful analysis
    ("dataset_modified", False),  # True when a destructive op has been run
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def api_key_is_set() -> bool:
    return bool(getattr(settings, "GROQ_API_KEY", None))


def safe_suggestions(s):
    return [x for x in (s or []) if isinstance(x, str) and x.strip()]


def _detect_destructive(query: str) -> bool:
    """Return True if the query is likely to filter/remove/clean rows."""
    return bool(re.search(
        r"remov|drop|delet|filter|clean|exclud|without|strip|trim|sanitiz|deduplic",
        query, re.IGNORECASE
    ))


def _intent_from_query(query: str) -> str:
    """Mirror of visualization._intent() — kept here to avoid circular imports."""
    q = query.lower()
    if re.search(r"outlier|anomal|box|iqr|extreme", q):                         return "outlier"
    if re.search(r"correlat|multicollinear|heatmap|vif|relationship between", q): return "correlation"
    if re.search(r"trend|over time|time series|monthly|daily|weekly|yearly", q):  return "trend"
    if re.search(r"distribut|histogram|frequency|spread of", q):                  return "histogram"
    if re.search(r"cluster|segment|kmeans|pca", q):                               return "cluster"
    if re.search(r"compar|group by|by categor|by gender|average.*by|mean.*by", q):"groupbar"
    if re.search(r"summar|descri|overview|statistic", q):                         return "summary"
    if re.search(r"explain|elaborate|tell me more|interpret|what does|understand", q): return "explain"
    return "auto"


# Intents where showing the raw data table adds noise, not value
_SUPPRESS_TABLE_FOR = {"correlation", "explain", "summary", "histogram", "trend"}


def should_show_table(intent: str, result) -> bool:
    """Show table only when it genuinely helps (outlier rows, group comparisons, etc.)."""
    if intent in _SUPPRESS_TABLE_FOR:
        return False
    if not isinstance(result, pd.DataFrame):
        return True   # scalar result → always show
    # For large DataFrames that are just the full dataset, suppress
    if isinstance(result, pd.DataFrame) and result.shape[0] > 200:
        return False
    return True


def safe_table_data(result):
    try:
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return None
            return result.head(50).to_dict(orient="records")
        elif isinstance(result, pd.Series):
            return result.reset_index().head(50).to_dict(orient="records")
        elif result is not None:
            return [{"Value": str(result)}]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# RENDER HELPERS
# ---------------------------------------------------------------------------
def render_insight_box(text: str):
    """Render insight in a styled blue-bordered box."""
    if text:
        st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def render_assistant_message(msg: dict, msg_idx: int):
    """Render a saved assistant message from history."""

    # Insight box
    if msg.get("insight"):
        render_insight_box(msg["insight"])

    # Table — only if flagged to show
    if msg.get("show_table") and msg.get("table"):
        with st.expander("📋 View Data Table", expanded=False):
            try:
                st.dataframe(pd.DataFrame(msg["table"]), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render table: {e}")

    # Chart
    if msg.get("chart"):
        try:
            fig = json.loads(msg["chart"])
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg_idx}")
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

    # Restore dataset banner
    if msg.get("show_restore"):
        st.markdown('<div class="restore-banner">⚠️ This analysis may have filtered your dataset.</div>',
                    unsafe_allow_html=True)
        if st.button("↩️ Restore Original Dataset", key=f"restore_{msg_idx}"):
            if st.session_state.original_df is not None:
                st.session_state.df_context.dataframe = st.session_state.original_df.copy()
                st.session_state.dataset_modified = False
                st.success("✅ Dataset restored to original.")
                st.rerun()

    # Suggestions
    suggestions = safe_suggestions(msg.get("suggestions"))
    if suggestions:
        st.markdown("**💬 Suggested next questions:**")
        cols = st.columns(min(len(suggestions), 3))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sugg_{msg_idx}_{i}"):
                st.session_state.messages.append({"role": "user", "content": s})
                st.rerun()


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
MAX_CORRECTION_ATTEMPTS = 2


def run_analysis_pipeline(prompt: str):
    """
    generate code → execute (with self-correction) → insight → chart → table decision.

    Returns dict with keys:
        insight, suggestions, chart_json, table_data, show_table,
        show_restore, intent
    """
    schema = st.session_state.df_context.columns
    df     = st.session_state.df_context.dataframe

    # ── Step 1: Generate + execute code (with self-correction) ────────────────
    code       = generate_code(schema, prompt)
    last_error = None

    for attempt in range(MAX_CORRECTION_ATTEMPTS + 1):
        try:
            result = execute_code(df, code)
            break
        except CodeExecutionError as e:
            last_error = str(e)
            logger.warning(f"Execution attempt {attempt+1} failed: {last_error}")
            if attempt < MAX_CORRECTION_ATTEMPTS:
                try:
                    code = regenerate_code_with_error(schema, prompt, code, last_error)
                except Exception as gen_err:
                    raise CodeExecutionError(f"Self-correction failed: {gen_err}")
            else:
                raise CodeExecutionError(
                    f"Analysis failed after {MAX_CORRECTION_ATTEMPTS+1} attempts. "
                    f"Last error: {last_error}"
                )

    # ── Step 2: Intent detection ───────────────────────────────────────────────
    intent = _intent_from_query(prompt)
    # For "explain" queries, fall back to the last successful intent
    if intent == "explain" and st.session_state.last_intent:
        resolved_intent = st.session_state.last_intent
    else:
        resolved_intent = intent
        if intent != "explain":
            st.session_state.last_intent = intent

    # ── Step 3: Insight (with conversation history for "explain" queries) ──────
    history = [
        {"role": m["role"], "insight": m.get("insight", "")}
        for m in st.session_state.messages
        if m.get("role") == "assistant"
    ]
    insight, suggestions = generate_insight(
        question=prompt,
        result=result,
        conversation_history=history,
    )
    suggestions = safe_suggestions(suggestions)

    # ── Step 4: Chart ──────────────────────────────────────────────────────────
    chart_json = None
    try:
        chart_json = generate_plotly_json(
            result,
            query=prompt,
            previous_intent=st.session_state.last_intent,
        )
    except Exception as e:
        logger.warning(f"Chart generation non-fatal error: {e}")

    # ── Step 5: Table decision ─────────────────────────────────────────────────
    show_table = should_show_table(resolved_intent, result)
    table_data = safe_table_data(result) if show_table else None

    # ── Step 6: Destructive operation detection ────────────────────────────────
    is_destructive = _detect_destructive(prompt)
    if is_destructive:
        st.session_state.dataset_modified = True

    return {
        "insight":      insight,
        "suggestions":  suggestions,
        "chart_json":   chart_json,
        "table_data":   table_data,
        "show_table":   show_table,
        "show_restore": is_destructive,
        "intent":       resolved_intent,
    }


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setup")

    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        st.success("✅ API Key loaded from .env")
        settings.GROQ_API_KEY = env_key
    else:
        user_key = st.text_input("Enter Groq API Key", type="password")
        if user_key:
            settings.GROQ_API_KEY = user_key
        else:
            st.warning("⚠️ No API key. Please enter one above.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file and uploaded_file.name != st.session_state.loaded_filename:
        try:
            with st.spinner("Reading CSV..."):
                ctx = ingest_file(uploaded_file.read(), uploaded_file.name)
                st.session_state.df_context      = ctx
                st.session_state.original_df     = ctx.dataframe.copy()  # immutable backup
                st.session_state.messages        = []
                st.session_state.loaded_filename = uploaded_file.name
                st.session_state.last_intent     = None
                st.session_state.dataset_modified = False
            st.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.df_context = None

    if st.session_state.df_context:
        df_now = st.session_state.df_context.dataframe
        st.caption(f"📄 {st.session_state.loaded_filename}")
        st.caption(f"Rows: {len(df_now)}  |  Cols: {df_now.shape[1]}")

        if st.session_state.dataset_modified:
            orig_rows = len(st.session_state.original_df)
            curr_rows = len(df_now)
            st.warning(f"⚠️ Dataset modified ({curr_rows}/{orig_rows} rows)")
            if st.button("↩️ Restore Original Dataset"):
                st.session_state.df_context.dataframe = st.session_state.original_df.copy()
                st.session_state.dataset_modified = False
                st.success("✅ Dataset restored!")
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages    = []
            st.session_state.last_intent = None
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("🤖 AI Data Analyst")

# ── 1. Render chat history ────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg, idx)
        else:
            st.markdown(msg["content"])


# ── 2. Process the latest user message ───────────────────────────────────────
if st.session_state.messages and st.session_state.df_context:
    last = st.session_state.messages[-1]

    if last["role"] == "user":
        prompt = last["content"]

        if not api_key_is_set():
            st.error("❌ No API key. Please enter your Groq API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing…"):
                    try:
                        out = run_analysis_pipeline(prompt)

                        # Render insight
                        render_insight_box(out["insight"])

                        # Render table (collapsible, only when relevant)
                        if out["show_table"] and out["table_data"]:
                            with st.expander("📋 View Data Table", expanded=True):
                                st.dataframe(pd.DataFrame(out["table_data"]),
                                             use_container_width=True)

                        # Render chart
                        if out["chart_json"]:
                            try:
                                fig = json.loads(out["chart_json"])
                                st.plotly_chart(fig, use_container_width=True, key="live_chart")
                            except Exception as ce:
                                logger.warning(f"Chart render error: {ce}")

                        # Restore banner
                        if out["show_restore"]:
                            st.markdown(
                                '<div class="restore-banner">'
                                '⚠️ This query may have filtered or removed rows from your dataset.'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        # Suggestions
                        if out["suggestions"]:
                            st.markdown("**💬 Suggested next questions:**")
                            s_cols = st.columns(min(len(out["suggestions"]), 3))
                            for i, sugg in enumerate(out["suggestions"]):
                                key = f"live_sugg_{len(st.session_state.messages)}_{i}"
                                if s_cols[i].button(sugg, key=key):
                                    st.session_state.messages.append(
                                        {"role": "user", "content": sugg}
                                    )
                                    st.rerun()

                        # Save to history
                        st.session_state.messages.append({
                            "role":         "assistant",
                            "insight":      out["insight"],
                            "table":        out["table_data"],
                            "show_table":   out["show_table"],
                            "chart":        out["chart_json"],
                            "suggestions":  out["suggestions"],
                            "show_restore": out["show_restore"],
                            "intent":       out["intent"],
                        })

                    except CodeExecutionError as e:
                        err = (
                            "🔧 I tried multiple times but couldn't execute this analysis. "
                            "Try rephrasing or breaking the question into smaller steps."
                        )
                        st.error(err)
                        logger.warning(f"CodeExecutionError: {e}")
                        st.session_state.messages.append({
                            "role": "assistant", "insight": err,
                            "table": None, "show_table": False, "chart": None,
                            "suggestions": [
                                "Give me a full statistical summary",
                                "Show correlation between numeric columns",
                                "Show outliers in the data",
                            ],
                            "show_restore": False, "intent": "general",
                        })

                    except AppException as e:
                        st.error(f"❌ {e.message}")
                        st.session_state.messages.append({
                            "role": "assistant", "insight": f"❌ {e.message}",
                            "table": None, "show_table": False, "chart": None,
                            "suggestions": [], "show_restore": False, "intent": "general",
                        })

                    except Exception as e:
                        st.error("🚫 Something went wrong. Please try a simpler question.")
                        logger.error(f"Unhandled error: {e}", exc_info=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "insight": "🚫 Something went wrong.",
                            "table": None, "show_table": False, "chart": None,
                            "suggestions": [], "show_restore": False, "intent": "general",
                        })

            # Rerun OUTSIDE the chat_message context manager
            st.rerun()


# ── 3. Sample questions (only when chat is empty) ─────────────────────────────
if st.session_state.df_context and not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    nc = [c.name for c in st.session_state.df_context.columns if c.dtype in ("integer", "float")]
    dc = [c.name for c in st.session_state.df_context.columns if "date" in c.name.lower()]

    samples = []
    if nc:
        samples.append(f"Show outliers in {nc[0]}")
        samples.append("What is the correlation between all numeric columns?")
        samples.append(f"Show the distribution of {nc[0]}")
        if len(nc) > 1:
            samples.append(f"Cluster the data into 3 groups")
    if dc:
        samples.append(f"Analyze trends over {dc[0]}")
    samples.append("Give me a full statistical summary of the data")

    for i in range(0, len(samples), 3):
        cols = st.columns(3)
        for j in range(i, min(i + 3, len(samples))):
            if cols[j - i].button(samples[j], key=f"sample_{j}"):
                st.session_state.messages.append({"role": "user", "content": samples[j]})
                st.rerun()
    st.markdown("---")


# ── 4. Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your data…"):
    if not st.session_state.df_context:
        st.warning("⚠️ Please upload a CSV file first.")
    elif not api_key_is_set():
        st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()