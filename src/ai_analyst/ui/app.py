import streamlit as st
import sys
import os
import pandas as pd
import json

# --- PATH SETUP ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.ai_analyst.core.ingestion import ingest_file
from src.ai_analyst.core.query_understanding import generate_code, regenerate_code_with_error
from src.ai_analyst.core.execution_engine import execute_code
from src.ai_analyst.core.insights import generate_insight
from src.ai_analyst.core.visualization import generate_plotly_json
from src.ai_analyst.utils.exceptions import AppException, CodeExecutionError
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.stApp { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_context" not in st.session_state:
    st.session_state.df_context = None
if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None


# --- HELPERS ---
def safe_suggestions(suggestions):
    if not suggestions or not isinstance(suggestions, list):
        return []
    return [s for s in suggestions if isinstance(s, str) and s.strip()]


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


def api_key_is_set():
    return bool(getattr(settings, "GROQ_API_KEY", None))


def render_assistant_message(msg, msg_idx):
    if msg.get("insight"):
        st.markdown(msg["insight"])
    if msg.get("table"):
        try:
            st.dataframe(pd.DataFrame(msg["table"]), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render table: {e}")
    if msg.get("chart"):
        try:
            fig = json.loads(msg["chart"])
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg_idx}")
        except Exception as e:
            st.warning(f"Could not render chart: {e}")
    suggestions = safe_suggestions(msg.get("suggestions"))
    if suggestions:
        st.markdown("**Suggested Questions:**")
        cols = st.columns(min(len(suggestions), 3))
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"sugg_{msg_idx}_{i}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()


# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE (with self-correction loop)
# ---------------------------------------------------------------------------
MAX_CORRECTION_ATTEMPTS = 2

def run_analysis_pipeline(prompt: str):
    """
    Full pipeline: generate code → execute → self-correct on failure → insight → chart.
    Returns: (insight, suggestions, chart_json, table_data)
    """
    schema = st.session_state.df_context.columns
    df     = st.session_state.df_context.dataframe

    # Step 1: Generate initial code
    code = generate_code(schema, prompt)
    last_error = None

    for attempt in range(MAX_CORRECTION_ATTEMPTS + 1):
        try:
            result = execute_code(df, code)
            break  # Success
        except CodeExecutionError as e:
            last_error = str(e)
            logger.warning(f"Execution attempt {attempt + 1} failed: {last_error}")
            if attempt < MAX_CORRECTION_ATTEMPTS:
                logger.info("Requesting LLM self-correction...")
                try:
                    code = regenerate_code_with_error(schema, prompt, code, last_error)
                except Exception as gen_err:
                    raise CodeExecutionError(f"Self-correction code generation failed: {gen_err}")
            else:
                raise CodeExecutionError(
                    f"Analysis failed after {MAX_CORRECTION_ATTEMPTS + 1} attempts. "
                    f"Last error: {last_error}"
                )

    # Step 2: Generate insight + suggestions
    insight, suggestions = generate_insight(prompt, result)
    suggestions = safe_suggestions(suggestions)

    # Step 3: Generate chart (non-fatal)
    chart_json = None
    try:
        chart_json = generate_plotly_json(result, query=prompt)
    except Exception as e:
        logger.warning(f"Chart generation failed (non-fatal): {e}")

    # Step 4: Build table data
    table_data = safe_table_data(result)

    return insight, suggestions, chart_json, table_data


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
            st.warning("⚠️ No API key found. Please enter one to proceed.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        if uploaded_file.name != st.session_state.loaded_filename:
            try:
                with st.spinner("Reading CSV..."):
                    st.session_state.df_context = ingest_file(
                        uploaded_file.read(), uploaded_file.name
                    )
                    st.session_state.messages = []
                    st.session_state.loaded_filename = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                logger.error(f"File ingestion error: {e}")
                st.session_state.df_context = None
                st.session_state.loaded_filename = None

    if st.session_state.df_context:
        st.caption(f"📄 {st.session_state.loaded_filename}")
        st.caption(f"Rows: {len(st.session_state.df_context.dataframe)}")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("🤖 AI Data Analyst")

# 1. RENDER CHAT HISTORY
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg, idx)
        else:
            st.markdown(msg["content"])

# 2. PROCESSING LOGIC
if st.session_state.messages and st.session_state.df_context:
    last_message = st.session_state.messages[-1]

    if last_message["role"] == "user":
        prompt = last_message["content"]

        if not api_key_is_set():
            st.error("❌ No API key configured. Please enter your Groq API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing... (may take a moment for complex queries)"):
                    try:
                        insight, suggestions, chart_json, table_data = run_analysis_pipeline(prompt)

                        if insight:
                            st.markdown(insight)
                        if table_data:
                            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                        if chart_json:
                            try:
                                fig = json.loads(chart_json)
                                st.plotly_chart(fig, use_container_width=True, key="live_chart")
                            except Exception as ce:
                                logger.warning(f"Chart display error: {ce}")

                        if suggestions:
                            st.markdown("**Suggested Questions:**")
                            temp_cols = st.columns(min(len(suggestions), 3))
                            for i, sugg in enumerate(suggestions):
                                btn_key = f"live_sugg_{len(st.session_state.messages)}_{i}"
                                if temp_cols[i].button(sugg, key=btn_key):
                                    st.session_state.messages.append(
                                        {"role": "user", "content": sugg}
                                    )
                                    st.rerun()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "insight": insight,
                            "table": table_data,
                            "chart": chart_json,
                            "suggestions": suggestions,
                        })

                    except CodeExecutionError as e:
                        err_msg = (
                            "🔧 I tried multiple times but couldn't execute this analysis. "
                            "Please try rephrasing or asking a simpler question."
                        )
                        st.error(err_msg)
                        logger.warning(f"Final CodeExecutionError: {e.message}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "insight": err_msg,
                            "table": None,
                            "chart": None,
                            "suggestions": [
                                "Give me a full statistical summary",
                                "Show correlation between numeric columns",
                                "What are the top 10 rows?",
                            ],
                        })
                    except AppException as e:
                        st.error(f"❌ {e.message}")
                        logger.error(f"AppException: {e.message}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "insight": f"❌ {e.message}",
                            "table": None, "chart": None, "suggestions": [],
                        })
                    except Exception as e:
                        st.error("🚫 Something went wrong. Please try a simpler question.")
                        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "insight": "🚫 Something went wrong.",
                            "table": None, "chart": None, "suggestions": [],
                        })

            # Rerun OUTSIDE the chat_message context manager
            st.rerun()

# 3. SAMPLE QUESTIONS (only when chat is empty)
if st.session_state.df_context and not st.session_state.messages:
    st.markdown("### 💡 Try asking:")

    numeric_cols = [
        c.name for c in st.session_state.df_context.columns
        if c.dtype in ["integer", "float"]
    ]
    date_cols = [
        c.name for c in st.session_state.df_context.columns
        if "date" in c.name.lower()
    ]

    samples = []
    if numeric_cols:
        samples.append(f"Show outliers in {numeric_cols[0]}")
        samples.append("Detect anomalies using Isolation Forest")
        if len(numeric_cols) > 1:
            samples.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        samples.append("Cluster the data into 3 groups")
    if date_cols:
        samples.append(f"Analyze trends over {date_cols[0]}")
    samples.append("Give me a full statistical summary")

    for i in range(0, len(samples), 3):
        c = st.columns(3)
        for j in range(i, min(i + 3, len(samples))):
            if c[j - i].button(samples[j], key=f"sample_{j}"):
                st.session_state.messages.append({"role": "user", "content": samples[j]})
                st.rerun()

    st.markdown("---")

# 4. CHAT INPUT
if prompt := st.chat_input("Ask a question about your data..."):
    if not st.session_state.df_context:
        st.warning("⚠️ Please upload a CSV file first.")
    elif not api_key_is_set():
        st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()