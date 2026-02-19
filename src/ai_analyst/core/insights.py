"""
insights.py
─────────────────────────────────────────────────────────────────────────────
Generates a detailed, query-aware textual insight + follow-up suggestions.

Key improvements over previous version:
 - Lazy Groq client (respects runtime API key changes)
 - Query-type detection → tailored system prompt per analysis type
 - Sends richer data context (stats + shape, not just 5 rows)
 - Returns detailed multi-sentence insight, not a single vague line
 - Suggestions are contextually relevant to the analysis performed
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
from groq import Groq
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────
def _get_client() -> Groq:
    """Lazily create the Groq client so API-key changes in sidebar take effect."""
    return Groq(api_key=settings.GROQ_API_KEY)


def _query_type(query: str) -> str:
    q = query.lower()
    if re.search(r"outlier|anomal|extreme|iqr|box", q):           return "outlier"
    if re.search(r"correlat|multicollinear|heatmap|vif", q):      return "correlation"
    if re.search(r"trend|over time|time series|monthly|daily", q): return "trend"
    if re.search(r"distribut|histogram|frequency|spread", q):      return "distribution"
    if re.search(r"cluster|segment|group|kmeans", q):              return "cluster"
    if re.search(r"summar|descri|overview|statistic", q):          return "summary"
    if re.search(r"compar|average.*by|mean.*by|by categor", q):    return "comparison"
    return "general"


def _build_data_context(result) -> str:
    """
    Build a rich text representation of the result for the LLM.
    Includes shape, dtypes, stats and sample rows.
    """
    if isinstance(result, pd.DataFrame):
        parts = []
        parts.append(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
        parts.append(f"Columns: {', '.join(result.columns.tolist())}")

        num_cols = result.select_dtypes(include="number").columns.tolist()
        if num_cols:
            stats = result[num_cols].describe().round(3).to_string()
            parts.append(f"\nNumeric statistics:\n{stats}")

        # Show up to 10 sample rows
        sample = result.head(10).to_string(index=False)
        parts.append(f"\nSample rows (up to 10):\n{sample}")
        return "\n".join(parts)

    elif isinstance(result, (int, float)):
        return f"Result value: {result}"
    else:
        return f"Result:\n{str(result)[:800]}"


# ── per-type system prompts ───────────────────────────────────────────────────
_SYSTEM_PROMPTS = {
    "outlier": (
        "You are an expert data analyst specialising in anomaly detection. "
        "The user asked about outliers. Explain:\n"
        "1. How many outliers were found and in which column(s).\n"
        "2. What the IQR method is (briefly) and why these values are outliers.\n"
        "3. What the outliers could mean for real-world data quality or business decisions.\n"
        "4. Whether the outliers look like data errors or genuine extreme values.\n"
        "Be specific using the actual numbers from the result. Write 3–4 sentences."
    ),
    "correlation": (
        "You are an expert data analyst specialising in multicollinearity and feature relationships. "
        "Explain:\n"
        "1. Which pairs of variables have the strongest positive correlation (r > 0.7).\n"
        "2. Which pairs have the strongest negative correlation (r < -0.7).\n"
        "3. Why high correlation matters for modelling (multicollinearity risk).\n"
        "4. Which variables appear to be redundant based on the correlation matrix.\n"
        "Be specific using actual column names and r values. Write 3–5 sentences."
    ),
    "trend": (
        "You are an expert data analyst specialising in time series. "
        "Explain:\n"
        "1. The overall direction of the trend (increasing / decreasing / stable).\n"
        "2. Any notable peaks, troughs or inflection points and when they occurred.\n"
        "3. Whether the trend looks seasonal or irregular.\n"
        "4. What the trend might imply for forecasting.\n"
        "Be specific with actual values and time periods. Write 3–5 sentences."
    ),
    "distribution": (
        "You are an expert data analyst specialising in statistical distributions. "
        "Explain:\n"
        "1. The shape of the distribution (normal, skewed left/right, bimodal, etc.).\n"
        "2. Key statistics: mean, median, std dev and what they tell us.\n"
        "3. Whether skewness or heavy tails indicate data quality issues.\n"
        "4. How the distribution affects downstream analysis or modelling.\n"
        "Be specific with actual numbers. Write 3–5 sentences."
    ),
    "cluster": (
        "You are an expert data analyst specialising in clustering and segmentation. "
        "Explain:\n"
        "1. How many clusters were found and their approximate sizes.\n"
        "2. What distinguishes each cluster based on the numeric values shown.\n"
        "3. What these segments might represent in the real world.\n"
        "4. How actionable these clusters are for decision-making.\n"
        "Be specific using actual cluster statistics. Write 3–5 sentences."
    ),
    "summary": (
        "You are an expert data analyst. "
        "Give a thorough summary of the dataset:\n"
        "1. Key statistical highlights (highest/lowest mean, most variable column, etc.).\n"
        "2. Which columns have the widest range and what that implies.\n"
        "3. Any obvious data quality concerns (very high std, suspicious min/max).\n"
        "4. Which columns would be most useful for predictive modelling.\n"
        "Be specific with actual column names and numbers. Write 4–6 sentences."
    ),
    "comparison": (
        "You are an expert data analyst specialising in comparative analysis. "
        "Explain:\n"
        "1. Which category/group has the highest and lowest values.\n"
        "2. The magnitude of difference between the best and worst groups.\n"
        "3. Whether the differences are likely statistically meaningful.\n"
        "4. What the comparison tells us about the underlying data.\n"
        "Be specific with actual group names and numbers. Write 3–5 sentences."
    ),
    "general": (
        "You are a senior data analyst. Answer the user's question clearly and concisely. "
        "Refer specifically to the result data provided. "
        "Write 2–4 informative sentences and highlight the most important finding."
    ),
}


def _build_suggestions(query_type: str, result, query: str) -> list[str]:
    """
    Return 2–3 contextually relevant follow-up suggestions based on analysis type.
    These are deterministic (no LLM call needed) and always useful.
    """
    base_col = ""
    if isinstance(result, pd.DataFrame):
        num_cols = result.select_dtypes(include="number").columns.tolist()
        base_col = num_cols[0] if num_cols else ""

    suggestions_map = {
        "outlier": [
            f"Show the distribution of {base_col}" if base_col else "Show the data distribution",
            f"What is the correlation between all numeric columns?",
            f"Give me a full statistical summary of the data",
        ],
        "correlation": [
            "Which columns have the highest variance?",
            "Show outliers in the most correlated columns",
            "Give me a full statistical summary of the data",
        ],
        "trend": [
            "What is the average value per month?",
            "Show outliers in this time series",
            "What is the overall summary of this data?",
        ],
        "distribution": [
            f"Show outliers in {base_col}" if base_col else "Show outliers",
            "What is the correlation between all numeric columns?",
            "Give me a full statistical summary",
        ],
        "cluster": [
            "What are the average values per cluster?",
            "Show the correlation between numeric columns",
            "How many outliers exist in each cluster?",
        ],
        "summary": [
            "Show outliers across all numeric columns",
            "What is the correlation between all numeric columns?",
            "Which column has the most skewed distribution?",
        ],
        "comparison": [
            "Show the distribution of values across all groups",
            "What is the overall average across all categories?",
            "Which group contains the most outliers?",
        ],
        "general": [
            "Give me a full statistical summary of the data",
            "Show outliers in the numeric columns",
            "What is the correlation between all numeric columns?",
        ],
    }
    return suggestions_map.get(query_type, suggestions_map["general"])[:3]


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def generate_insight(question: str, result) -> tuple[str, list[str]]:
    """
    Generate a detailed, query-aware insight and follow-up suggestions.

    Args:
        question : The original user question.
        result   : Output from execute_code (DataFrame, scalar, etc.)

    Returns:
        (insight_text: str, suggestions: list[str])
    """
    logger.info(f"Generating insight for: '{question[:60]}'")

    qtype       = _query_type(question)
    data_ctx    = _build_data_context(result)
    system_msg  = _SYSTEM_PROMPTS.get(qtype, _SYSTEM_PROMPTS["general"])
    suggestions = _build_suggestions(qtype, result, question)

    user_msg = (
        f"User Question: {question}\n\n"
        f"Analysis Result:\n{data_ctx}\n\n"
        "Please provide your detailed insight now."
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            model=settings.DEFAULT_MODEL,
            temperature=0.4,
            max_tokens=400,      # enough for 4–6 sentences
        )
        insight = response.choices[0].message.content.strip()

        # Clean up if LLM prefixes with "Insight:" label
        insight = re.sub(r"^(insight\s*:\s*)", "", insight, flags=re.IGNORECASE)
        logger.info("Insight generated successfully.")
        return insight, suggestions

    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        # Graceful fallback — at least show basic stats
        fallback = _fallback_insight(result, question)
        return fallback, suggestions


def _fallback_insight(result, question: str) -> str:
    """Generate a basic insight from data stats without an LLM call."""
    if isinstance(result, pd.DataFrame) and not result.empty:
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col  = num_cols[0]
            mean = result[col].mean()
            mn   = result[col].min()
            mx   = result[col].max()
            n    = len(result)
            return (
                f"The analysis returned **{n} rows** for the query '{question}'. "
                f"Column **{col}** ranges from **{mn:.2f}** to **{mx:.2f}** "
                f"with a mean of **{mean:.2f}**."
            )
        return f"The analysis returned {len(result)} rows with columns: {', '.join(result.columns[:5])}."
    return f"Analysis completed. Result: {str(result)[:200]}"