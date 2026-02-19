"""
insights.py
─────────────────────────────────────────────────────────────────────────────
Generates detailed, query-aware insights + follow-up suggestions.

Key features:
 - Conversation memory: "explain above heatmap" uses last assistant result
 - Query-type detection with "explain" intent that re-uses previous context
 - max_tokens=900 so explanations never get cut off
 - Rich data context (full describe() stats, not just 5 rows)
 - Per-type specialist system prompts (6-10 sentence explanations)
 - Deterministic contextual suggestions (no extra LLM call)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
from groq import Groq
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)


# ── lazy client ───────────────────────────────────────────────────────────────
def _get_client() -> Groq:
    return Groq(api_key=settings.GROQ_API_KEY)


# ── intent / query-type detection ─────────────────────────────────────────────
def _query_type(query: str) -> str:
    q = query.lower()
    # "explain" / "describe" the previous result → re-use last context
    if re.search(r"explain|describe|what does|what do|elaborate|tell me more|interpret|understand", q):
        return "explain"
    if re.search(r"outlier|anomal|extreme|iqr|box", q):            return "outlier"
    if re.search(r"correlat|multicollinear|heatmap|vif|relationship between", q): return "correlation"
    if re.search(r"trend|over time|time series|monthly|daily|weekly|yearly", q):   return "trend"
    if re.search(r"distribut|histogram|frequency|spread", q):       return "distribution"
    if re.search(r"cluster|segment|kmeans|pca", q):                 return "cluster"
    if re.search(r"summar|overview|statistic|describe all", q):     return "summary"
    if re.search(r"compar|average.*by|mean.*by|by categor|by group", q): return "comparison"
    return "general"


# ── data context builder ──────────────────────────────────────────────────────
def _build_data_context(result, max_rows: int = 20) -> str:
    """
    Build a rich text representation of the result for the LLM.
    For correlation matrices we include the full matrix (all pairs).
    For other DataFrames we include describe() stats + sample rows.
    """
    if not isinstance(result, pd.DataFrame):
        return f"Result value: {str(result)[:800]}"

    if result.empty:
        return "The result DataFrame is empty."

    parts = []
    parts.append(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
    parts.append(f"Columns: {', '.join(result.columns.astype(str).tolist())}")

    num_cols = result.select_dtypes(include="number").columns.tolist()

    # Detect if this looks like a correlation matrix (square, all numeric, values -1..1)
    is_corr = (
        result.shape[0] == result.shape[1]
        and len(num_cols) == result.shape[1]
        and result.values.min() >= -1.0
        and result.values.max() <= 1.0
    )

    if is_corr:
        parts.append("\nFull Correlation Matrix (r values):")
        parts.append(result.round(3).to_string())
        # Extract strong pairs
        strong = []
        cols = result.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = result.iloc[i, j]
                if abs(r) >= 0.5:
                    strong.append((cols[i], cols[j], round(r, 3)))
        if strong:
            strong.sort(key=lambda x: abs(x[2]), reverse=True)
            lines = [f"  {a} ↔ {b}: r = {r}" for a, b, r in strong[:15]]
            parts.append("\nStrong correlations (|r| ≥ 0.5):\n" + "\n".join(lines))
    else:
        if num_cols:
            parts.append(f"\nStatistical summary:\n{result[num_cols].describe().round(3).to_string()}")
        sample = result.head(max_rows).to_string(index=False)
        parts.append(f"\nSample rows (up to {max_rows}):\n{sample}")

    return "\n".join(parts)


# ── system prompts ────────────────────────────────────────────────────────────
_SYSTEM_PROMPTS = {

    "explain": (
        "You are a senior data analyst and teacher. The user wants a detailed explanation of "
        "the analysis result shown above. Provide a thorough, educational explanation covering:\n"
        "1. What this type of analysis is and what it measures.\n"
        "2. How to read the chart or table (axes, colours, values).\n"
        "3. The key findings from the specific data — quote actual numbers and column names.\n"
        "4. What those findings mean practically (implications, risks, actionable insights).\n"
        "5. Any caveats or limitations of this analysis method.\n"
        "Write at least 6–8 sentences. Be specific, educational and actionable."
    ),

    "outlier": (
        "You are an expert data analyst specialising in anomaly and outlier detection.\n"
        "Provide a thorough explanation covering:\n"
        "1. WHAT: How many outliers were found, in which column(s), and what their actual values are.\n"
        "2. METHOD: Explain the IQR method — what Q1, Q3, IQR, and the 1.5×IQR fence rule mean.\n"
        "3. WHY THEY STAND OUT: Compare outlier values to the median and normal range.\n"
        "4. REAL-WORLD MEANING: Are these likely data entry errors, measurement issues, or genuine "
        "extreme cases? What impact do they have on averages and models?\n"
        "5. RECOMMENDATION: Should these outliers be removed, capped, or kept? Why?\n"
        "Write 6–8 sentences. Use actual numbers from the data."
    ),

    "correlation": (
        "You are an expert data analyst specialising in correlation and multicollinearity.\n"
        "Provide a comprehensive explanation covering:\n"
        "1. WHAT IS CORRELATION: Explain the Pearson r coefficient and what -1, 0, +1 mean.\n"
        "2. HOW TO READ THE HEATMAP: Explain colours (red=positive, blue=negative, white=none).\n"
        "3. STRONGEST PAIRS: Name the top 3-5 strongest correlations with their exact r values.\n"
        "4. MULTICOLLINEARITY RISK: Which pairs have |r| > 0.7 and why this is a problem in "
        "regression models (inflated standard errors, unstable coefficients).\n"
        "5. REDUNDANT FEATURES: Which columns carry similar information and could be dropped.\n"
        "6. WEAK/NO CORRELATION: Any pairs with r ≈ 0 that are truly independent.\n"
        "7. RECOMMENDATION: What to do with highly correlated features before modelling.\n"
        "Write 8–10 sentences. Quote actual column names and r values throughout."
    ),

    "trend": (
        "You are an expert data analyst specialising in time series analysis.\n"
        "Provide a thorough explanation covering:\n"
        "1. OVERALL DIRECTION: Is the trend increasing, decreasing, or flat? By how much?\n"
        "2. KEY POINTS: Identify peaks, troughs and inflection points with their dates/values.\n"
        "3. RATE OF CHANGE: Is the change gradual or sudden? Consistent or erratic?\n"
        "4. SEASONALITY: Any repeating patterns (monthly, quarterly)?\n"
        "5. ANOMALIES: Any unexpected spikes or drops worth investigating?\n"
        "6. FORECAST IMPLICATION: Based on this trend, what might happen next?\n"
        "Write 6–8 sentences. Use actual dates and values from the data."
    ),

    "distribution": (
        "You are an expert data analyst specialising in statistical distributions.\n"
        "Provide a thorough explanation covering:\n"
        "1. WHAT IS A DISTRIBUTION: What the histogram shows (frequency of values).\n"
        "2. SHAPE: Is it normal (bell curve), right-skewed, left-skewed, or bimodal? What causes this?\n"
        "3. KEY STATS: Explain what mean, median and std dev tell us — and why mean ≠ median "
        "implies skewness.\n"
        "4. OUTLIERS: Are there long tails? Values far from the centre?\n"
        "5. DATA QUALITY: Does the distribution look realistic, or are there suspicious gaps/spikes?\n"
        "6. MODELLING IMPACT: How does this distribution shape affect statistical tests or ML models?\n"
        "Write 6–8 sentences. Use actual numbers from the data."
    ),

    "cluster": (
        "You are an expert data analyst specialising in clustering and customer segmentation.\n"
        "Provide a thorough explanation covering:\n"
        "1. WHAT IS CLUSTERING: What the algorithm does and how it groups similar records.\n"
        "2. CLUSTER SIZES: How many records are in each cluster?\n"
        "3. CLUSTER PROFILES: What are the characteristic values of each cluster? "
        "What makes each group distinct?\n"
        "4. REAL-WORLD LABELS: Suggest meaningful names for each cluster based on the data.\n"
        "5. ACTIONABILITY: How could each segment be treated differently?\n"
        "6. LIMITATIONS: What are the limitations of this clustering approach?\n"
        "Write 6–8 sentences. Use actual cluster statistics and column names."
    ),

    "summary": (
        "You are an expert data analyst providing a comprehensive dataset overview.\n"
        "Cover all of the following:\n"
        "1. DATASET OVERVIEW: Number of rows, columns, data types.\n"
        "2. NUMERIC HIGHLIGHTS: Which column has the highest mean? Widest range? Highest std dev?\n"
        "3. DATA QUALITY FLAGS: Columns with suspicious min/max values, very high std, or "
        "large gap between mean and median (possible skew/outliers).\n"
        "4. DISTRIBUTION INSIGHTS: Which columns appear normally distributed vs skewed?\n"
        "5. MODELLING POTENTIAL: Which numeric columns look most useful as features or targets?\n"
        "6. RECOMMENDED NEXT STEPS: What analyses should the user do next?\n"
        "Write 7–10 sentences. Use actual column names and numbers throughout."
    ),

    "comparison": (
        "You are an expert data analyst specialising in comparative and group analysis.\n"
        "Provide a thorough explanation covering:\n"
        "1. WHAT WAS COMPARED: Which groups/categories were compared and on which metric.\n"
        "2. TOP/BOTTOM PERFORMERS: Which group ranks highest and lowest? By how much?\n"
        "3. SPREAD: Is there a large or small difference between groups?\n"
        "4. STATISTICAL NOTE: Are the differences likely meaningful or within normal variation?\n"
        "5. BUSINESS IMPLICATION: What do these differences mean for decision-making?\n"
        "Write 5–7 sentences. Quote actual group names and values."
    ),

    "general": (
        "You are a senior data analyst. Answer the user's question clearly and thoroughly. "
        "Refer specifically to the result data — quote actual column names, numbers, and statistics. "
        "Explain what the result means, why it matters, and what the user should do with this information. "
        "Write at least 4–6 informative sentences."
    ),
}


# ── suggestions ───────────────────────────────────────────────────────────────
def _build_suggestions(query_type: str, result, query: str) -> list[str]:
    base_col = ""
    if isinstance(result, pd.DataFrame) and not result.empty:
        num_cols = result.select_dtypes(include="number").columns.tolist()
        base_col = num_cols[0] if num_cols else ""

    MAP = {
        "explain": [
            "Give me a full statistical summary of the data",
            "Show outliers across all numeric columns",
            "What is the correlation between all numeric columns?",
        ],
        "outlier": [
            f"Show the distribution of {base_col}" if base_col else "Show the distribution",
            "What is the correlation between all numeric columns?",
            "Give me a full statistical summary",
        ],
        "correlation": [
            "Show outliers across all numeric columns",
            "Which column has the most skewed distribution?",
            "Give me a full statistical summary",
        ],
        "trend": [
            "Show outliers in this time series",
            "What is the average value per period?",
            "Give me a full statistical summary",
        ],
        "distribution": [
            f"Show outliers in {base_col}" if base_col else "Show outliers",
            "What is the correlation between all numeric columns?",
            "Give me a full statistical summary",
        ],
        "cluster": [
            "What are the average values per cluster?",
            "What is the correlation between numeric columns?",
            "Show outliers across all numeric columns",
        ],
        "summary": [
            "Show outliers across all numeric columns",
            "What is the correlation between all numeric columns?",
            "Which column has the most skewed distribution?",
        ],
        "comparison": [
            "Show the distribution of values across groups",
            "What is the overall average across all categories?",
            "Which group contains the most outliers?",
        ],
        "general": [
            "Give me a full statistical summary of the data",
            "Show outliers in the numeric columns",
            "What is the correlation between all numeric columns?",
        ],
    }
    return MAP.get(query_type, MAP["general"])[:3]


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def generate_insight(
    question: str,
    result,
    conversation_history: list = None,
) -> tuple[str, list[str]]:
    """
    Generate a detailed, query-aware insight and follow-up suggestions.

    Args:
        question             : The original user question.
        result               : Output from execute_code (DataFrame, scalar, etc.)
        conversation_history : List of previous message dicts {role, content} for context.
                               Used when query is "explain above heatmap" etc.

    Returns:
        (insight_text: str, suggestions: list[str])
    """
    logger.info(f"Generating insight for: '{question[:80]}'")

    qtype       = _query_type(question)
    data_ctx    = _build_data_context(result)
    system_msg  = _SYSTEM_PROMPTS.get(qtype, _SYSTEM_PROMPTS["general"])
    suggestions = _build_suggestions(qtype, result, question)

    # Build messages list
    messages = [{"role": "system", "content": system_msg}]

    # For "explain" queries, inject the last few conversation turns so the LLM
    # knows what "above heatmap" or "the result" refers to
    if qtype == "explain" and conversation_history:
        # Take last 4 assistant messages for context (trim to avoid token overflow)
        relevant = [
            m for m in conversation_history
            if m.get("role") == "assistant" and m.get("insight")
        ][-4:]
        for prev in relevant:
            messages.append({
                "role": "assistant",
                "content": prev.get("insight", "")
            })

    messages.append({
        "role": "user",
        "content": (
            f"User Question: {question}\n\n"
            f"Current Analysis Result:\n{data_ctx}\n\n"
            "Please provide your detailed explanation now."
        )
    })

    try:
        client = _get_client()
        response = client.chat.completions.create(
            messages=messages,
            model=settings.DEFAULT_MODEL,
            temperature=0.4,
            max_tokens=900,          # ← was 400 — now enough for 8-10 sentences
        )
        insight = response.choices[0].message.content.strip()
        insight = re.sub(r"^(insight\s*:\s*)", "", insight, flags=re.IGNORECASE)
        logger.info("Insight generated successfully.")
        return insight, suggestions

    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        return _fallback_insight(result, question), suggestions


def _fallback_insight(result, question: str) -> str:
    if isinstance(result, pd.DataFrame) and not result.empty:
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col  = num_cols[0]
            mean = result[col].mean()
            mn   = result[col].min()
            mx   = result[col].max()
            n    = len(result)
            return (
                f"The analysis returned **{n} rows**. "
                f"Column **{col}** ranges from **{mn:.2f}** to **{mx:.2f}** "
                f"with a mean of **{mean:.2f}**."
            )
        return f"Analysis returned {len(result)} rows: {', '.join(result.columns[:5].tolist())}."
    return f"Analysis completed. Result: {str(result)[:200]}"