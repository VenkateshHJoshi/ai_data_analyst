import re
from groq import Groq
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger
from src.ai_analyst.utils.exceptions import QueryGenerationError
from src.ai_analyst.models import ColumnSchema

logger = get_logger(__name__)


def _get_client() -> Groq:
    """Lazily build the Groq client so API key changes in session state take effect."""
    return Groq(api_key=settings.GROQ_API_KEY)


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return """You are an expert Python Data Analyst. You are given a pandas DataFrame named 'df'.
Your task is to write Python code to answer the user's question.

ALWAYS-AVAILABLE (use these freely, no imports needed):
- pd          → pandas
- np          → numpy
- iqr_outliers(df, 'col')      → returns outlier rows using IQR method (USE THIS for outlier detection)
- zscore_outliers(df, thresh)  → returns outlier rows using Z-score method
- SKLEARN_AVAILABLE            → bool, True if scikit-learn can be imported

OPTIONALLY AVAILABLE (check SKLEARN_AVAILABLE before using, or wrap in try/except):
- StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
- IsolationForest, RandomForestClassifier, RandomForestRegressor
- GradientBoostingClassifier, GradientBoostingRegressor
- KMeans, DBSCAN, PCA
- accuracy_score, mean_squared_error, r2_score, confusion_matrix
- scipy_stats → scipy.stats
- sm          → statsmodels.api

OUTPUT RULES (CRITICAL — follow exactly):
1. Assign a single scalar / string answer to:          result   = ...
2. Assign a table / DataFrame / Series answer to:      result_df = ...
3. NEVER assign to both result and result_df — pick one.
4. DO NOT call print(), display(), or plt.show().
5. Use ONLY columns that exist in the schema provided.
6. Return ONLY raw Python code inside a single ```python ... ``` block. No explanation.

ANALYSIS PATTERNS (prefer these — they always work):

Outlier Detection (preferred — always works):
    result_df = iqr_outliers(df, 'col')
    # Adds IQR bounds as extra columns for context:
    Q1 = df['col'].quantile(0.25)
    Q3 = df['col'].quantile(0.75)
    IQR = Q3 - Q1
    result_df = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)].copy()
    result_df['lower_bound'] = round(Q1 - 1.5*IQR, 2)
    result_df['upper_bound'] = round(Q3 + 1.5*IQR, 2)

Z-Score Outliers (always works):
    result_df = zscore_outliers(df, threshold=3.0)

Outlier Detection with sklearn (only if SKLEARN_AVAILABLE):
    if SKLEARN_AVAILABLE:
        model = IsolationForest(contamination=0.05, random_state=42)
        num_df = df.select_dtypes(include=[np.number]).dropna()
        preds = model.fit_predict(num_df)
        result_df = df.loc[num_df.index[preds == -1]].copy()
    else:
        result_df = iqr_outliers(df)

Correlation Matrix:
    result_df = df.select_dtypes(include=[np.number]).corr().round(3)

Distribution Summary:
    result_df = df.describe().T.round(3)

Group Aggregation:
    result_df = df.groupby('cat_col')['num_col'].agg(['mean','sum','count']).reset_index()

Trend Over Time:
    df['date_col'] = pd.to_datetime(df['date_col'])
    result_df = df.groupby(df['date_col'].dt.to_period('M'))['num_col'].sum().reset_index()
    result_df['date_col'] = result_df['date_col'].astype(str)

Clustering (only if SKLEARN_AVAILABLE):
    if SKLEARN_AVAILABLE:
        num_df = df.select_dtypes(include=[np.number]).dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_df)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        result_df = df.loc[num_df.index].copy()
        result_df['cluster'] = kmeans.fit_predict(scaled)
    else:
        result_df = df.describe().T

PCA (only if SKLEARN_AVAILABLE):
    if SKLEARN_AVAILABLE:
        num_df = df.select_dtypes(include=[np.number]).dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_df)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        result_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
"""


def _build_user_prompt(schema: list[ColumnSchema], query: str) -> str:
    schema_str = "\n".join([f"  - {col.name}: {col.dtype}" for col in schema])
    return (
        f"DataFrame Schema:\n{schema_str}\n\n"
        f"User Question: {query}\n\n"
        f"Generate the Python code:"
    )


def _build_correction_prompt(
    schema: list[ColumnSchema],
    query: str,
    bad_code: str,
    error: str,
) -> list[dict]:
    """Build a correction prompt that sends the failed code + error back to the LLM."""
    schema_str = "\n".join([f"  - {col.name}: {col.dtype}" for col in schema])
    return [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"DataFrame Schema:\n{schema_str}\n\n"
                f"User Question: {query}\n\n"
                f"Your previous code raised an error. Fix it.\n\n"
                f"Previous code:\n```python\n{bad_code}\n```\n\n"
                f"Error:\n{error}\n\n"
                f"Write corrected Python code:"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# CODE EXTRACTION
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """
    Extract Python code from LLM response.
    Handles: ```python ... ```, ``` ... ```, or raw code.
    """
    # Try explicit python block first
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    match = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Discard if it starts with a shell command
        if not code.startswith("$") and not code.startswith("pip"):
            return code

    # Fall back to raw response
    return raw.strip()


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_code(schema: list[ColumnSchema], query: str, max_retries: int = 2) -> str:
    """
    Generate Python analysis code from a natural language query.

    Includes a self-correction retry loop: if the caller reports a CodeExecutionError,
    it passes the error back so we can ask the LLM to fix its own code.

    Args:
        schema: List of ColumnSchema objects describing the DataFrame.
        query: The user's natural language question.
        max_retries: How many times to retry on API error.

    Returns:
        A Python code string ready to be passed to execute_code().

    Raises:
        QueryGenerationError: If the LLM API fails after all retries.
    """
    logger.info(f"Generating code for query: '{query}'")
    client = _get_client()
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_user_prompt(schema, query)},
    ]

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=settings.DEFAULT_MODEL,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            code = _extract_code(raw)
            logger.debug(f"Generated code (attempt {attempt + 1}):\n{code}")
            return code

        except Exception as e:
            last_exc = e
            logger.warning(f"LLM API attempt {attempt + 1} failed: {e}")

    raise QueryGenerationError(f"AI provider error after {max_retries + 1} attempts: {last_exc}")


def regenerate_code_with_error(
    schema: list[ColumnSchema],
    query: str,
    bad_code: str,
    error_message: str,
    max_retries: int = 2,
) -> str:
    """
    Ask the LLM to fix previously generated code that caused a runtime error.
    Called by the app layer when execute_code() raises CodeExecutionError.

    Args:
        schema: Column schema of the DataFrame.
        query: Original user question.
        bad_code: The code that failed.
        error_message: The exception message from execute_code().
        max_retries: API retry count.

    Returns:
        A corrected Python code string.

    Raises:
        QueryGenerationError: If correction also fails.
    """
    logger.info(f"Asking LLM to self-correct code. Error was: {error_message}")
    client = _get_client()
    messages = _build_correction_prompt(schema, query, bad_code, error_message)

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=settings.DEFAULT_MODEL,
                temperature=0.1,   # Lower temperature for correction
                max_tokens=settings.MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            code = _extract_code(raw)
            logger.debug(f"Corrected code (attempt {attempt + 1}):\n{code}")
            return code

        except Exception as e:
            last_exc = e
            logger.warning(f"LLM correction attempt {attempt + 1} failed: {e}")

    raise QueryGenerationError(f"LLM self-correction failed: {last_exc}")