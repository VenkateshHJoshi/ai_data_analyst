import sys
import subprocess
import importlib
import pandas as pd
import numpy as np
import builtins
from src.ai_analyst.utils.logger import get_logger
from src.ai_analyst.utils.exceptions import CodeExecutionError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LIBRARY REGISTRY
# Map import names → pip package names for auto-install
# ---------------------------------------------------------------------------
_KNOWN_PACKAGES = {
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "plotly": "plotly",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

def _auto_install(module_name: str) -> bool:
    """Try to pip-install a missing package. Returns True if successful."""
    pip_name = _KNOWN_PACKAGES.get(module_name, module_name)
    logger.info(f"Auto-installing missing package: {pip_name}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to auto-install: {pip_name}")
        return False


def _try_import(module_name: str):
    """
    Import a module safely. Returns the module or None.
    Catches ALL import-related errors (including ImportError from broken
    native extensions like numpy/_multiarray_umath conflicts) so a bad
    environment never crashes the app — it just degrades gracefully.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Package not installed at all — try auto-install
        root = module_name.split(".")[0]
        if _auto_install(root):
            try:
                return importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"Post-install import of '{module_name}' failed: {e}")
    except ImportError as e:
        # Package exists but has broken native extensions (e.g. numpy ABI mismatch)
        logger.warning(
            f"Cannot import '{module_name}' due to environment conflict: {e}. "
            f"Run: pip install numpy scipy scikit-learn --upgrade --force-reinstall"
        )
    except Exception as e:
        logger.warning(f"Unexpected error importing '{module_name}': {e}")
    return None


def _build_safe_globals(df: pd.DataFrame) -> dict:
    """
    Build a rich execution namespace that:
    - Exposes all standard builtins (safe: no file I/O or os/sys access)
    - Pre-loads common data science libraries
    - Provides a safe __import__ that auto-installs missing packages
    """

    # Safe builtins: everything except file/process/network ops
    _BLOCKED = {"open", "eval", "exec", "compile", "__import__", "input",
                "memoryview", "breakpoint"}
    safe_builtins = {
        k: v for k, v in vars(builtins).items() if k not in _BLOCKED
    }

    # Safe __import__ that auto-installs missing packages
    def safe_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in ("os", "sys", "subprocess", "socket", "shutil",
                    "pathlib", "io", "glob"):
            raise ImportError(f"Import of '{name}' is not allowed.")
        try:
            return builtins.__import__(name, *args, **kwargs)
        except ModuleNotFoundError:
            if _auto_install(root):
                return builtins.__import__(name, *args, **kwargs)
            raise

    safe_builtins["__import__"] = safe_import

    # Pre-load common libraries so generated code can use them without importing.
    # Each import is individually guarded — one broken library won't affect others.
    sklearn_mod      = _try_import("sklearn")
    sklearn_ensemble = _try_import("sklearn.ensemble")
    sklearn_linear   = _try_import("sklearn.linear_model")
    sklearn_preproc  = _try_import("sklearn.preprocessing")
    sklearn_metrics  = _try_import("sklearn.metrics")
    sklearn_cluster  = _try_import("sklearn.cluster")
    sklearn_decomp   = _try_import("sklearn.decomposition")
    scipy_mod        = _try_import("scipy")
    scipy_stats      = _try_import("scipy.stats")
    statsmodels_api  = _try_import("statsmodels.api")

    sklearn_available = sklearn_mod is not None
    if not sklearn_available:
        logger.warning(
            "scikit-learn is not importable (likely numpy/scipy ABI mismatch). "
            "Falling back to pure pandas/numpy implementations. "
            "Fix with: pip install numpy scipy scikit-learn --upgrade --force-reinstall"
        )

    # IQR-based outlier detection as a pure numpy/pandas fallback
    # Injected so generated code can call iqr_outliers(df, 'col') even without sklearn
    def iqr_outliers(dataframe: pd.DataFrame, col: str = None) -> pd.DataFrame:
        """Return rows that are outliers by IQR method on numeric columns."""
        num_cols = [col] if col else list(dataframe.select_dtypes(include=[np.number]).columns)
        mask = pd.Series(False, index=dataframe.index)
        for c in num_cols:
            q1, q3 = dataframe[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask |= (dataframe[c] < q1 - 1.5 * iqr) | (dataframe[c] > q3 + 1.5 * iqr)
        return dataframe[mask].copy()

    def zscore_outliers(dataframe: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Return rows where any numeric column has |z-score| > threshold."""
        num_df = dataframe.select_dtypes(include=[np.number]).dropna()
        z = np.abs((num_df - num_df.mean()) / num_df.std())
        return dataframe.loc[num_df[(z > threshold).any(axis=1)].index].copy()

    safe_globals = {
        "__builtins__": safe_builtins,
        # Core data libraries (always available)
        "pd": pd,
        "np": np,
        "df": df.copy(),          # Always work on a copy — never mutate the original
        "scipy": scipy_mod,
        # Pure pandas/numpy fallbacks — always available regardless of sklearn
        "iqr_outliers": iqr_outliers,
        "zscore_outliers": zscore_outliers,
        "SKLEARN_AVAILABLE": sklearn_available,
    }

    # Inject sklearn submodules if available
    if sklearn_mod:
        safe_globals["sklearn"] = sklearn_mod
    if sklearn_ensemble:
        import sklearn.ensemble as _ske
        safe_globals["RandomForestClassifier"] = _ske.RandomForestClassifier
        safe_globals["RandomForestRegressor"]  = _ske.RandomForestRegressor
        safe_globals["GradientBoostingClassifier"] = _ske.GradientBoostingClassifier
        safe_globals["GradientBoostingRegressor"]  = _ske.GradientBoostingRegressor
        safe_globals["IsolationForest"] = _ske.IsolationForest
    if sklearn_preproc:
        import sklearn.preprocessing as _skp
        safe_globals["StandardScaler"]   = _skp.StandardScaler
        safe_globals["MinMaxScaler"]     = _skp.MinMaxScaler
        safe_globals["LabelEncoder"]     = _skp.LabelEncoder
        safe_globals["OneHotEncoder"]    = _skp.OneHotEncoder
    if sklearn_metrics:
        import sklearn.metrics as _skm
        safe_globals["accuracy_score"]   = _skm.accuracy_score
        safe_globals["mean_squared_error"] = _skm.mean_squared_error
        safe_globals["r2_score"]         = _skm.r2_score
        safe_globals["confusion_matrix"] = _skm.confusion_matrix
    if sklearn_cluster:
        import sklearn.cluster as _skc
        safe_globals["KMeans"] = _skc.KMeans
        safe_globals["DBSCAN"] = _skc.DBSCAN
    if sklearn_decomp:
        import sklearn.decomposition as _skd
        safe_globals["PCA"] = _skd.PCA
    if scipy_stats:
        safe_globals["scipy_stats"] = scipy_stats
    if statsmodels_api:
        safe_globals["sm"] = statsmodels_api

    return safe_globals


def _extract_result(local_vars: dict, df: pd.DataFrame):
    """
    Extract the result from local_vars after exec().
    Priority: result_df → result → modified df → raise error.
    """
    if "result_df" in local_vars:
        val = local_vars["result_df"]
        if isinstance(val, pd.DataFrame):
            logger.info("Execution returned result_df (DataFrame).")
            return val
        # LLM sometimes assigns a Series to result_df — convert it
        if isinstance(val, pd.Series):
            logger.info("Execution returned result_df (Series → DataFrame).")
            return val.reset_index()

    if "result" in local_vars:
        val = local_vars["result"]
        logger.info(f"Execution returned result ({type(val).__name__}).")
        # If it's a Series, convert to DataFrame for consistent display
        if isinstance(val, pd.Series):
            return val.reset_index()
        return val

    # Fallback: if df was reassigned in local scope
    if "df" in local_vars and isinstance(local_vars["df"], pd.DataFrame):
        if not local_vars["df"].equals(df):
            logger.warning("No explicit result variable found; returning modified df.")
            return local_vars["df"]

    raise CodeExecutionError(
        "The generated code ran successfully but produced no output. "
        "Make sure to assign output to 'result' (scalar) or 'result_df' (DataFrame)."
    )


def execute_code(df: pd.DataFrame, code: str, max_retries: int = 2):
    """
    Executes generated Python code against a Pandas DataFrame.

    Features:
    - Rich execution namespace (numpy, sklearn, scipy, statsmodels pre-loaded)
    - Safe __import__ with auto-install for missing packages
    - Retry loop: on failure, attempts a cleaned version of the code
    - Consistent result extraction (result / result_df / modified df)

    Args:
        df (pd.DataFrame): The dataset to analyze.
        code (str): Python code string generated by the LLM.
        max_retries (int): Number of retry attempts after initial failure.

    Returns:
        Any: result (scalar, list) or result_df (DataFrame/Series→DataFrame).

    Raises:
        CodeExecutionError: If all execution attempts fail.
    """
    logger.info("Executing generated code...")
    logger.debug(f"Code to execute:\n{code}")

    last_error = None

    for attempt in range(max_retries + 1):
        safe_globals = _build_safe_globals(df)
        local_vars = {}

        try:
            exec(code, safe_globals, local_vars)  # noqa: S102
            return _extract_result(local_vars, df)

        except CodeExecutionError:
            # Re-raise extraction errors immediately (code ran, just bad output)
            raise

        except ModuleNotFoundError as e:
            # A module that couldn't be auto-installed
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            logger.warning(f"Attempt {attempt+1}: Missing module '{missing}'. Trying auto-install...")
            _auto_install(missing)
            last_error = e
            # Retry with same code after install attempt

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt+1} failed: {type(e).__name__}: {e}")

            if attempt < max_retries:
                # Try to self-heal common LLM code mistakes before retrying
                code = _patch_common_errors(code, e)
            else:
                break

    logger.error(f"All execution attempts failed. Last error: {last_error}")
    raise CodeExecutionError(
        f"Code execution failed after {max_retries + 1} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


def _patch_common_errors(code: str, error: Exception) -> str:
    """
    Apply heuristic patches for common LLM code generation mistakes.
    Returns a (possibly patched) version of the code.
    """
    err_str = str(error).lower()
    patched = code

    # Fix: LLM uses 'numeric_only' param on old pandas
    if "numeric_only" in err_str or "unexpected keyword argument 'numeric_only'" in err_str:
        patched = patched.replace(".corr(numeric_only=True)", ".corr()")
        patched = patched.replace(".corr(numeric_only=False)", ".corr()")
        patched = patched.replace(".mean(numeric_only=True)", ".mean()")
        patched = patched.replace(".std(numeric_only=True)", ".std()")
        logger.info("Patched: removed numeric_only keyword arguments.")

    # Fix: LLM uses df.select_dtypes but forgets to assign
    if "select_dtypes" in patched and "result" not in patched:
        patched = patched.replace(
            "df.select_dtypes",
            "result_df = df.select_dtypes"
        )
        logger.info("Patched: assigned select_dtypes result to result_df.")

    # Fix: LLM forgets result variable at end
    if "result" not in patched and "result_df" not in patched:
        # Wrap last line as result
        lines = [l for l in patched.strip().splitlines() if l.strip()]
        if lines:
            last = lines[-1].strip()
            if not last.startswith("#") and "=" not in last:
                patched = "\n".join(lines[:-1]) + f"\nresult = {last}"
                logger.info("Patched: wrapped last expression as result.")

    # Fix: 'object' dtype columns used in numeric ops
    if "could not convert string to float" in err_str or "unsupported operand" in err_str:
        # Force numeric conversion at top of code
        prefix = (
            "# Auto-patch: ensure numeric columns are cast\n"
            "df = df.copy()\n"
            "for _col in df.columns:\n"
            "    try:\n"
            "        df[_col] = pd.to_numeric(df[_col], errors='ignore')\n"
            "    except Exception:\n"
            "        pass\n"
        )
        patched = prefix + patched
        logger.info("Patched: added auto numeric-cast prefix.")

    return patched