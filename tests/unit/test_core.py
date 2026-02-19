import pytest
import pandas as pd
import io
import numpy as np
from src.ai_analyst.core.ingestion import ingest_file
from src.ai_analyst.core.execution_engine import execute_code
from src.ai_analyst.core.visualization import generate_plotly_json
from src.ai_analyst.utils.exceptions import FileProcessingError, CodeExecutionError

# --- Tests for Ingestion ---

def test_ingest_valid_csv():
    """Test that a valid CSV is parsed correctly."""
    csv_content = "A,B\n1,2\n3,4"
    context = ingest_file(csv_content.encode('utf-8'), "test.csv")
    assert context.filename == "test.csv"
    assert context.dataframe.shape == (2, 2)
    assert len(context.columns) == 2

def test_ingest_empty_csv():
    """Test that an empty CSV raises an error."""
    csv_content = ""
    with pytest.raises(FileProcessingError):
        ingest_file(csv_content.encode('utf-8'), "empty.csv")

# --- Tests for Execution Engine ---

def test_execute_valid_sum():
    """Test that valid sum code works."""
    df = pd.DataFrame({"Sales": [10, 20, 30]})
    code = "result = df['Sales'].sum()"
    res = execute_code(df, code)
    assert res == 60

def test_execute_invalid_column():
    """Test that referencing a non-existent column raises an error."""
    df = pd.DataFrame({"Sales": [10, 20]})
    code = "result = df['NonExistent'].sum()"
    with pytest.raises(CodeExecutionError):
        execute_code(df, code)

def test_execute_returns_dataframe():
    """Test that code returning a DataFrame is handled."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    code = "result_df = df[df['A'] > 1]"
    res = execute_code(df, code)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1

# --- Tests for Visualization ---

def test_viz_scalar_returns_none():
    """Test that scalar inputs do not generate a chart."""
    res = 500
    assert generate_plotly_json(res) is None

def test_viz_dataframe_generates_json():
    """Test that a DataFrame input generates JSON."""
    df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
    json_str = generate_plotly_json(df)
    assert json_str is not None
    assert isinstance(json_str, str)
    # Check for common Plotly JSON keys
    assert "data" in json_str
    assert "layout" in json_str