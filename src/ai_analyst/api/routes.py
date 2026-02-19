from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np  # Added import
from typing import Optional

from src.ai_analyst.config import settings
from src.ai_analyst.utils.exceptions import AppException
from src.ai_analyst.utils.logger import get_logger

# Import core logic
from src.ai_analyst.core.ingestion import ingest_file
from src.ai_analyst.core.query_understanding import generate_code
from src.ai_analyst.core.execution_engine import execute_code
from src.ai_analyst.core.insights import generate_insight
from src.ai_analyst.core.visualization import generate_plotly_json
from src.ai_analyst.models import DatasetContext

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs"
)

# --- In-Memory Session Store ---
ACTIVE_CONTEXT: Optional[DatasetContext] = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "AI Data Analyst API is running"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Uploads a CSV file, validates it, and prepares it for analysis.
    """
    global ACTIVE_CONTEXT
    
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Process via Ingestion Layer
        ACTIVE_CONTEXT = ingest_file(content, file.filename)
        
        # Prepare column info for response
        columns_info = [{"name": c.name, "type": c.dtype} for c in ACTIVE_CONTEXT.columns]
        
        return {
            "message": "File uploaded and processed successfully.",
            "filename": ACTIVE_CONTEXT.filename,
            "rows": len(ACTIVE_CONTEXT.dataframe),
            "columns": columns_info
        }
        
    except AppException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def ask_query(payload: dict):
    """
    Accepts a natural language query, executes analysis, and returns insights + charts.
    Expected Payload: {"query": "What is the average sales?"}
    """
    global ACTIVE_CONTEXT
    
    if ACTIVE_CONTEXT is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV file first.")
    
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query field is required.")

    try:
        # 1. Query Understanding (LLM generates Python code)
        code = generate_code(ACTIVE_CONTEXT.columns, query)
        
        # 2. Execution (Run code on DataFrame)
        result = execute_code(ACTIVE_CONTEXT.dataframe, code)
        
        # 3. Insight Generation (LLM summarizes result)
        insight = generate_insight(query, result)
        
        # 4. Visualization (Generate Plotly JSON)
        chart_json = generate_plotly_json(result)
        
        # 5. Prepare Table Data (Serialization Fix)
        table_data = None
        if isinstance(result, pd.DataFrame):
            # Convert to list of dicts
            raw_data = result.head(50).to_dict(orient="records")
            # Explicitly convert numpy types to native python types to avoid FastAPI serialization errors
            table_data = []
            for row in raw_data:
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_row[k] = int(v) if isinstance(v, np.integer) else float(v)
                    elif pd.isna(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                table_data.append(clean_row)
        else:
            # Handle Scalar
            value = result
            if isinstance(result, (np.integer, np.floating)):
                value = int(result) if isinstance(result, np.integer) else float(result)
            table_data = [{"value": value}]

        return {
            "query": query,
            "insight": insight,
            "table": table_data,
            "chart": chart_json
        }

    except AppException as e:
        raise e
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query.")