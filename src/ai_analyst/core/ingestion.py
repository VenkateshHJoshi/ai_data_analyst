import pandas as pd
import io
import csv
from src.ai_analyst.utils.logger import get_logger
from src.ai_analyst.utils.exceptions import FileProcessingError
from src.ai_analyst.models import DatasetContext, ColumnSchema
from src.ai_analyst.config import settings

logger = get_logger(__name__)

TYPE_MAPPING = {
    'int64': 'integer', 'int32': 'integer',
    'float64': 'float', 'float32': 'float',
    'object': 'string',
    'bool': 'boolean',
    'datetime64[ns]': 'datetime',
}

def ingest_file(file_content: bytes, filename: str) -> DatasetContext:
    logger.info(f"Starting ingestion for file: {filename}")
    
    try:
        # 1. Validate File Size
        size_mb = len(file_content) / (100 * 1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise FileProcessingError(f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit.")

        # 2. Detect Separator (Fixes semicolon issue)
        # We decode a small chunk to sniff the delimiter
        try:
            decoded_chunk = file_content[:1024].decode('utf-8')
            delimiter = csv.Sniffer().sniff(decoded_chunk).delimiter
        except Exception:
            delimiter = ','  # Fallback to comma

        logger.info(f"Detected delimiter: '{delimiter}'")

        # 3. Read CSV with detected delimiter
        df = pd.read_csv(io.BytesIO(file_content), sep=delimiter, on_bad_lines='warn', encoding='utf-8')

        # 4. Basic Validation
        if df.empty:
            raise FileProcessingError("The uploaded file contains no data.")

        # 5. Pre-processing
        df.columns = df.columns.str.strip()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True)
                except (ValueError, TypeError):
                    pass

        # 6. Extract Schema
        columns_schema = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            simplified_type = TYPE_MAPPING.get(dtype_str, 'string')
            columns_schema.append(ColumnSchema(name=col, dtype=simplified_type))

        logger.info(f"Ingestion successful. Shape: {df.shape}")
        
        return DatasetContext(
            dataframe=df,
            columns=columns_schema,
            filename=filename
        )

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise FileProcessingError(f"Failed to parse CSV: {str(e)}")
