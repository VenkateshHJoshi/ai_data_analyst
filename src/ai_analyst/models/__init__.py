from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

class ColumnSchema(BaseModel):
    """Represents metadata for a single column."""
    name: str
    dtype: str  # Simplified type: 'integer', 'float', 'string', 'datetime'

class DatasetContext(BaseModel):
    """
    Represents the active dataset in memory.
    Contains the DataFrame and its derived schema.
    """
    dataframe: pd.DataFrame
    columns: List[ColumnSchema]
    filename: str
    
    class Config:
        arbitrary_types_allowed = True # Allow Pandas DataFrame in Pydantic