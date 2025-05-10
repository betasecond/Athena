"""
Common schema models used across the Athena API.
"""

import time
import uuid
from typing import Generic, List, Optional, TypeVar, Any

from pydantic import BaseModel, Field, ConfigDict

T = TypeVar('T')
IT = TypeVar('IT')  # ItemType for paginated data


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response format for all endpoints."""
    code: int = 200
    message: str = "成功"
    data: Optional[T] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    process_time: Optional[float] = None  # in milliseconds
    
    model_config = ConfigDict(populate_by_name=True)

    def set_process_time(self, start_time: float) -> "ApiResponse[T]":
        """Calculate and set the API processing time in milliseconds."""
        self.process_time = round((time.time() - start_time) * 1000, 2)
        return self


class PaginatedData(BaseModel, Generic[IT]):
    """Standard pagination data structure."""
    items: List[IT]
    total: int
    page: int
    page_size: int = Field(alias="pageSize")
    total_pages: int = Field(alias="totalPages")
    has_more: bool = Field(alias="hasMore")
    
    model_config = ConfigDict(populate_by_name=True)


class PaginatedResponse(ApiResponse[PaginatedData[IT]]):
    """API response with paginated data."""
    pass