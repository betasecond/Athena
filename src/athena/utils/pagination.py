"""
Utility functions for handling pagination in API responses.
"""

from typing import TypeVar, List, Type, Generic, Any, Dict, Optional
from math import ceil

from athena.schemas.common import PaginatedData

T = TypeVar('T')


def create_paginated_data(
    items: List[T], 
    total: int, 
    page: int, 
    page_size: int,
    data_class: Type[PaginatedData[T]] = PaginatedData
) -> PaginatedData[T]:
    """
    Create a paginated data structure.
    
    Args:
        items: List of items for the current page
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        page_size: Number of items per page
        data_class: The specific PaginatedData class to instantiate
        
    Returns:
        A PaginatedData instance
    """
    total_pages = ceil(total / page_size) if total > 0 else 1
    has_more = page * page_size < total
    
    return data_class(
        items=items,
        total=total,
        page=page,
        pageSize=page_size,
        totalPages=total_pages,
        hasMore=has_more
    )


def apply_pagination(
    query_data: List[Dict[str, Any]], 
    page: int, 
    page_size: int
) -> List[Dict[str, Any]]:
    """
    Apply pagination to a list of data.
    
    This is useful for in-memory pagination when not using a database query.
    
    Args:
        query_data: List of data to paginate
        page: Current page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        A slice of the original data for the requested page
    """
    start = (page - 1) * page_size
    end = start + page_size
    
    return query_data[start:end]


def get_pagination_params(
    query_params: Dict[str, Any], 
    default_page: int = 1, 
    default_page_size: int = 10, 
    max_page_size: int = 100
) -> Dict[str, int]:
    """
    Extract and validate pagination parameters from request query params.
    
    Args:
        query_params: Dictionary of query parameters
        default_page: Default page if not specified
        default_page_size: Default page size if not specified
        max_page_size: Maximum allowed page size
        
    Returns:
        Dictionary with validated 'page' and 'page_size'
    """
    page = query_params.get('page', default_page)
    page_size = query_params.get('pageSize', default_page_size)
    
    # Validate values
    page = max(1, int(page))  # Page should be at least 1
    page_size = min(max_page_size, max(1, int(page_size)))  # Limit page size
    
    return {
        'page': page,
        'page_size': page_size
    }