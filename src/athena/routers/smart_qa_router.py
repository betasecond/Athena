"""
Router for the Smart QA service endpoints.
"""

import time
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from athena.schemas.common import ApiResponse, PaginatedData
from athena.schemas.smart_qa import (
    UserQueryRequest, QueryResponseData,
    FeedbackRequest, FeedbackResponseData,
    HotQuestionItem, UserHistoryItem
)
from athena.services.rag_service import rag_service
from athena.db.session import get_db

# Create router
router = APIRouter(prefix="/api", tags=["智能问答服务"])


@router.post("/query", response_model=ApiResponse[QueryResponseData])
async def submit_user_query(
    request: UserQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Submit a user query and get a response.
    """
    start_time = time.time()
    
    try:
        # Generate response using RAG service
        rag_result = await rag_service.generate_answer(request.query)
        
        # TODO: Store query and response in database
        # query_id = store_query_in_db(request.query, request.user_id, request.session_id, db)
        # response_id = store_response_in_db(query_id, rag_result, db)
        
        # Prepare response data
        response_data = QueryResponseData(
            originalQuery=request.query,
            suggestedAnswer=rag_result["answer"],
            confidence=rag_result["confidence"],
            needsHumanReview=rag_result["needs_human_review"],
            # For now we'll leave these as None, but in a real implementation, 
            # they would be populated based on additional processing
            keywordAnalysis=None,
            knowledgeMatches=None,
            references=None,
            relatedQuestions=None
        )
        
        return ApiResponse[QueryResponseData](
            data=response_data
        ).set_process_time(start_time)
        
    except Exception as e:
        # Log the error
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=ApiResponse[FeedbackResponseData])
async def submit_user_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback for a previous query and response.
    """
    start_time = time.time()
    
    try:
        # TODO: Store feedback in database
        # store_feedback_in_db(feedback.query_id, feedback.response_id, 
        #                     feedback.feedback_type, feedback.comment, feedback.user_id, db)
        
        # If feedback is negative, potentially add to review queue
        if feedback.feedback_type == "not-helpful":
            # TODO: Add to review queue if negative feedback
            # add_to_review_queue(feedback.query_id, feedback.response_id, 
            #                     "user_feedback_answer_bad", db)
            pass
        
        return ApiResponse[FeedbackResponseData](
            data=FeedbackResponseData(success=True)
        ).set_process_time(start_time)
        
    except Exception as e:
        # Log the error
        print(f"Error storing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-questions", response_model=ApiResponse[PaginatedData[HotQuestionItem]])
async def get_hot_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    sort_by: Optional[str] = Query(None, alias="sortBy"),
    sort_order: Optional[Literal["asc", "desc"]] = Query("desc", alias="sortOrder"),
    db: Session = Depends(get_db)
):
    """
    Get a list of hot/trending questions.
    """
    start_time = time.time()
    
    try:
        # TODO: Fetch hot questions from database
        # This is mock data for now
        mock_items = [
            HotQuestionItem(id=str(i), question=f"热门物流问题 {i}", count=100-i*5) 
            for i in range(1, page_size+1)
        ]
        mock_total = 50
        
        # Calculate pagination values
        total_pages = (mock_total + page_size - 1) // page_size
        has_more = page * page_size < mock_total
        
        # Create paginated response
        paginated_data = PaginatedData[HotQuestionItem](
            items=mock_items,
            total=mock_total,
            page=page,
            pageSize=page_size,
            totalPages=total_pages,
            hasMore=has_more
        )
        
        return ApiResponse[PaginatedData[HotQuestionItem]](
            data=paginated_data
        ).set_process_time(start_time)
        
    except Exception as e:
        # Log the error
        print(f"Error fetching hot questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-history", response_model=ApiResponse[PaginatedData[UserHistoryItem]])
async def get_user_history(
    user_id: str = Query(..., alias="userId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    sort_by: Optional[str] = Query(None, alias="sortBy"),
    sort_order: Optional[Literal["asc", "desc"]] = Query("desc", alias="sortOrder"),
    db: Session = Depends(get_db)
):
    """
    Get the query history for a specific user.
    """
    start_time = time.time()
    
    try:
        # TODO: Fetch user history from database
        # This is mock data for now
        current_time = datetime.now().isoformat()
        mock_items = [
            UserHistoryItem(
                id=str(i),
                query=f"用户 {user_id} 的历史查询 {i}",
                timestamp=current_time
            ) 
            for i in range(1, page_size+1)
        ]
        mock_total = 30
        
        # Calculate pagination values
        total_pages = (mock_total + page_size - 1) // page_size
        has_more = page * page_size < mock_total
        
        # Create paginated response
        paginated_data = PaginatedData[UserHistoryItem](
            items=mock_items,
            total=mock_total,
            page=page,
            pageSize=page_size,
            totalPages=total_pages,
            hasMore=has_more
        )
        
        return ApiResponse[PaginatedData[UserHistoryItem]](
            data=paginated_data
        ).set_process_time(start_time)
        
    except Exception as e:
        # Log the error
        print(f"Error fetching user history: {e}")
        raise HTTPException(status_code=500, detail=str(e))