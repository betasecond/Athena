"""
Schema models for the Smart QA service.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class UserQueryRequest(BaseModel):
    """Request schema for submitting a user query."""
    query: str
    session_id: Optional[str] = Field(None, alias="sessionId")
    user_id: Optional[str] = Field(None, alias="userId")
    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(populate_by_name=True)


class KeywordAnalysisItem(BaseModel):
    """Schema for keyword analysis item."""
    keyword: str
    type: str
    confidence: float


class KnowledgeMatchItem(BaseModel):
    """Schema for knowledge match item."""
    id: str
    standard_question: str = Field(..., alias="standardQuestion")
    relevance_score: float = Field(..., alias="relevanceScore")
    last_updated: str = Field(..., alias="lastUpdated")  # ISO 8601 datetime string
    tags: Optional[List[str]] = None
    
    model_config = ConfigDict(populate_by_name=True)


class QueryResponseData(BaseModel):
    """Response schema for query response data."""
    original_query: str = Field(..., alias="originalQuery")
    keyword_analysis: Optional[List[KeywordAnalysisItem]] = Field(None, alias="keywordAnalysis")
    knowledge_matches: Optional[List[KnowledgeMatchItem]] = Field(None, alias="knowledgeMatches")
    suggested_answer: str = Field(..., alias="suggestedAnswer")
    confidence: float
    needs_human_review: bool = Field(..., alias="needsHumanReview")
    references: Optional[List[str]] = None
    related_questions: Optional[List[str]] = Field(None, alias="relatedQuestions")
    
    model_config = ConfigDict(populate_by_name=True)


class FeedbackRequest(BaseModel):
    """Request schema for submitting user feedback."""
    query_id: str = Field(..., alias="queryId")
    response_id: str = Field(..., alias="responseId")
    feedback_type: Literal["helpful", "not-helpful", "partially-helpful"] = Field(..., alias="feedbackType")
    comment: Optional[str] = None
    user_id: Optional[str] = Field(None, alias="userId")
    
    model_config = ConfigDict(populate_by_name=True)


class FeedbackResponseData(BaseModel):
    """Response schema for feedback submission."""
    success: bool


class HotQuestionItem(BaseModel):
    """Schema for hot question item."""
    id: str
    question: str
    count: int


class UserHistoryItem(BaseModel):
    """Schema for user history item."""
    id: str
    query: str
    timestamp: str  # ISO 8601 datetime string