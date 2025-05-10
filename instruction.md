好的，这份前端 API 文档非常清晰和全面！基于这份文档，我们可以为你的 Python 后端服务（代号 "Athena"）设计相应的接口和数据结构。

我将主要使用 **FastAPI** 作为 Web 框架（因为它与 Pydantic 天然集成，非常适合构建 API），并使用 **Pydantic** 来定义数据结构（请求体、响应体、数据库模型等）。

**核心原则：**

1.  **Pydantic 模型对应数据结构：** API 文档中定义的每个 `object` 或复杂类型都将对应一个 Pydantic 模型。
2.  **FastAPI 路由对应接口：** 每个 API 接口将对应一个 FastAPI 路径操作函数 (Path Operation Function)。
3.  **类型提示：** Python 代码将大量使用类型提示，以确保数据一致性和编辑器支持。
4.  **模块化：** 建议将不同服务（智能问答、客服辅助、知识库审核）的路由和逻辑分散到不同的 Python 模块中。

下面是针对每个服务的设计建议：

## 1. 通用规范 (Python 实现思路)

你的 API 文档中已经定义了非常好的通用响应格式。在 FastAPI 中，我们可以创建一个通用的响应模型，并在每个路由函数中返回它。

```python
# src/athena/schemas/common.py (或类似路径)
from typing import TypeVar, Generic, List, Optional, Any
from pydantic import BaseModel, Field
import time
import uuid

T = TypeVar('T')
IT = TypeVar('IT') # ItemType for paginated data

class ApiResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "成功"
    data: Optional[T] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    process_time: Optional[float] = None # in milliseconds

    def set_process_time(self, start_time: float):
        self.process_time = round((time.time() - start_time) * 1000, 2)
        return self

class PaginatedData(BaseModel, Generic[IT]):
    items: List[IT]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_more: bool

class PaginatedResponse(ApiResponse[PaginatedData[IT]]):
    pass

# 可以在 FastAPI 中使用中间件或依赖注入来自动添加 requestId 和 processTime
# 例如，一个简单的中间件:
# from fastapi import Request, Response
#
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response: Response = await call_next(request)
#     process_time_ms = round((time.time() - start_time) * 1000, 2)
#     # 通常我们会将 requestId 和 processTime 放入响应体，而不是header
#     # 但如果放入header，可以这样:
#     # response.headers["X-Process-Time-Ms"] = str(process_time_ms)
#     # response.headers["X-Request-Id"] = str(uuid.uuid4()) # request_id 更适合在响应体
#     return response

# 错误处理: FastAPI 允许你定义自定义异常处理器，将自定义异常转换为标准错误响应格式。
# from fastapi import FastAPI, Request, status
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError
#
# class APIException(Exception):
#     def __init__(self, code: int, message: str, data: Any = None):
#         self.code = code
#         self.message = message
#         self.data = data
#
# async def api_exception_handler(request: Request, exc: APIException):
#     return JSONResponse(
#         status_code=200, # API 文档中错误响应的 HTTP 状态码通常仍是 200，业务码在 code 字段体现
#         content=ApiResponse(
#             code=exc.code,
#             message=exc.message,
#             data=exc.data,
#             request_id=str(uuid.uuid4()) # 也可以从 request 中获取或生成
#         ).model_dump(exclude_none=True)
#     )
#
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     # Pydantic 验证错误
#     return JSONResponse(
#         status_code=200,
#         content=ApiResponse(
#             code=400,
#             message="请求参数错误",
#             data={"details": exc.errors()} # Pydantic 错误详情
#         ).model_dump(exclude_none=True)
#     )
#
# # 在 FastAPI app 初始化时注册:
# # app = FastAPI()
# # app.add_exception_handler(APIException, api_exception_handler)
# # app.add_exception_handler(RequestValidationError, validation_exception_handler)

```

## 2. 智能问答服务 (Smart QA Service)

我们将为请求体和响应体中的 `data` 部分定义 Pydantic 模型。

```python
# src/athena/schemas/smart_qa.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from .common import PaginatedData # 假设 common.py 在同一目录下或可导入

# --- 提交用户问题 (/api/query) ---
class UserQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(None, alias="sessionId")
    user_id: Optional[str] = Field(None, alias="userId")
    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class KeywordAnalysisItem(BaseModel):
    keyword: str
    type: str
    confidence: float

class KnowledgeMatchItem(BaseModel):
    id: str
    standard_question: str = Field(..., alias="standardQuestion")
    relevance_score: float = Field(..., alias="relevanceScore")
    last_updated: str = Field(..., alias="lastUpdated") # ISO 8601 datetime string
    tags: Optional[List[str]] = None

class QueryResponseData(BaseModel):
    original_query: str = Field(..., alias="originalQuery")
    keyword_analysis: Optional[List[KeywordAnalysisItem]] = Field(None, alias="keywordAnalysis")
    knowledge_matches: Optional[List[KnowledgeMatchItem]] = Field(None, alias="knowledgeMatches")
    suggested_answer: str = Field(..., alias="suggestedAnswer")
    confidence: float
    needs_human_review: bool = Field(..., alias="needsHumanReview")
    references: Optional[List[str]] = None
    related_questions: Optional[List[str]] = Field(None, alias="relatedQuestions")

# --- 提交用户反馈 (/api/feedback) ---
class FeedbackRequest(BaseModel):
    query_id: str = Field(..., alias="queryId")
    response_id: str = Field(..., alias="responseId")
    feedback_type: Literal["helpful", "not-helpful", "partially-helpful"] = Field(..., alias="feedbackType")
    comment: Optional[str] = None
    user_id: Optional[str] = Field(None, alias="userId")

class FeedbackResponseData(BaseModel):
    success: bool

# --- 获取热门问题 (/api/hot-questions) ---
class HotQuestionItem(BaseModel):
    id: str
    question: str
    count: int

class HotQuestionsResponseData(PaginatedData[HotQuestionItem]):
    pass

# --- 获取用户历史查询 (/api/user-history) ---
class UserHistoryItem(BaseModel):
    id: str
    query: str
    timestamp: str # ISO 8601 datetime string

class UserHistoryResponseData(PaginatedData[UserHistoryItem]):
    pass

```

**FastAPI 路由示例 (智能问答服务):**

```python
# src/athena/routers/smart_qa_router.py
from fastapi import APIRouter, Query, Body, Depends
from typing import Optional
from ..schemas.common import ApiResponse # 从你的 common schema 导入
from ..schemas.smart_qa import (
    UserQueryRequest, QueryResponseData,
    FeedbackRequest, FeedbackResponseData,
    HotQuestionsResponseData, HotQuestionItem, # 确保 HotQuestionsResponseData 包含 PaginatedData 结构
    UserHistoryResponseData, UserHistoryItem   # 同上
)
# 假设你有一个函数来包装分页数据
from ..utils.pagination import create_paginated_response # 你需要自己实现这个工具函数

router = APIRouter(prefix="/api", tags=["智能问答服务"])

@router.post("/query", response_model=ApiResponse[QueryResponseData])
async def submit_user_query(request_body: UserQueryRequest):
    # 你的 RAG 和业务逻辑在这里
    # ...
    # 示例响应数据
    response_data = QueryResponseData(
        originalQuery=request_body.query,
        # ... 其他字段
        suggestedAnswer="这是基于RAG生成的答案。",
        confidence=0.85,
        needsHumanReview=False
    )
    return ApiResponse[QueryResponseData](data=response_data)

@router.post("/feedback", response_model=ApiResponse[FeedbackResponseData])
async def submit_user_feedback(request_body: FeedbackRequest):
    # 存储反馈的逻辑
    # ...
    return ApiResponse[FeedbackResponseData](data=FeedbackResponseData(success=True))

@router.get("/hot-questions", response_model=ApiResponse[HotQuestionsResponseData])
async def get_hot_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    sort_by: Optional[str] = Query(None, alias="sortBy"),
    sort_order: Optional[Literal["asc", "desc"]] = Query("desc", alias="sortOrder")
):
    # 获取热门问题逻辑 (通常从数据库查询和聚合)
    # items_from_db = [...] # 假设这是从数据库获取的 HotQuestionItem 列表
    # total_count_from_db = 0 # 假设这是总数
    # return create_paginated_response(items_from_db, total_count_from_db, page, page_size, HotQuestionsResponseData)
    # 伪代码示例
    mock_items = [HotQuestionItem(id=str(i), question=f"热门问题 {i}", count=100-i*5) for i in range(page_size)]
    mock_total = 50
    return ApiResponse[HotQuestionsResponseData](
        data=HotQuestionsResponseData(
            items=mock_items,
            total=mock_total,
            page=page,
            pageSize=page_size,
            totalPages=(mock_total + page_size - 1) // page_size,
            hasMore=page * page_size < mock_total
        )
    )


@router.get("/user-history", response_model=ApiResponse[UserHistoryResponseData])
async def get_user_history(
    user_id: str = Query(..., alias="userId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    sort_by: Optional[str] = Query(None, alias="sortBy"),
    sort_order: Optional[Literal["asc", "desc"]] = Query("desc", alias="sortOrder")
):
    # 获取用户历史逻辑
    # items_from_db = [...] # UserHistoryItem 列表
    # total_count_from_db = 0
    # return create_paginated_response(items_from_db, total_count_from_db, page, page_size, UserHistoryResponseData)
    # 伪代码示例
    mock_items = [UserHistoryItem(id=str(i), query=f"用户 {user_id} 的历史查询 {i}", timestamp="2024-05-10T10:00:00Z") for i in range(page_size)]
    mock_total = 30
    return ApiResponse[UserHistoryResponseData](
        data=UserHistoryResponseData(
            items=mock_items,
            total=mock_total,
            page=page,
            pageSize=page_size,
            totalPages=(mock_total + page_size - 1) // page_size,
            hasMore=page * page_size < mock_total
        )
    )

```

## 3. 客服辅助服务 (Agent Assist Service)

这部分的数据结构相对复杂，特别是 `context` 对象。

```python
# src/athena/schemas/agent_assist.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# --- 获取实时辅助建议 (/api/agent-assist/suggestions) ---
class Sentiment(BaseModel):
    type: Literal["positive", "negative", "neutral"]
    score: float = Field(..., ge=0, le=1)

class MessageItem(BaseModel):
    id: str
    sender: Literal["customer", "agent", "system"]
    content: str
    timestamp: str # ISO 8601 datetime string
    type: Literal["text", "image", "file", "system-notice"] = "text"
    status: Optional[Literal["sending", "sent", "read", "failed"]] = None
    sentiment: Optional[Sentiment] = None

class CustomerInfo(BaseModel):
    name: Optional[str] = None
    vip_level: Optional[int] = Field(None, alias="vipLevel")
    order_count: Optional[int] = Field(None, alias="orderCount")
    tags: Optional[List[str]] = None

class OrderInfo(BaseModel):
    order_id: Optional[str] = Field(None, alias="orderId")
    status: Optional[str] = None
    amount: Optional[float] = None
    order_date: Optional[str] = Field(None, alias="orderDate") # ISO 8601 datetime string

class ConversationContext(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    customer_id: Optional[str] = Field(None, alias="customerId")
    agent_id: str = Field(..., alias="agentId")
    history: List[MessageItem]
    customer_info: Optional[CustomerInfo] = Field(None, alias="customerInfo")
    order_info: Optional[OrderInfo] = Field(None, alias="orderInfo")
    topics: Optional[List[str]] = None

class AgentAssistSuggestionsRequest(BaseModel):
    context: ConversationContext
    current_draft: str = Field(..., alias="currentDraft")
    cursor_position: Optional[int] = Field(None, alias="cursorPosition")
    assist_type: Optional[Literal["auto", "completion", "sentiment", "compliance"]] = Field(None, alias="assistType")

class SuggestionItem(BaseModel):
    id: str
    type: Literal["completion", "sentiment", "compliance", "information", "greeting", "closing"]
    content: str
    priority: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0, le=1)
    reason: Optional[str] = None
    source_id: Optional[str] = Field(None, alias="sourceId")

class SentimentAnalysisResult(Sentiment): # Sentiment 已经定义了 type 和 score
    keywords: Optional[List[str]] = None

class ComplianceIssueItem(BaseModel):
    type: str
    description: str
    severity: Literal["low", "medium", "high"]
    suggestion: Optional[str] = None

class ComplianceAnalysisResult(BaseModel):
    has_issues: bool = Field(..., alias="hasIssues")
    issues: Optional[List[ComplianceIssueItem]] = None

class InformationCompletenessResult(BaseModel):
    missing_info: Optional[List[str]] = Field(None, alias="missingInfo")
    suggested_questions: Optional[List[str]] = Field(None, alias="suggestedQuestions")

class AgentAssistSuggestionsResponseData(BaseModel):
    suggestions: List[SuggestionItem]
    sentiment_analysis: Optional[SentimentAnalysisResult] = Field(None, alias="sentimentAnalysis")
    compliance_analysis: Optional[ComplianceAnalysisResult] = Field(None, alias="complianceAnalysis")
    information_completeness: Optional[InformationCompletenessResult] = Field(None, alias="informationCompleteness")

# --- 搜索知识库 (/api/knowledge/search) ---
class KnowledgeSearchResultItem(BaseModel):
    id: str
    title: str
    content: str
    last_updated: str = Field(..., alias="lastUpdated") # ISO 8601 datetime string
    tags: Optional[List[str]] = None
    relevance_score: float = Field(..., alias="relevanceScore")

# Response data for /api/knowledge/search is List[KnowledgeSearchResultItem]

# --- 记录建议使用情况 (/api/agent-assist/track-usage) ---
class TrackUsageRequest(BaseModel):
    suggestion_id: str = Field(..., alias="suggestionId")
    session_id: str = Field(..., alias="sessionId")

class TrackUsageResponseData(BaseModel):
    success: bool

# --- 获取回复模板 (/api/agent-assist/templates) ---
class TemplateItem(BaseModel):
    id: str
    title: str
    content: str
    category: Optional[str] = None

# Response data for /api/agent-assist/templates is List[TemplateItem]
```

**FastAPI 路由示例 (客服辅助服务):**

```python
# src/athena/routers/agent_assist_router.py
from fastapi import APIRouter, Query, Body
from typing import Optional, List
from ..schemas.common import ApiResponse
from ..schemas.agent_assist import (
    AgentAssistSuggestionsRequest, AgentAssistSuggestionsResponseData,
    KnowledgeSearchResultItem,
    TrackUsageRequest, TrackUsageResponseData,
    TemplateItem
)

router = APIRouter(prefix="/api/agent-assist", tags=["客服辅助服务"])

@router.post("/suggestions", response_model=ApiResponse[AgentAssistSuggestionsResponseData])
async def get_agent_assist_suggestions(request_body: AgentAssistSuggestionsRequest):
    # 复杂的逻辑：处理上下文，调用 RAG，情感/合规分析等
    # ...
    # 示例响应
    response_data = AgentAssistSuggestionsResponseData(
        suggestions=[
            SuggestionItem(id="sugg1", type="completion", content="您可以尝试这样回复...", priority=1, confidence=0.9)
        ]
        # ... 其他字段
    )
    return ApiResponse[AgentAssistSuggestionsResponseData](data=response_data)

@router.get("/knowledge/search", response_model=ApiResponse[List[KnowledgeSearchResultItem]])
async def search_knowledge_base(
    query: str,
    limit: Optional[int] = Query(10, ge=1, le=50),
    tags: Optional[List[str]] = Query(None) # FastAPI 会自动处理查询参数中的多值 (e.g., ?tags=foo&tags=bar)
):
    # RAG 搜索逻辑
    # ...
    # 示例响应
    results = [
        KnowledgeSearchResultItem(id="kb1", title="关于退货", content="退货政策详情...", lastUpdated="2024-01-01T00:00:00Z", relevanceScore=0.95)
    ]
    return ApiResponse[List[KnowledgeSearchResultItem]](data=results)

@router.post("/track-usage", response_model=ApiResponse[TrackUsageResponseData])
async def track_suggestion_usage(request_body: TrackUsageRequest):
    # 记录建议使用情况的逻辑
    # ...
    return ApiResponse[TrackUsageResponseData](data=TrackUsageResponseData(success=True))

@router.get("/templates", response_model=ApiResponse[List[TemplateItem]])
async def get_reply_templates(
    agent_id: str = Query(..., alias="agentId"),
    category: Optional[str] = Query(None)
):
    # 获取模板逻辑
    # ...
    # 示例响应
    templates = [
        TemplateItem(id="tpl1", title="标准问候语", content="您好，很高兴为您服务！", category="问候")
    ]
    return ApiResponse[List[TemplateItem]](data=templates)

```

## 4. 知识库审核队列服务 (Knowledge Base Review Queue Service)

这部分与你之前讨论的审核页面 UI 紧密相关。

```python
# src/athena/schemas/review_queue.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from .common import PaginatedData # 确保可以导入

# --- 获取审核队列列表 (/api/review-queue/items) ---
class ReviewMetadata(BaseModel):
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    expiration_date: Optional[str] = Field(None, alias="expirationDate") # ISO 8601 datetime string

class ReviewQueueItem(BaseModel):
    id: str
    source: str
    original_query: Optional[str] = Field(None, alias="originalQuery")
    current_answer: Optional[str] = Field(None, alias="currentAnswer")
    suggested_answer: Optional[str] = Field(None, alias="suggestedAnswer")
    timestamp: str # ISO 8601 datetime string
    status: Literal["pending", "approved", "rejected", "needsInfo"]
    standard_question: Optional[str] = Field(None, alias="standardQuestion")
    metadata: Optional[ReviewMetadata] = None

class ReviewQueueListResponseData(PaginatedData[ReviewQueueItem]):
    pass

# --- 获取审核项详情 (/api/review-queue/items/{itemId}) ---
class RelatedKnowledgeItem(BaseModel):
    id: str
    question: str
    similarity_score: float = Field(..., alias="similarityScore")

class ReviewHistoryLog(BaseModel):
    reviewer_id: str = Field(..., alias="reviewerId")
    reviewer_name: str = Field(..., alias="reviewerName")
    timestamp: str # ISO 8601 datetime string
    action: str # e.g., "approved", "rejected", "commented", "edited"
    comment: Optional[str] = None

class ReviewQueueItemDetail(ReviewQueueItem): # 继承自 ReviewQueueItem
    related_knowledge_items: Optional[List[RelatedKnowledgeItem]] = Field(None, alias="relatedKnowledgeItems")
    review_history: Optional[List[ReviewHistoryLog]] = Field(None, alias="reviewHistory")

# --- 提交审核决定 (/api/review-queue/decision) ---
class ReviewDecisionRequest(BaseModel):
    item_id: str = Field(..., alias="itemId")
    decision: Literal["approve", "reject", "needsInfo"]
    standard_question: Optional[str] = Field(None, alias="standardQuestion") # 批准时必填
    suggested_answer: Optional[str] = Field(None, alias="suggestedAnswer")   # 批准时必填
    metadata: Optional[ReviewMetadata] = None
    comment: Optional[str] = None

class ReviewDecisionResponseData(BaseModel):
    success: bool
    knowledge_item_id: Optional[str] = Field(None, alias="knowledgeItemId") # 如果批准，返回创建的知识条目ID

# --- 批量操作审核项 (/api/review-queue/batch-operation) ---
class BatchOperationRequest(BaseModel):
    operation: Literal["approve", "reject", "markNeedsInfo"]
    item_ids: List[str] = Field(..., alias="itemIds")
    comment: Optional[str] = None

class BatchOperationResponseData(BaseModel):
    success: bool
    processed_count: int = Field(..., alias="processedCount")

# --- 获取审核标签列表 (/api/review-queue/tags) ---
# Response data is List[str]

# --- 获取审核来源类型 (/api/review-queue/sources) ---
# Response data is List[str]

# --- 获取审核统计数据 (/api/review-queue/stats) ---
class ReviewStatsBySource(BaseModel):
    user_query_system_no_answer: Optional[int] = Field(0, alias="用户提问-系统未答出")
    user_feedback_answer_bad: Optional[int] = Field(0, alias="用户反馈-答案差评")
    low_confidence_answer: Optional[int] = Field(0, alias="低置信度回答")
    # 支持动态添加其他来源，或者使用 Dict[str, int]
    # For dynamic keys, Pydantic offers RootModel or using __root__ with Dict
    # Alternatively, define all known keys as optional as above.

class ReviewStatsData(BaseModel):
    total_items: int = Field(..., alias="totalItems")
    pending_count: int = Field(..., alias="pendingCount")
    approved_count: int = Field(..., alias="approvedCount")
    rejected_count: int = Field(..., alias="rejectedCount")
    needs_info_count: int = Field(..., alias="needsInfoCount")
    by_source: Optional[Dict[str, int]] = Field(None, alias="bySource") # 更灵活的方式处理动态来源
                                                                    # 或者使用上面定义的 ReviewStatsBySource

```

**FastAPI 路由示例 (知识库审核队列服务):**

```python
# src/athena/routers/review_queue_router.py
from fastapi import APIRouter, Query, Body, Path
from typing import Optional, List, Literal
from ..schemas.common import ApiResponse
from ..schemas.review_queue import (
    ReviewQueueListResponseData, ReviewQueueItem, # 确保 ReviewQueueListResponseData 包含 PaginatedData 结构
    ReviewQueueItemDetail,
    ReviewDecisionRequest, ReviewDecisionResponseData,
    BatchOperationRequest, BatchOperationResponseData,
    ReviewStatsData
)
# 假设你有一个函数来包装分页数据
from ..utils.pagination import create_paginated_response # 你需要自己实现这个工具函数


router = APIRouter(prefix="/api/review-queue", tags=["知识库审核队列服务"])

@router.get("/items", response_model=ApiResponse[ReviewQueueListResponseData])
async def get_review_queue_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    sort_by: Optional[str] = Query(None, alias="sortBy"),
    sort_order: Optional[Literal["asc", "desc"]] = Query("desc", alias="sortOrder"),
    keyword: Optional[str] = Query(None),
    status: Optional[Literal["pending", "approved", "rejected", "needsInfo"]] = Query(None),
    source: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None), # e.g., ?tags=foo&tags=bar
    start_date: Optional[str] = Query(None, alias="startDate"), # Consider validating date format
    end_date: Optional[str] = Query(None, alias="endDate")
):
    # 获取审核队列的逻辑，带筛选和分页
    # items_from_db = [...] # ReviewQueueItem 列表
    # total_count_from_db = 0
    # return create_paginated_response(items_from_db, total_count_from_db, page, page_size, ReviewQueueListResponseData)
    # 伪代码示例
    mock_items = [ReviewQueueItem(id=str(i), source="用户提问-系统未答出", originalQuery=f"问题 {i}?", timestamp="2024-05-10T12:00:00Z", status="pending") for i in range(page_size)]
    mock_total = 25
    return ApiResponse[ReviewQueueListResponseData](
        data=ReviewQueueListResponseData(
            items=mock_items,
            total=mock_total,
            page=page,
            pageSize=page_size,
            totalPages=(mock_total + page_size - 1) // page_size,
            hasMore=page * page_size < mock_total
        )
    )


@router.get("/items/{item_id}", response_model=ApiResponse[ReviewQueueItemDetail])
async def get_review_item_detail(item_id: str = Path(..., alias="itemId")): # FastAPI 建议使用 itemId 而不是 item_id
    # 获取单个审核项详情逻辑
    # ...
    # 示例响应
    item_detail = ReviewQueueItemDetail(
        id=item_id,
        source="用户提问-系统未答出",
        originalQuery="这是一个详细的问题",
        timestamp="2024-05-10T12:00:00Z",
        status="pending",
        # ... 其他字段
    )
    return ApiResponse[ReviewQueueItemDetail](data=item_detail)

@router.post("/decision", response_model=ApiResponse[ReviewDecisionResponseData])
async def submit_review_decision(request_body: ReviewDecisionRequest):
    # 处理审核决定的逻辑 (更新数据库，如果批准则触发知识入库到向量数据库)
    # ...
    if request_body.decision == "approve":
        # 检查 standardQuestion 和 suggestedAnswer 是否提供
        if not request_body.standard_question or not request_body.suggested_answer:
            # 你应该通过APIException抛出错误，会被全局异常处理器捕获
            # raise APIException(code=400, message="批准时，标准问题和标准答案为必填项。")
            return ApiResponse(code=400, message="批准时，标准问题和标准答案为必填项。") # 简易错误处理

        # TODO: 将 standardQuestion 和 suggestedAnswer 添加到向量数据库的逻辑
        # knowledge_item_id = "new_kb_id_123"
        knowledge_item_id = "new_kb_id_123" if request_body.decision == "approve" else None
        return ApiResponse[ReviewDecisionResponseData](data=ReviewDecisionResponseData(success=True, knowledgeItemId=knowledge_item_id))
    
    return ApiResponse[ReviewDecisionResponseData](data=ReviewDecisionResponseData(success=True))


@router.post("/batch-operation", response_model=ApiResponse[BatchOperationResponseData])
async def batch_operate_review_items(request_body: BatchOperationRequest):
    # 批量操作逻辑
    # ...
    processed_count = len(request_body.item_ids) # 伪代码
    return ApiResponse[BatchOperationResponseData](data=BatchOperationResponseData(success=True, processedCount=processed_count))

@router.get("/tags", response_model=ApiResponse[List[str]])
async def get_review_tags():
    # 从数据库或配置中获取标签列表
    tags = ["会员", "积分规则", "支付问题", "退款", "产品"]
    return ApiResponse[List[str]](data=tags)

@router.get("/sources", response_model=ApiResponse[List[str]])
async def get_review_sources():
    # 从数据库或配置中获取来源类型列表
    sources = ["用户提问-系统未答出", "用户反馈-答案差评", "低置信度回答"]
    return ApiResponse[List[str]](data=sources)

@router.get("/stats", response_model=ApiResponse[ReviewStatsData])
async def get_review_stats():
    # 获取统计数据逻辑
    # ...
    stats = ReviewStatsData(
        totalItems=100,
        pendingCount=20,
        approvedCount=70,
        rejectedCount=5,
        needsInfoCount=5,
        bySource={
            "用户提问-系统未答出": 50,
            "用户反馈-答案差评": 30,
            "低置信度回答": 20
        }
    )
    return ApiResponse[ReviewStatsData](data=stats)

```

**数据库表结构/文档结构 (初步设想):**

* **UserQueries:** `id`, `user_id`, `session_id`, `query_text`, `source`, `context_json`, `timestamp`, `llm_response_id`
* **LLMResponses:** `id`, `query_id` (FK to UserQueries), `suggested_answer`, `confidence`, `needs_review`, `keyword_analysis_json`, `knowledge_matches_json`, `references_json`, `related_questions_json`, `timestamp`
* **Feedbacks:** `id`, `query_id`, `response_id`, `user_id`, `feedback_type`, `comment`, `timestamp`
* **HotQuestions:** `id`, `question_text` (or FK to a StandardQuestion table), `query_count`, `last_queried_timestamp`
* **KnowledgeItems:** `id`, `standard_question`, `answer_text`, `tags_json`, `source`, `status` (e.g., active, deprecated), `created_at`, `last_updated_at`, `metadata_json` (用于存储如有效期等信息)
* **ReviewQueueItems:** `id`, `source_type` (e.g., 'user_query', 'feedback', 'low_confidence'), `source_data_id` (e.g., UserQuery.id or Feedback.id), `original_query_text`, `current_system_answer_text`, `status` ('pending', 'approved', 'rejected', 'needsInfo'), `assigned_reviewer_id`, `created_at`, `updated_at`, `submitted_standard_question`, `submitted_answer`, `review_notes`, `metadata_json` (tags, keywords from review)
* **ReviewHistory:** `id`, `review_item_id` (FK), `reviewer_id`, `action`, `comment`, `timestamp`, `changes_json` (记录修改详情)
* **AgentAssistSessions:** `id`, `agent_id`, `customer_id`, `session_start_time`, `session_end_time`
* **ConversationHistory:** `id`, `session_id` (FK to AgentAssistSessions), `message_id_external`, `sender_type`, `content`, `message_type`, `status`, `sentiment_json`, `timestamp`
* **AgentAssistSuggestionsLog:** `id`, `session_id`, `suggestion_id_provided`, `suggestion_type`, `suggestion_content`, `is_used_by_agent`, `timestamp`
* **ReplyTemplates:** `id`, `title`, `content`, `category`, `created_by_agent_id`, `is_shared`

**项目结构建议:**

```
athena/
├── src/
│   └── athena/
│       ├── __init__.py
│       ├── main.py             # FastAPI app instance and main entry point
│       ├── core/               # Core logic, settings, security
│       │   ├── config.py
│       │   └── security.py
│       ├── db/                 # Database connection, session management, models (if using ORM)
│       │   ├── base.py
│       │   ├── session.py
│       │   └── models/         # SQLAlchemy models or Pymongo ODM models
│       ├── schemas/            # Pydantic schemas (request/response models)
│       │   ├── common.py
│       │   ├── smart_qa.py
│       │   ├── agent_assist.py
│       │   └── review_queue.py
│       ├── routers/            # API routers for different services
│       │   ├── smart_qa_router.py
│       │   ├── agent_assist_router.py
│       │   └── review_queue_router.py
│       ├── services/           # Business logic for each service
│       │   ├── smart_qa_service.py
│       │   ├── agent_assist_service.py
│       │   ├── review_queue_service.py
│       │   └── rag_service.py  # RAG specific logic
│       └── utils/              # Utility functions (e.g., pagination helpers, datetime formatters)
│           └── pagination.py
├── tests/                    # Unit and integration tests
├── .env.example              # Environment variables template
├── pyproject.toml
└── README.md
```

**关键后续步骤：**

1.  **详细设计服务层 (`services/`):** 每个路由函数中的实际业务逻辑应该放在服务层。例如 `smart_qa_service.py` 会包含调用 RAG 管道、与数据库交互等复杂逻辑。
2.  **数据库交互实现 (`db/`):** 选择数据库并实现数据持久化逻辑。如果使用 SQL 数据库，可以考虑 SQLAlchemy ORM。
3.  **RAG 管道实现 (`services/rag_service.py`):** 实现文本嵌入、向量存储与检索、LLM调用等。
4.  **认证与授权 (`core/security.py`):** 为需要保护的 API 端点添加认证机制。
5.  **错误处理和日志记录:** 完善全局错误处理和结构化日志。

这份设计应该为你构建 "Athena" 后端服务提供了一个坚实的起点。你的前端 API 文档非常出色，使得后端设计能够很好地与之对应！