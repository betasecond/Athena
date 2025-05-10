"""
Main application module for the Athena API service.
"""

import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from athena.core.config import settings

# Application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A continually learning RAG-powered AI customer service agent specialized for the logistics industry.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    debug=settings.DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add process time tracking middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Measure and record API processing time."""
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response

# Root endpoint
@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
    }

# Import and include routers
from athena.routers.smart_qa_router import router as smart_qa_router
app.include_router(smart_qa_router)

# Import other routers when they're created
# from athena.routers import agent_assist_router, review_queue_router
# app.include_router(agent_assist_router.router)
# app.include_router(review_queue_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("athena.main:app", host="0.0.0.0", port=8000, reload=True)