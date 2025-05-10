"""
Tests for the Smart QA service.
"""

import pytest
from fastapi.testclient import TestClient
from athena.main import app
from athena.schemas.smart_qa import UserQueryRequest


client = TestClient(app)


def test_api_health():
    """Test the API health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_submit_query():
    """Test the query submission endpoint."""
    request_data = {
        "query": "What are the shipping options for international packages?",
        "sessionId": "test-session-123",
        "userId": "test-user-456"
    }
    
    response = client.post("/api/query", json=request_data)
    assert response.status_code == 200
    
    # Verify basic response structure
    json_response = response.json()
    assert json_response["code"] == 200
    assert json_response["message"] == "成功"
    assert "data" in json_response
    assert "requestId" in json_response
    
    # Verify data structure
    data = json_response["data"]
    assert data["originalQuery"] == request_data["query"]
    assert "suggestedAnswer" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert isinstance(data["needsHumanReview"], bool)


def test_submit_feedback():
    """Test the feedback submission endpoint."""
    feedback_data = {
        "queryId": "mock-query-123",
        "responseId": "mock-response-456",
        "feedbackType": "helpful", 
        "comment": "This was a good answer",
        "userId": "test-user-789"
    }
    
    response = client.post("/api/feedback", json=feedback_data)
    assert response.status_code == 200
    
    json_response = response.json()
    assert json_response["code"] == 200
    assert json_response["data"]["success"] is True