from pydantic import BaseModel
from typing import List, Optional


class ResumeQuery(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SourceChunk(BaseModel):
    text: str
    section: str
    score: float


class ResumeResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceChunk]] = []
    confidence: float


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
