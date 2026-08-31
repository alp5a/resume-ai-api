from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import database
from app.models import HealthResponse, ResumeQuery, ResumeResponse, SourceChunk
from app.resume_data import build_chunks, load_resume

app = FastAPI(
    title="Resume AI API",
    description="A small retrieval-based API that answers questions about a resume.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_resume_cache = None


@app.on_event("startup")
def startup_event():
    global _resume_cache
    _resume_cache = load_resume()
    chunks = build_chunks(_resume_cache)
    count = database.index_chunks(chunks)
    print(f"✅ Indexed {count} resume chunks")


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Resume AI API is running.",
        "docs": "/docs",
        "endpoints": ["/api/query", "/api/resume", "/health"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", chunks_indexed=database.chunk_count())


@app.get("/api/resume", tags=["resume"])
def get_full_resume():
    if _resume_cache is None:
        raise HTTPException(status_code=503, detail="Resume not loaded yet.")
    return _resume_cache


@app.post("/api/query", response_model=ResumeResponse, tags=["resume"])
def query_resume(payload: ResumeQuery):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    results = database.query_chunks(payload.question, top_k=payload.top_k or 3)

    if not results:
        return ResumeResponse(
            answer="I couldn't find anything relevant in the resume for that question.",
            sources=[],
            confidence=0.0,
        )

    top = results[0]
    # Simple extractive answer: lead with the best-matching chunk, and if
    # other chunks are close in relevance, include a short excerpt of the
    # runner-up too.
    answer = top["text"]
    if len(results) > 1 and results[1]["score"] > 0.4:
        answer += " " + results[1]["text"]

    sources = [SourceChunk(text=r["text"], section=r["section"], score=round(r["score"], 3)) for r in results]

    return ResumeResponse(answer=answer, sources=sources, confidence=round(top["score"], 3))
