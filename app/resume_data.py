"""
Loads the resume JSON from /data/sample_resume.json and converts it into a
list of small text "chunks" that can be embedded and searched. Each chunk
keeps track of which section of the resume it came from.
"""

import json
import os
from typing import Dict, List

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sample_resume.json",
)


def load_resume(path: str = DATA_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks(resume: Dict = None) -> List[Dict]:
    """
    Turns the structured resume dict into a flat list of
    {"text": ..., "section": ...} chunks suitable for embedding.
    """
    if resume is None:
        resume = load_resume()

    chunks: List[Dict] = []

    info = resume.get("personal_info", {})
    if info:
        chunks.append(
            {
                "section": "personal_info",
                "text": (
                    f"{info.get('name', '')} is a {info.get('title', '')} "
                    f"based in {info.get('location', '')}. "
                    f"Contact: {info.get('email', '')}, {info.get('phone', '')}. "
                    f"LinkedIn: {info.get('linkedin', '')}. GitHub: {info.get('github', '')}."
                ),
            }
        )

    if resume.get("summary"):
        chunks.append({"section": "summary", "text": resume["summary"]})

    if resume.get("skills"):
        skills = resume["skills"]
        if isinstance(skills, dict):
            for category, items in skills.items():
                chunks.append(
                    {
                        "section": "skills",
                        "text": f"{category} skills: {', '.join(items)}.",
                    }
                )
        else:
            chunks.append({"section": "skills", "text": "Skills: " + ", ".join(skills)})

    for job in resume.get("experience", []):
        text = (
            f"{job.get('title', '')} at {job.get('company', '')} "
            f"({job.get('start_date', '')} - {job.get('end_date', 'Present')}), "
            f"{job.get('location', '')}. "
            + " ".join(job.get("bullets", []))
        )
        chunks.append({"section": "experience", "text": text})

    for proj in resume.get("projects", []):
        text = (
            f"Project: {proj.get('name', '')}. {proj.get('description', '')} "
            f"Technologies: {', '.join(proj.get('technologies', []))}."
        )
        chunks.append({"section": "projects", "text": text})

    for edu in resume.get("education", []):
        text = (
            f"{edu.get('degree', '')} from {edu.get('institution', '')} "
            f"({edu.get('start_date', '')} - {edu.get('end_date', '')}). "
            f"{edu.get('details', '')}"
        )
        chunks.append({"section": "education", "text": text})

    for cert in resume.get("certifications", []):
        text = f"Certification: {cert.get('name', '')} from {cert.get('issuer', '')} ({cert.get('date', '')})."
        chunks.append({"section": "certifications", "text": text})

    return chunks
