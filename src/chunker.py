"""
ContextCore — Truth Document Chunker
Splits the structured truth document into RAG-ready text chunks.
Each top-level section becomes its own chunk so retrieval is precise.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from schema import TruthDocument


@dataclass
class Chunk:
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_truth_document(doc: TruthDocument) -> List[Chunk]:
    chunks: List[Chunk] = []

    # Identity
    chunks.append(Chunk(
        section="identity",
        text=(
            f"Name: {doc.identity.name}. Role: {doc.identity.role}. "
            f"Institution: {doc.identity.institution}. Location: {doc.identity.location}. "
            f"CGPA: {doc.identity.cgpa}. Graduating: {doc.identity.graduation_year}."
        ),
        metadata={"section": "identity", "owner": doc.meta.owner}
    ))

    # Experience — one chunk per entry
    for exp in doc.experience:
        chunks.append(Chunk(
            section="experience",
            text=(
                f"Experience at {exp.org} as {exp.role} ({exp.type}). "
                f"Skills used: {', '.join(exp.skills_used)}. Summary: {exp.summary}."
            ),
            metadata={"section": "experience", "org": exp.org, "role": exp.role}
        ))

    # Projects — one chunk per project
    for proj in doc.projects:
        chunks.append(Chunk(
            section="projects",
            text=(
                f"Project: {proj.name} ({proj.type}, status: {proj.status}). "
                f"Stack: {', '.join(proj.stack)}. Summary: {proj.summary}."
                + (f" Repo: {proj.repo}." if proj.repo else "")
            ),
            metadata={"section": "projects", "name": proj.name, "status": proj.status}
        ))

    # Skills
    chunks.append(Chunk(
        section="skills",
        text=(
            f"Programming languages: {', '.join(doc.skills.languages)}. "
            f"Domains: {', '.join(doc.skills.domains)}. "
            f"Tools and frameworks: {', '.join(doc.skills.tools)}."
        ),
        metadata={"section": "skills"}
    ))

    # Goals
    chunks.append(Chunk(
        section="goals",
        text=(
            f"Immediate goal: {doc.goals.immediate}. "
            f"Constraints: {', '.join(doc.goals.constraints)}. "
            f"Internship tracks: {', '.join(doc.goals.tracks)}."
        ),
        metadata={"section": "goals"}
    ))

    # Preferences
    if doc.preferences.avoid_assumptions or doc.preferences.communication_style:
        chunks.append(Chunk(
            section="preferences",
            text=(
                f"Communication style: {doc.preferences.communication_style}. "
                f"Do not assume: {', '.join(doc.preferences.avoid_assumptions)}."
            ),
            metadata={"section": "preferences"}
        ))

    # Ongoing
    if doc.ongoing.current_focus:
        chunks.append(Chunk(
            section="ongoing",
            text=(
                f"Current focus: {doc.ongoing.current_focus}. "
                f"Active tasks: {', '.join(doc.ongoing.active_tasks)}."
            ),
            metadata={"section": "ongoing"}
        ))

    return chunks


if __name__ == "__main__":
    doc = TruthDocument.from_json("data/truth_document.json")
    chunks = chunk_truth_document(doc)
    print(f"✓ Generated {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c.section}] {c.text[:80]}...")
