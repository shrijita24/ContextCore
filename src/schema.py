"""
ContextCore — Truth Document Schema
Pydantic models for validating and loading the user truth document.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Meta(BaseModel):
    version: str
    last_updated: str
    owner: str


class Identity(BaseModel):
    name: str
    role: str
    institution: Optional[str] = None
    location: Optional[str] = None
    cgpa: Optional[float] = None
    graduation_year: Optional[int] = None


class Experience(BaseModel):
    org: str
    role: str
    type: Literal["internship", "job", "freelance", "volunteer"]
    skills_used: List[str] = Field(default_factory=list)
    summary: str


class Project(BaseModel):
    name: str
    type: Literal["personal", "fyp", "academic", "open_source"]
    stack: List[str] = Field(default_factory=list)
    summary: str
    status: Literal["complete", "in_progress", "planned"]
    repo: Optional[str] = None


class Skills(BaseModel):
    languages: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)


class Goals(BaseModel):
    immediate: str
    constraints: List[str] = Field(default_factory=list)
    tracks: List[str] = Field(default_factory=list)


class Preferences(BaseModel):
    communication_style: Optional[str] = None
    avoid_assumptions: List[str] = Field(default_factory=list)


class Ongoing(BaseModel):
    current_focus: Optional[str] = None
    active_tasks: List[str] = Field(default_factory=list)


class TruthDocument(BaseModel):
    """
    Root model for the ContextCore truth document.
    Validates the full JSON and provides typed access to every field.
    """
    meta: Meta
    identity: Identity
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Skills
    goals: Goals
    preferences: Preferences
    ongoing: Ongoing

    @classmethod
    def from_json(cls, path: str) -> "TruthDocument":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        exp_orgs = ", ".join(e.org for e in self.experience)
        proj_names = ", ".join(p.name for p in self.projects)
        return (
            f"{self.identity.name} is a {self.identity.role} at "
            f"{self.identity.institution or 'university'} (CGPA {self.identity.cgpa}). "
            f"Experience at: {exp_orgs}. "
            f"Projects: {proj_names}. "
            f"Current goal: {self.goals.immediate}."
        )


if __name__ == "__main__":
    doc = TruthDocument.from_json("data/truth_document.json")
    print("✓ Schema valid")
    print(doc.summary())
