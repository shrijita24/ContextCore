"""
ContextCore — Contradiction Detection
LLM-as-judge module: given an arbitrary statement (e.g. something another
AI assistant said about the user), retrieves the most relevant facts from
the truth document and asks a judge model whether the statement conflicts
with them. Runs on the same free stack as rag_pipeline.py (local
embeddings + Groq), no OpenAI dependency.
"""

from __future__ import annotations
import os
import json
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from vector_store import search

load_dotenv()

# Reuse the same free Groq model as rag_pipeline.py for consistency.
JUDGE_MODEL = "openai/gpt-oss-20b"

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You are given:
1. A STATEMENT — something an AI assistant said about a person.
2. TRUTH — verified facts about that person, retrieved from their truth document.

Decide whether the STATEMENT contradicts the TRUTH. A contradiction means the \
statement asserts something that directly conflicts with a fact in TRUTH \
(wrong institution, wrong CGPA, wrong role, wrong project status, etc). \
It is NOT a contradiction if the statement simply adds information not present \
in TRUTH, or is vague/general.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "contradicts": true or false,
  "confidence": a number from 0 to 1,
  "explanation": "one or two sentences justifying the verdict"
}}
"""

JUDGE_USER_PROMPT = """TRUTH:
{context}

STATEMENT:
{statement}
"""


class ContradictionResult(TypedDict):
    contradicts: bool
    confidence: float
    explanation: str
    evidence: List[Document]


def _format_context(matches: list) -> str:
    return "\n".join(
        f"- [{m['metadata'].get('section', 'unknown')}] {m['text']}" for m in matches
    )


def check_contradiction(statement: str, top_k: int = 3) -> ContradictionResult:
    """
    Judges whether `statement` contradicts the truth document.

    1. Retrieves the top_k most relevant truth-document chunks for the
       statement (same free local-embedding search as vector_store.py).
    2. Asks a Groq-hosted judge model to compare the statement against
       those chunks and return a structured verdict.
    """
    matches = search(statement, top_k=top_k)
    context = _format_context(matches)

    llm = ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])

    response = llm.invoke([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_USER_PROMPT.format(context=context, statement=statement)),
    ])

    raw = response.content.strip()
    # Judge models occasionally wrap JSON in ```json fences despite instructions —
    # strip those defensively rather than letting json.loads fail on them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if raw.lower().startswith("json") else raw

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "contradicts": False,
            "confidence": 0.0,
            "explanation": f"Judge model returned unparseable output: {raw[:200]}",
        }

    evidence = [
        Document(page_content=m["text"], metadata=m["metadata"]) for m in matches
    ]

    return {
        "contradicts": bool(parsed.get("contradicts", False)),
        "confidence": float(parsed.get("confidence", 0.0)),
        "explanation": parsed.get("explanation", ""),
        "evidence": evidence,
    }


if __name__ == "__main__":
    test_cases = [
        "She studies at IEM Kolkata and has a CGPA of 9.25.",              # should NOT contradict
        "She studies at IIT Bombay and has a CGPA of 7.1.",                # should contradict
        "She is looking for a founder's-office or GenAI internship role.", # should NOT contradict
    ]

    for stmt in test_cases:
        print(f"\nStatement: {stmt}")
        result = check_contradiction(stmt)
        verdict = "⚠️  CONTRADICTION" if result["contradicts"] else "✓ Consistent"
        print(f"{verdict}  (confidence: {result['confidence']:.2f})")
        print(f"  {result['explanation']}")
