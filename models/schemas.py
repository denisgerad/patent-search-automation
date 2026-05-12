"""Data contracts for every pipeline stage.

All services should consume and produce these types;
never pass raw dicts between stages.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PatentRecord(BaseModel):
    """A single patent as returned by the PatentsView API."""

    patent_id: str
    patent_title: str
    patent_abstract: Optional[str] = None
    patent_type: Optional[str] = None
    patent_date: Optional[str] = None


class RankedPatent(BaseModel):
    """A patent with its hybrid-ranking scores attached."""

    patent: PatentRecord
    cosine_score: float
    bm25_score: float
    hybrid_score: float
    coverage: float = 0.0
    concept_hits: dict = {}


class PipelineResult(BaseModel):
    """The full output of one end-to-end pipeline run."""

    query: str
    expanded_queries: list[str]
    ranked_patents: list[RankedPatent]
    comparison_summary: str
    report_markdown: str
