"""Data contracts for every pipeline stage.

All services should consume and produce these types;
never pass raw dicts between stages.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatentRecord(BaseModel):
    """A single patent as returned by the PatentsView API."""

    patent_id: str
    patent_title: str
    patent_abstract: Optional[str] = None
    patent_type: Optional[str] = None
    patent_date: Optional[str] = None
    continuity_family_id: Optional[str] = None

    # USPTO classification data.
    uspc_class: Optional[str] = None
    uspc_subclass: Optional[str] = None
    cpc_classifications: list[str] = Field(default_factory=list)
    ipc_classifications: list[str] = Field(default_factory=list)


class SearchConcept(BaseModel):
    """One patent-search concept and its alternative terminology."""

    name: str
    importance: str = "important"
    terms: list[str] = Field(default_factory=list)
    operator: str = "OR"


class ProximityRule(BaseModel):
    """A structured proximity relationship between two search terms."""

    left_term: str
    right_term: str

    operator: Literal["ADJ", "NEAR"] = "ADJ"

    # Number of searchable terms allowed between the two terms.
    distance: int = 1

    field: Literal["title", "abstract", "claims"] = "claims"


class SearchStrategy(BaseModel):
    """User-approved search strategy for patent retrieval."""

    original_input: str

    key_inventive_points: list[str] = Field(default_factory=list)

    concepts: list[SearchConcept] = Field(default_factory=list)

    # Operator between concept groups.
    # Example:
    # (battery OR rechargeable battery)
    # AND
    # (charging OR recharging)
    concept_operator: str = "AND"

    # Patent-search fields.
    # These will later map to USPTO/WIPO field syntax.
    search_fields: list[str] = Field(
        default_factory=lambda: ["title", "abstract", "claims"]
    )

    # Proximity expressions such as:
    # "blood pressure" NEAR10 "non invasive"
    proximity_rules: list[ProximityRule] = Field(default_factory=list)

    # US / IPC / CPC classifications used for refinement.
    classifications: list[str] = Field(default_factory=list)

    # Generated search strings for each supported search format.
    uspto_search_strings: list[str] = Field(default_factory=list)
    wipo_search_strings: list[str] = Field(default_factory=list)


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
