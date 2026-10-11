from models.schemas import PatentRecord


def get_patent_text_status(patent: PatentRecord) -> tuple[str, int]:
    """Return text availability status and total abstract/claims characters."""
    abstract = patent.patent_abstract or ""
    claims = patent.patent_claims or ""

    text_characters = len(abstract) + len(claims)

    if abstract.strip() and claims.strip():
        status = "full_text"
    elif abstract.strip() or claims.strip():
        status = "partial_text"
    else:
        status = "insufficient_text"

    return status, text_characters