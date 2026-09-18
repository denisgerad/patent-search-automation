from models.schemas import PatentRecord


def _normalize(value: str) -> str:
    """Normalize classification text for comparison."""
    return " ".join(str(value).strip().upper().split())


def _matches_any(
    values: list[str],
    selected: list[str],
) -> bool:
    """Return True when any selected value exists in the record values."""
    normalized_values = {_normalize(value) for value in values if value}
    normalized_selected = {
        _normalize(value) for value in selected if value
    }

    return bool(normalized_values & normalized_selected)


def filter_patents_by_classification(
    patents: list[PatentRecord],
    selected_cpc: list[str] | None = None,
    selected_uspc_class: list[str] | None = None,
    selected_uspc_subclass: list[str] | None = None,
    operator: str = "AND",
) -> list[PatentRecord]:
    """
    Filter an existing USPTO PatentRecord list by selected classifications.

    This performs local refinement against classifications already retrieved
    from USPTO ODP. It does not make another USPTO API request.

    operator:
        AND = patent must match every selected classification group
        OR  = patent may match any selected classification group
    """

    selected_cpc = selected_cpc or []
    selected_uspc_class = selected_uspc_class or []
    selected_uspc_subclass = selected_uspc_subclass or []

    operator = operator.upper()

    if operator not in {"AND", "OR"}:
        raise ValueError(
            f"Unsupported classification operator: {operator}"
        )

    groups = []

    if selected_cpc:
        groups.append(
            lambda patent: _matches_any(
                patent.cpc_classifications,
                selected_cpc,
            )
        )

    if selected_uspc_class:
        groups.append(
            lambda patent: _normalize(patent.uspc_class or "")
            in {
                _normalize(value)
                for value in selected_uspc_class
                if value
            }
        )

    if selected_uspc_subclass:
        groups.append(
            lambda patent: _normalize(patent.uspc_subclass or "")
            in {
                _normalize(value)
                for value in selected_uspc_subclass
                if value
            }
        )

    if not groups:
        return list(patents)

    refined = []

    for patent in patents:
        matches = [group(patent) for group in groups]

        if operator == "AND":
            keep = all(matches)
        else:
            keep = any(matches)

        if keep:
            refined.append(patent)

    return refined
