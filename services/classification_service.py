from collections import Counter
from models.schemas import PatentRecord


def aggregate_classifications(
    patents: list[PatentRecord],
) -> dict:
    """Aggregate classification frequencies across patent candidates."""

    cpc_counter = Counter()
    uspc_counter = Counter()
    ipc_counter = Counter()

    for patent in patents:

        for cpc in patent.cpc_classifications:
            cpc = cpc.strip()
            if cpc:
                cpc_counter[cpc] += 1

        if patent.uspc_class:
            uspc = patent.uspc_class.strip()

            if uspc:
                uspc_counter[uspc] += 1

        if patent.uspc_class and patent.uspc_subclass:
            uspc_full = (
                f"{patent.uspc_class.strip()}/"
                f"{patent.uspc_subclass.strip()}"
            )
        else:
            uspc_full = ""

        if uspc_full:
            uspc_counter[uspc_full] += 1

        for ipc in patent.ipc_classifications:
            ipc = ipc.strip()
            if ipc:
                ipc_counter[ipc] += 1

    return {
        "cpc": cpc_counter,
        "uspc": uspc_counter,
        "ipc": ipc_counter,
    }
from collections import Counter
from models.schemas import PatentRecord


def aggregate_classifications(
    patents: list[PatentRecord],
) -> dict:
    """
    Aggregate classification frequencies across patent candidates.

    Each classification is counted at most once per patent.
    USPC class and subclass are kept as separate levels.
    """

    cpc_counter = Counter()
    ipc_counter = Counter()
    uspc_class_counter = Counter()
    uspc_subclass_counter = Counter()

    for patent in patents:

        # ── CPC ──────────────────────────────────────────────────────────
        # Count each CPC only once for a given patent.
        seen_cpc = set()

        for cpc in patent.cpc_classifications:
            cpc = cpc.strip()

            if cpc and cpc not in seen_cpc:
                cpc_counter[cpc] += 1
                seen_cpc.add(cpc)

        # ── USPC ─────────────────────────────────────────────────────────
        if patent.uspc_class:
            uspc_class = patent.uspc_class.strip()

            if uspc_class:
                uspc_class_counter[uspc_class] += 1

        if patent.uspc_subclass:
            uspc_subclass = patent.uspc_subclass.strip()

            if uspc_subclass:
                uspc_subclass_counter[uspc_subclass] += 1

        # ── IPC ──────────────────────────────────────────────────────────
        # Currently expected to be empty for the ODP response
        # we have inspected, but retained for future enrichment.
        seen_ipc = set()

        for ipc in patent.ipc_classifications:
            ipc = ipc.strip()

            if ipc and ipc not in seen_ipc:
                ipc_counter[ipc] += 1
                seen_ipc.add(ipc)

    return {
        "cpc": cpc_counter,
        "uspc": {
            "class": uspc_class_counter,
            "subclass": uspc_subclass_counter,
        },
        "ipc": ipc_counter,
    }
from collections import Counter
from models.schemas import PatentRecord


def aggregate_classifications(
    patents: list[PatentRecord],
) -> dict:
    """
    Aggregate classification frequencies across patent candidates.

    Each classification is counted at most once per patent.

    The aggregation also preserves patent IDs so that classifications
    can later be used for classification-based search refinement.
    """

    cpc_counter = Counter()
    ipc_counter = Counter()
    uspc_class_counter = Counter()
    uspc_subclass_counter = Counter()

    cpc_patents = {}
    ipc_patents = {}
    uspc_class_patents = {}
    uspc_subclass_patents = {}

    for patent in patents:
        patent_id = patent.patent_id

        # ---------------------------------------------------------
        # CPC
        # ---------------------------------------------------------
        seen_cpc = set()

        for cpc in patent.cpc_classifications:
            cpc = cpc.strip()

            if not cpc or cpc in seen_cpc:
                continue

            cpc_counter[cpc] += 1
            cpc_patents.setdefault(cpc, []).append(patent_id)

            seen_cpc.add(cpc)

        # ---------------------------------------------------------
        # USPC class
        # ---------------------------------------------------------
        if patent.uspc_class:
            uspc_class = patent.uspc_class.strip()

            if uspc_class:
                uspc_class_counter[uspc_class] += 1
                uspc_class_patents.setdefault(
                    uspc_class,
                    [],
                ).append(patent_id)

        # ---------------------------------------------------------
        # USPC subclass
        # ---------------------------------------------------------
        if patent.uspc_subclass:
            uspc_subclass = patent.uspc_subclass.strip()

            if uspc_subclass:
                uspc_subclass_counter[uspc_subclass] += 1
                uspc_subclass_patents.setdefault(
                    uspc_subclass,
                    [],
                ).append(patent_id)

        # ---------------------------------------------------------
        # IPC
        # ---------------------------------------------------------
        seen_ipc = set()

        for ipc in patent.ipc_classifications:
            ipc = ipc.strip()

            if not ipc or ipc in seen_ipc:
                continue

            ipc_counter[ipc] += 1
            ipc_patents.setdefault(ipc, []).append(patent_id)

            seen_ipc.add(ipc)

    # -------------------------------------------------------------
    # Convert counters + patent lists into a UI/search-friendly
    # structure.
    # -------------------------------------------------------------

    cpc = {
        classification: {
            "count": cpc_counter[classification],
            "patent_ids": cpc_patents.get(classification, []),
        }
        for classification in cpc_counter
    }

    ipc = {
        classification: {
            "count": ipc_counter[classification],
            "patent_ids": ipc_patents.get(classification, []),
        }
        for classification in ipc_counter
    }

    uspc_class = {
        classification: {
            "count": uspc_class_counter[classification],
            "patent_ids": uspc_class_patents.get(classification, []),
        }
        for classification in uspc_class_counter
    }

    uspc_subclass = {
        classification: {
            "count": uspc_subclass_counter[classification],
            "patent_ids": uspc_subclass_patents.get(
                classification,
                [],
            ),
        }
        for classification in uspc_subclass_counter
    }

    return {
        "cpc": cpc,
        "uspc": {
            "class": uspc_class,
            "subclass": uspc_subclass,
        },
        "ipc": ipc,
    }


def build_cpc_query(classifications: list[str]) -> str:
    """
    Build a USPTO CPC classification expression.

    USPTO CPC syntax does not include spaces in the classification
    symbol, e.g. G06V 20/588 -> G06V20/588.CPC.
    """
    values = []

    for classification in classifications:
        classification = classification.strip()

        if not classification:
            continue

        normalized = classification.replace(" ", "")
        values.append(f"{normalized}.CPC.")

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    return "(" + " OR ".join(values) + ")"


def build_uspc_class_query(classifications: list[str]) -> str:
    """
    Build a USPTO USPC class expression.

    CLAS searches the USPC classification text. USPTO examples
    show quoted class values such as "435".CLAS.
    """
    values = []

    for classification in classifications:
        classification = classification.strip()

        if not classification:
            continue

        values.append(f'"{classification}".CLAS.')

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    return "(" + " OR ".join(values) + ")"


def build_classification_query(
    cpc_classifications: list[str] | None = None,
    uspc_classes: list[str] | None = None,
    operator: str = "AND",
) -> str:
    """
    Build a combined USPTO classification expression.

    CPC and USPC class groups are combined using the supplied
    operator. Terms within each classification family use OR.
    """

    cpc_classifications = cpc_classifications or []
    uspc_classes = uspc_classes or []

    operator = operator.upper()

    if operator not in {"AND", "OR"}:
        raise ValueError(
            "Classification operator must be AND or OR"
        )

    groups = []

    cpc_query = build_cpc_query(cpc_classifications)

    if cpc_query:
        groups.append(cpc_query)

    uspc_query = build_uspc_class_query(uspc_classes)

    if uspc_query:
        groups.append(uspc_query)

    if not groups:
        return ""

    if len(groups) == 1:
        return groups[0]

    return f" {operator} ".join(groups)


def combine_with_original_search(
    original_search: str,
    classification_query: str,
    operator: str = "AND",
) -> str:
    """
    Combine an existing USPTO search expression with a
    classification expression.

    This function only builds the query. It does not execute
    a USPTO search.
    """

    original_search = original_search.strip()
    classification_query = classification_query.strip()

    operator = operator.upper()

    if operator not in {"AND", "OR"}:
        raise ValueError(
            "Search combination operator must be AND or OR"
        )

    if not original_search:
        return classification_query

    if not classification_query:
        return original_search

    return (
        f"({original_search}) "
        f"{operator} "
        f"({classification_query})"
    )
