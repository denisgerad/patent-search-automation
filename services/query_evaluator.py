"""
services/query_evaluator.py

Evaluates query expansion quality based on preservation of domain/function/technology
and avoidance of generic terms.
"""
import re
from typing import Dict, List, Optional, Tuple


# Generic patent filler words that reduce search specificity
GENERIC_TERMS = {
    # "system", "method", "device", "apparatus" intentionally excluded —
    # these are legitimate patent claim terms (apparatus claims, method claims)
    # and must never be penalised.
    "module", "unit",
    "data", "processing", "implementation", "computer", "digital",
    "automatic", "automated", "electronic", "mechanism", "technology",
    "solution", "technique", "process", "procedure", "component"
}

# Overly broad domain terms that need more specificity
BROAD_DOMAINS = {
    "computer vision", "machine learning", "artificial intelligence",
    "data processing", "information technology", "communication",
    "network", "database", "software", "hardware"
}

# Dangerous standalone terms (too generic without context)
DANGEROUS_STANDALONE = {
    "infrared sensors", "sensors", "camera", "detector", "processor",
    "algorithm", "software", "hardware", "network", "database",
    "vision", "learning", "neural network", "deep learning"
}


def evaluate_expansion(
    original_query: str,
    expanded_terms: List[str],
    structured_data: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    Score query expansion quality out of 100 points.
    
    Scoring criteria (out of 100):
    - Structural grouping (20): Quality and completeness of domain/function/technology groups
    - Domain preservation (20): How well domain concepts are preserved
    - Function preservation (20): How well function concepts are preserved
    - Technology preservation (20): How well technology concepts are preserved
    - Generic leakage control (20): Penalty for generic/filler terms
    
    Args:
        original_query: The original user query
        expanded_terms: List of expanded query strings (excluding original)
        structured_data: Optional dict with domain/function/technology lists
        
    Returns:
        Dict with score, breakdown, and evaluation metrics
    """
    breakdown = []
    category_scores = {}
    
    # Extract key tokens from original query (filter out stop words)
    stop_words = {"a", "an", "the", "for", "using", "with", "in", "on", "at", "to", "of", "by"}
    original_tokens = [
        w.lower() for w in re.findall(r'\b\w+\b', original_query)
        if w.lower() not in stop_words and len(w) > 2
    ]
    
    # Identify domain-like words (nouns, application contexts)
    domain_indicators = {"vehicle", "autonomous", "system", "network", "medical", "industrial", "mobile", "web"}
    # Identify function-like words (verbs, actions, purposes)
    function_indicators = {"detection", "tracking", "processing", "control", "analysis", "monitoring", "classification"}
    # Identify technology-like words (sensors, algorithms, materials)
    tech_indicators = {"sensor", "infrared", "camera", "algorithm", "neural", "laser", "lidar", "radar"}
    
    domains = []
    functions = []
    techs = []
    
    if structured_data:
        domains = structured_data.get("domain", [])
        functions = structured_data.get("function", [])
        techs = structured_data.get("technology", [])
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. STRUCTURAL GROUPING (0-20 points)
    # ═══════════════════════════════════════════════════════════════════
    structural_score = 0
    misclassification_found = False
    
    # Check existence and quality of each group
    group_scores = []
    
    # Domain group check
    if domains:
        domain_score = 0
        # Check for misclassified terms (function words in domain)
        for d in domains:
            if any(func in d.lower() for func in function_indicators):
                misclassification_found = True
                domain_score = max(0, domain_score - 2)  # Penalty for misclassification
        
        if len(domains) >= 2:
            domain_score += 7 if not misclassification_found else 4  # Reduced if misclassified
        elif len(domains) == 1:
            domain_score += 4 if not misclassification_found else 2
        group_scores.append(max(0, domain_score))
    else:
        group_scores.append(0)
    
    # Function group check
    if functions:
        func_score = 0
        if len(functions) >= 2:
            func_score = 7
        elif len(functions) == 1:
            func_score = 4
        group_scores.append(func_score)
    else:
        group_scores.append(0)
    
    # Technology group check  
    if techs:
        tech_score = 0
        # Check if tech is too simple (just standalone tech without context)
        simple_techs = [t for t in techs if len(t.split()) <= 2 and any(danger in t.lower() for danger in DANGEROUS_STANDALONE)]
        
        if simple_techs:
            tech_score = 2  # Very low score for dangerous standalone
        elif len(techs) >= 2:
            tech_score = 6
        elif len(techs) == 1:
            tech_score = 3
        group_scores.append(tech_score)
    else:
        group_scores.append(0)
    
    structural_score = sum(group_scores)
    category_scores["structural_grouping"] = structural_score
    notes = " [misclassification detected]" if misclassification_found else ""
    breakdown.append(f"Structural grouping: {structural_score}/20 (D:{group_scores[0]}/7, F:{group_scores[1]}/7, T:{group_scores[2]}/6){notes}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. DOMAIN PRESERVATION (0-20 points)
    # ═══════════════════════════════════════════════════════════════════
    domain_score = 0
    
    # Check if domain tokens from original query are preserved
    combined_expanded = " ".join(expanded_terms).lower()
    domain_tokens_in_query = [t for t in original_tokens if any(ind in t for ind in domain_indicators)]
    
    if domains:
        # Base points for having domain group (reduced if misclassification detected)
        domain_score += 10 if not misclassification_found else 7
        
        # Check token preservation in domain terms - be strict
        domain_text = " ".join(domains).lower()
        preserved_domain_tokens = [t for t in domain_tokens_in_query if t in domain_text]
        
        if domain_tokens_in_query:
            preservation_ratio = len(preserved_domain_tokens) / len(domain_tokens_in_query)
            # Stricter grading: need high preservation for full points
            if preservation_ratio >= 0.8:
                domain_score += 8
            elif preservation_ratio >= 0.5:
                domain_score += 5
            else:
                domain_score += 2
        else:
            # If no clear domain indicators, check general presence - partial credit only
            relevant_tokens = [t for t in original_tokens[:3] if t in domain_text]
            domain_score += min(5, len(relevant_tokens) * 2)
    else:
        # No domain group - maximum 5 points if domain concepts still appear
        if domain_tokens_in_query and any(t in combined_expanded for t in domain_tokens_in_query):
            domain_score += 3
    
    category_scores["domain_preservation"] = domain_score
    breakdown.append(f"Domain preservation: {domain_score}/20")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. FUNCTION PRESERVATION (0-20 points)
    # ═══════════════════════════════════════════════════════════════════
    function_score = 0
    
    function_tokens_in_query = [t for t in original_tokens if any(ind in t for ind in function_indicators)]
    
    if functions:
        # Base points for having function group
        function_score += 8
        
        # Check token preservation - strict
        function_text = " ".join(functions).lower()
        preserved_function_tokens = [t for t in function_tokens_in_query if t in function_text]
        
        if function_tokens_in_query:
            preservation_ratio = len(preserved_function_tokens) / len(function_tokens_in_query)
            # Stricter grading
            if preservation_ratio >= 0.8:
                function_score += 7
            elif preservation_ratio >= 0.5:
                function_score += 4
            else:
                function_score += 2
        else:
            # Check if key action/purpose words are preserved
            action_words = [t for t in original_tokens if t in function_indicators or t.endswith("ing") or t.endswith("ion")]
            matching = [w for w in action_words if w in function_text]
            function_score += min(5, len(matching) * 2)
    else:
        # No function group - maximum 5 points
        if function_tokens_in_query and any(t in combined_expanded for t in function_tokens_in_query):
            function_score += 3
    
    category_scores["function_preservation"] = function_score
    breakdown.append(f"Function preservation: {function_score}/20")
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. TECHNOLOGY PRESERVATION (0-20 points)
    # ═══════════════════════════════════════════════════════════════════
    technology_score = 0
    
    tech_tokens_in_query = [t for t in original_tokens if any(ind in t for ind in tech_indicators)]
    
    if techs:
        # Base points for having technology group
        technology_score += 10
        
        # Check token preservation and specificity - strict
        tech_text = " ".join(techs).lower()
        preserved_tech_tokens = [t for t in tech_tokens_in_query if t in tech_text]
        
        if tech_tokens_in_query:
            preservation_ratio = len(preserved_tech_tokens) / len(tech_tokens_in_query)
            if preservation_ratio >= 0.8:
                technology_score += 10
            elif preservation_ratio >= 0.5:
                technology_score += 6
            else:
                technology_score += 3
        else:
            # Check if any technical terms are preserved
            tech_in_original = [t for t in original_tokens if len(t) > 4]  # Longer words often technical
            matching = [t for t in tech_in_original if t in tech_text]
            technology_score += min(5, len(matching))
    else:
        # No tech group - maximum 5 points
        if tech_tokens_in_query and any(t in combined_expanded for t in tech_tokens_in_query):
            technology_score += 3
    
    category_scores["technology_preservation"] = technology_score
    breakdown.append(f"Technology preservation: {technology_score}/20")
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. GENERIC LEAKAGE CONTROL (0-20 points, penalties applied)
    # ═══════════════════════════════════════════════════════════════════
    leakage_score = 20  # Start with full points, deduct for issues
    leakage_issues = []
    
    # Check for dangerous standalone terms (CRITICAL PENALTY)
    dangerous_found = []
    for term in expanded_terms:
        term_lower = term.lower().strip()
        for danger in DANGEROUS_STANDALONE:
            if term_lower == danger or (len(term.split()) <= 2 and danger in term_lower):
                dangerous_found.append(term)
                break
    
    if dangerous_found:
        # Very heavy penalty: 12 points per dangerous standalone term
        penalty = min(20, len(dangerous_found) * 12)
        leakage_score -= penalty
        leakage_issues.append(f"Dangerous standalone terms: {', '.join(dangerous_found[:2])}")
    
    # Find generic filler terms in expanded queries
    generic_found = []
    for term in expanded_terms:
        term_lower = term.lower()
        for generic in GENERIC_TERMS:
            if re.search(r'\b' + re.escape(generic) + r'\b', term_lower):
                generic_found.append(generic)
    
    unique_generics = list(set(generic_found))
    
    # Deduct 3 points per unique generic term
    if unique_generics:
        penalty = min(6, len(unique_generics) * 3)
        leakage_score -= penalty
        leakage_issues.append(f"Generic filler: {', '.join(unique_generics[:2])}")
    
    # Check for single-word terms (weak for Boolean search)
    single_word_terms = [t for t in expanded_terms if len(t.split()) == 1]
    if len(single_word_terms) >= 2:
        leakage_score -= 4
        leakage_issues.append(f"{len(single_word_terms)} single-word terms")
    
    # Check for overly broad domains
    broad_found = []
    for term in expanded_terms:
        term_lower = term.lower()
        for broad in BROAD_DOMAINS:
            if broad in term_lower:
                broad_found.append(broad)
    
    if broad_found:
        leakage_score -= 3
        leakage_issues.append(f"Broad terms: {', '.join(list(set(broad_found))[:2])}")
    
    leakage_score = max(0, leakage_score)  # Don't go negative
    category_scores["generic_leakage_control"] = leakage_score
    
    if leakage_issues:
        breakdown.append(f"Generic leakage control: {leakage_score}/20 ({'; '.join(leakage_issues)})")
    else:
        breakdown.append(f"Generic leakage control: {leakage_score}/20 ✓")
    
    # ═══════════════════════════════════════════════════════════════════
    # TOTAL SCORE
    # ═══════════════════════════════════════════════════════════════════
    total_score = sum(category_scores.values())
    
    # ─── Calculate additional metrics ───
    avg_words_per_term = sum(len(t.split()) for t in expanded_terms) / len(expanded_terms) if expanded_terms else 0
    if avg_words_per_term >= 3:
        specificity = "High"
    elif avg_words_per_term >= 2:
        specificity = "Medium"
    else:
        specificity = "Low"
    
    multi_word_ratio = 1 - (len(single_word_terms) / len(expanded_terms)) if expanded_terms else 0
    if multi_word_ratio >= 0.8:
        boolean_fitness = "Strong"
    elif multi_word_ratio >= 0.5:
        boolean_fitness = "Medium"
    else:
        boolean_fitness = "Weak"
    
    preserved_count = sum(1 for tok in original_tokens if tok in combined_expanded)
    preservation_ratio = preserved_count / len(original_tokens) if original_tokens else 0
    
    return {
        "score": total_score,
        "normalized_score": round(total_score / 10, 1),  # Convert to 0-10 scale
        "breakdown": breakdown,
        "category_scores": category_scores,
        "metrics": {
            "domain_preserved": "Yes" if domains else "Partial" if domain_score > 0 else "No",
            "function_preserved": "Yes" if functions else "Partial" if function_score > 0 else "No",
            "technology_preserved": "Yes" if techs else "Partial" if technology_score > 0 else "No",
            "specificity": specificity,
            "dangerous_generic_terms": len(unique_generics) > 0,
            "boolean_fitness": boolean_fitness,
            "token_preservation_ratio": round(preservation_ratio, 2),
            "single_word_count": len(single_word_terms),
            "generic_term_count": len(unique_generics),
        }
    }
