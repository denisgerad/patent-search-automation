#!/usr/bin/env python3
"""
Quick test of the new scoring system with the example query.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.query_evaluator import evaluate_expansion

# Example from user's request
original_query = "A system for autonomous vehicle lane detection using infrared sensors"

expanded_terms = [
    "vehicle lane detection infrared sensors",
    "infrared sensors",
    "autonomous vehicle infrared sensors",
    "lane detection infrared sensors"
]

structured_data = {
    "domain": ["autonomous vehicle"],
    "function": ["lane detection"],
    "technology": ["infrared sensors"]
}

# Evaluate
result = evaluate_expansion(original_query, expanded_terms, structured_data)

# Display results
print("=" * 70)
print("QUERY EXPANSION EVALUATION")
print("=" * 70)
print(f"\nOriginal Query: {original_query}")
print("\nExpanded Terms:")
for i, term in enumerate(expanded_terms, 1):
    print(f"  {i}. {term}")

print("\nStructured Data:")
print(f"  Domain: {structured_data['domain']}")
print(f"  Function: {structured_data['function']}")
print(f"  Technology: {structured_data['technology']}")

print("\n" + "=" * 70)
print("SCORING RESULTS")
print("=" * 70)
print(f"\nTotal Score: {result['score']}/100")
print(f"Normalized: {result['normalized_score']}/10")

print("\nCategory Breakdown:")
for category, score in result['category_scores'].items():
    category_name = category.replace('_', ' ').title()
    print(f"  {category_name:30s} {score:2d}/20")

print("\nDetailed Breakdown:")
for line in result['breakdown']:
    print(f"  • {line}")

print("\nMetrics:")
metrics = result['metrics']
print(f"  Domain preserved: {metrics['domain_preserved']}")
print(f"  Function preserved: {metrics['function_preserved']}")
print(f"  Technology preserved: {metrics['technology_preserved']}")
print(f"  Specificity: {metrics['specificity']}")
print(f"  Boolean fitness: {metrics['boolean_fitness']}")
print(f"  Token preservation: {int(metrics['token_preservation_ratio']*100)}%")
print(f"  Generic term count: {metrics['generic_term_count']}")
print(f"  Single-word terms: {metrics['single_word_count']}")

print("\n" + "=" * 70)
