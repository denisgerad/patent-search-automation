# Usage: python scripts/test_patentsview.py

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("patentsview_api_key", "").strip()
URL = "https://search.patentsview.org/api/v1/patent/"

print(f"API key present: {'YES' if API_KEY else 'NO — check .env'}")
print(f"Key prefix: {API_KEY[:8]}..." if API_KEY else "")

# Test 1: connectivity + known patent
print("\n[Test 1] Fetch known patent 10000000...")
payload = {
    "q": {"_eq": {"patent_id": "10000000"}},
    "f": [
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_type",
        "patent_date",
    ],
}
resp = requests.post(
    URL,
    headers={
        "X-Api-Key": API_KEY,
        "Content-Type": "application/json",
    },
    json=payload,
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    patents = data.get("patents", [])
    if patents:
        p = patents[0]
        print(f"  ID:       {p.get('patent_id')}")
        print(f"  Title:    {p.get('patent_title')}")
        print(f"  Abstract: {(p.get('patent_abstract') or '')[:120]}...")
        print(f"  Fields returned: {list(p.keys())}")
        print("  ✅ Test 1 PASSED")
    else:
        print("  ❌ No patents returned — check field names or API key")
else:
    print(f"  ❌ HTTP error: {resp.text[:300]}")

# Test 2: keyword search
print("\n[Test 2] Keyword search: 'infrared lane detection'...")
payload2 = {
    "q": {
        "_and": [
            {"_text_phrase": {"patent_abstract": "lane detection"}},
            {"_text_any": {"patent_abstract": "infrared"}},
        ]
    },
    "f": ["patent_id", "patent_title", "patent_abstract"],
    "o": {"size": 5},
}
resp2 = requests.post(
    URL,
    headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
    json=payload2,
)
print(f"Status: {resp2.status_code}")
if resp2.status_code == 200:
    data2 = resp2.json()
    patents2 = data2.get("patents", [])
    print(f"  Results: {len(patents2)}")
    for p in patents2:
        print(f"  → {p.get('patent_id')} | {p.get('patent_title')}")
    if patents2:
        has_abstract = any(p.get("patent_abstract") for p in patents2)
        print(f"  Abstracts present: {'YES' if has_abstract else 'NO'}")
        print("  ✅ Test 2 PASSED")
    else:
        print("  ⚠️  No results — query may be too narrow for this corpus")
else:
    print(f"  ❌ HTTP error: {resp2.text[:300]}")

# Test 3: confirm using numeric USPTO-style IDs
print("\n[Test 3] Verify USPTO numeric ID format...")
if resp.status_code == 200:
    pid = patents[0].get("patent_id", "")
    if pid.isdigit():
        print(f"  ID '{pid}' is numeric ✅ — confirmed PatentsView/USPTO source")
    else:
        print(f"  ❌ ID '{pid}' is NOT numeric — wrong API source")
