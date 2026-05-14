import requests
from requests.auth import HTTPBasicAuth
from lxml import etree

# ==========================================
# CONFIG
# ==========================================


CONSUMER_KEY = "wAZaXr2azN7S0y5NFZLuTlUg1HM8At13t6bAsEvYTYZ6DG8e"
CONSUMER_SECRET = "0M3B4sjE3Xv1GAEzHa8rj4IwCh1YCl7woctGlbj11hjGrX29HDjA89J2ZfYO9X7m"

SEARCH_QUERY = "infrared lane detection"

# ==========================================
# STEP 1 — AUTHENTICATE
# ==========================================

auth_url = "https://ops.epo.org/3.2/auth/accesstoken"

print("\n[1] Requesting access token...")

auth_response = requests.post(
    auth_url,
    auth=HTTPBasicAuth(CONSUMER_KEY, CONSUMER_SECRET),
    data={"grant_type": "client_credentials"},
)

print("Status:", auth_response.status_code)

if auth_response.status_code != 200:
    print(auth_response.text)
    exit()

token = auth_response.json()["access_token"]

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/xml",
}

print("Access token received.\n")

# ==========================================
# STEP 2 — SEARCH PATENTS
# ==========================================

search_url = (
    "https://ops.epo.org/3.2/rest-services/"
    f"published-data/search/biblio?q={SEARCH_QUERY}"
)

print("[2] Searching patents...")
print(search_url)

search_response = requests.get(search_url, headers=headers)

print("Search status:", search_response.status_code)

if search_response.status_code != 200:
    print(search_response.text)
    exit()

print("Search successful.\n")

# ==========================================
# STEP 3 — PARSE SEARCH RESULTS
# ==========================================

root = etree.fromstring(search_response.content)

namespaces = {
    "ops": "http://ops.epo.org",
    "exchange": "http://www.epo.org/exchange"
}

docs = root.xpath("//exchange:exchange-document", namespaces=namespaces)

print(f"[3] Found {len(docs)} patents\n")

if not docs:
    print("No patents found.")
    exit()

# ==========================================
# STEP 4 — READ PATENT IDS
# ==========================================

patent_ids = []

# ==========================================
# STEP 4 — READ PATENT IDS
# ==========================================

patent_ids = []

for doc in docs[:5]:

    country = doc.get("country", "")
    doc_number = doc.get("doc-number", "")
    kind = doc.get("kind", "")

    # IMPORTANT: OPS epodoc format
    patent_id = f"{country}.{doc_number}.{kind}"

    patent_ids.append(patent_id)

    print("Patent ID:", patent_id)

print()

# ==========================================
# STEP 5 — FETCH FULL PATENT BIBLIO
# ==========================================

for patent_id in patent_ids:

    print("=" * 80)
    print("FETCHING:", patent_id)

    biblio_url = (
        "https://ops.epo.org/3.2/rest-services/"
        f"published-data/publication/epodoc/{patent_id}/biblio"
    )

    response = requests.get(biblio_url, headers=headers)

    print("Biblio status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        continue

    try:
        root = etree.fromstring(response.content)

        # ------------------------------
        # TITLE
        # ------------------------------
        title_nodes = root.xpath(
            "//exchange:invention-title",
            namespaces=namespaces
        )

        title = (
            title_nodes[0].text.strip()
            if title_nodes and title_nodes[0].text
            else "NO TITLE FOUND"
        )

        # ------------------------------
        # ABSTRACT
        # ------------------------------
        abstract_nodes = root.xpath(
            "//exchange:abstract//exchange:p",
            namespaces=namespaces
        )

        abstract_parts = []

        for node in abstract_nodes:
            if node.text:
                abstract_parts.append(node.text.strip())

        abstract = " ".join(abstract_parts)

        if not abstract:
            abstract = "NO ABSTRACT FOUND"

        # ------------------------------
        # PRINT RESULTS
        # ------------------------------
        print("\nTITLE:")
        print(title)

        print("\nABSTRACT:")
        print(abstract[:1000])

        print("\nTEXT LENGTH:", len(title + abstract))

    except Exception as e:
        print("PARSE ERROR:", str(e))