import requests

from app.config import settings
from services.uspto_search_service import _build_patent_record

url = "https://api.uspto.gov/api/v1/patent/applications/search"

headers = {
    "X-API-KEY": settings.patentsview_api_key,
    "Accept": "application/json",
}

params = {
    "q": 'applicationMetaData.inventionTitle:"lane detection"',
    "limit": 1,
    "offset": 0,
}

print("Searching...")

r = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60,
)

print("Search HTTP:", r.status_code)

data = r.json()
raw = data["patentFileWrapperDataBag"][0]

print("Raw record received")
print("Calling _build_patent_record()...")

try:
    patent = _build_patent_record(raw)

    print("\nBUILD RETURNED")
    print("Patent:", patent)

    if patent:
        print("ID:", patent.patent_id)
        print("TITLE:", patent.patent_title)
        print("TYPE:", patent.patent_type)
        print("DATE:", patent.patent_date)
        print("ABSTRACT LENGTH:", len(patent.patent_abstract))

except BaseException as exc:
    print("\nBUILD FAILED")
    print("Exception:", type(exc).__name__)
    print("Message:", str(exc))