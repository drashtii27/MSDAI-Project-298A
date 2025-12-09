"""
Fetch job postings from Adzuna API and store in bronze layer.
"""

import os
import json
import time
import datetime as dt
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

# Configuration
APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")
MAX_PAGES = int(os.getenv("ADZUNA_MAX_PAGES", "40"))
RESULTS_PER_PAGE = int(os.getenv("ADZUNA_RESULTS_PER_PAGE", "50"))
QUERY = os.getenv(
    "ADZUNA_QUERY",
    'software engineer, data engineer, machine learning, python developer, backend engineer'
)
COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")
WHERE = os.getenv("ADZUNA_WHERE", "United States")
SORT_BY = os.getenv("ADZUNA_SORT", "date")

BASE_URL = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/{{page}}"
HEADERS = {"User-Agent": "career-mentor/2.0"}

# Setup session with retries
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def fetch_page(page: int) -> dict:
    """Fetch a single page of results from Adzuna API."""
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "what_or": QUERY,
        "where": WHERE,
        "results_per_page": RESULTS_PER_PAGE,
        "sort_by": SORT_BY,
        "content-type": "application/json",
    }
    
    url = BASE_URL.format(page=page)
    
    try:
        response = session.get(url, params=params, headers=HEADERS, timeout=30)
        
        if response.status_code >= 400:
            print(f"⚠ HTTP {response.status_code} on page {page}")
            return {}
        
        return response.json()
    
    except Exception as e:
        print(f"⚠ Error fetching page {page}: {e}")
        return {}


def main():
    """Main job fetching pipeline."""
    assert APP_ID and APP_KEY, "Set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env"
    
    # Setup output directory
    date_str = dt.date.today().isoformat()
    out_dir = Path(f"data/bronze/jobs/date={date_str}")
    ensure_dir(out_dir)
    out_path = out_dir / "adzuna.jsonl"
    
    print(f"Fetching jobs from Adzuna...")
    print(f"Query: {QUERY}")
    print(f"Max pages: {MAX_PAGES}")
    print(f"Output: {out_path}")
    
    seen_ids = set()
    saved_count = 0
    
    with open(out_path, "w", encoding="utf-8") as f:
        for page in range(1, MAX_PAGES + 1):
            data = fetch_page(page)
            
            if not data:
                time.sleep(1.0)
                continue
            
            if page == 1:
                print(f"Total count: {data.get('count', 'N/A')}")
            
            results = data.get("results", [])
            print(f"Page {page}: {len(results)} results")
            
            if not results:
                break
            
            for job in results:
                job_id = str(job.get("id"))
                if not job_id or job_id in seen_ids:
                    continue
                
                seen_ids.add(job_id)
                
                record = {
                    "id": job_id,
                    "title": job.get("title"),
                    "company": job.get("company", {}).get("display_name"),
                    "location": job.get("location", {}).get("display_name"),
                    "latitude": job.get("latitude"),
                    "longitude": job.get("longitude"),
                    "description": job.get("description"),
                    "category": job.get("category", {}).get("label"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "created": job.get("created"),
                    "redirect_url": job.get("redirect_url"),
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_count += 1
            
            time.sleep(0.5)  # Rate limiting
    
    print(f"\n✓ Saved {saved_count} unique jobs to {out_path}")


if __name__ == "__main__":
    main()