import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.law.cornell.edu/uscode/text"
visited = set()
results = []

# Setup requests session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}


# get soup with delay and retry
def get_soup(url):
    try:
        time.sleep(0.5)  # polite delay
        print(f"Visiting: {url}")
        response = session.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f" Error fetching {url}: {e}")
        return None


# Extract final text data if page contains it
def extract_final_data(soup):
    content_div = soup.select_one("div.section > div.content")
    if content_div:
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"
        content = content_div.get_text(separator="\n", strip=True)
        return {
            "title": title,
            "content": content,
            "lang": "en"
        }
    return None


# Get all nested links from <ol class="list-unstyled">
def get_links(soup):
    return [urljoin(BASE_URL, a['href']) for ol in soup.find_all("ol", class_="list-unstyled")
            for a in ol.find_all("a", href=True)]


# Recursive scraping function
def scrape_recursive(url, depth=0, max_depth=15):
    if url in visited or depth > max_depth:
        return

    visited.add(url)
    soup = get_soup(url)
    if not soup:
        return

    # Check if final page with actual law text
    data = extract_final_data(soup)
    if data:
        results.append(data)
        return

    # Otherwise, recursively follow links
    links = get_links(soup)
    for link in links:
        scrape_recursive(link, depth + 1, max_depth)


# START SCRAPING
start_url = "https://www.law.cornell.edu/uscode/text"
scrape_recursive(start_url)

# SAVE RESULTS TO JSON
with open("usc_code_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n Finished! Saved {len(results)} entries to 'usc_code_data.json'")
