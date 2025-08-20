import requests
from bs4 import BeautifulSoup
import json
import re

# Base URL template
BASE_URL = "https://www.govinfo.gov/content/pkg/USCODE-2023-title{}/html/USCODE-2023-title{}.htm"


# Function to clean text
def clean_text(text):
    return ' '.join(text.strip().split())


# Function to scrape a single title
def scrape_title(title_number):
    url = BASE_URL.format(title_number, title_number)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        sections = []
        # Find all section elements
        section_elements = soup.find_all(['div', 'p'], id=re.compile(r'sec\d+.*'))

        if not section_elements:
            # Fallback: Look for statutory body paragraphs
            section_elements = soup.find_all('p', class_=r'statutory-body')

        for element in section_elements:
            try:
                # Find the section title
                title = ""
                prev_sibling = element.find_previous(['h3', 'h4', 'p'], text=re.compile(r'§\s*\d+.*'))
                if prev_sibling:
                    title = clean_text(prev_sibling.get_text())
                else:
                    # Fallback: Use ID or nearby text
                    if element.get('id'):
                        title = clean_text(element.get('id').replace('sec', '§ '))

                # Get full content, including subsequent paragraphs or lists
                content_parts = []
                current_element = element
                while current_element:
                    # Check if the element has a class starting with 'statutory-body-' or is a relevant tag
                    element_classes = current_element.get('class', [])
                    is_statutory_body = any(re.match(r'statutory-body(-\d+em)?', cls) for cls in element_classes)
                    is_relevant_tag = current_element.name in ['p', 'ul', 'li']

                    if is_statutory_body or is_relevant_tag:
                        content_parts.append(clean_text(current_element.get_text()))
                    else:
                        break  # Stop if we hit a non-relevant element

                    current_element = current_element.find_next_sibling()

                content = ' '.join(content_parts)

                if title and content:
                    sections.append({
                        "title": title,
                        "content": content,
                        "lang": "en"
                    })
            except Exception as e:
                print(f"Error processing section in title {title_number}: {e}")
                continue

        return sections

    except requests.RequestException as e:
        print(f"Failed to fetch title {title_number}: {e}")
        return []


def main():
    all_sections = []

    # Scrape titles 1 to 54
    for title_number in range(1, 55):
        print(f"Scraping Title {title_number}...")
        sections = scrape_title(title_number)
        all_sections.extend(sections)

    # Save to JSON file
    with open('../data/fr&en_data/uscode_titles.json', 'w', encoding='utf-8') as f:
        json.dump(all_sections, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_sections)} sections to uscode_titles.json")


if __name__ == "__main__":
    main()
