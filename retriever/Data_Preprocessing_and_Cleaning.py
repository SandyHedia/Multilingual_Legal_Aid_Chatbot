import os
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# extract text from PDF using PyPDFLoader
def extract_pdf_text(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


# clean text (OCR errors for PDFs, general cleaning for all)
def clean_text(text):
    text = re.sub(r'(\b\w+\b)(\s*\1){2,}', r'\1', text)  # Remove repetitive patterns
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
    text = re.sub(r'\b\d{1,3}\b\s*(?:/|-)\s*\d{1,3}\b', '', text)  # Remove page numbers
    return text


# extract preamble and articles from French legal text
def extract_preamble_and_articles(text):
    articles = []
    article_pattern = r'(Article\s+(?:premier|\d+[er]?(?:-\d+)?))\s*[:\.]?\s*(.*?)(?=(Article\s+(?:premier|\d+[er]?(?:-\d+)?))|$)'
    matches = re.findall(article_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        article_num, content, _ = match
        articles.append({
            "number": article_num.strip(),
            "content": content.strip()
        })

    preamble_match = re.match(r'^(.*?)(?=Article\s+(?:premier|\d+[er]?(?:-\d+)?))', text, re.DOTALL | re.IGNORECASE)
    preamble = preamble_match.group(1).strip() if preamble_match else ""

    return preamble, articles


# extract sections from English JSON data (treat titles as article numbers)
def extract_english_sections(data):
    articles = []
    for entry in data:
        articles.append({
            "number": entry["title"].strip(),
            "content": clean_text(entry["content"])
        })
    return articles


# split text into chunks, prioritizing article boundaries
def split_text(text, articles, chunk_size=1500, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nArticle", "\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    if articles:
        for article in articles:
            article_text = article["content"]
            if len(article_text) > chunk_size:
                article_chunks = text_splitter.split_text(article_text)
                chunks.extend(article_chunks)
            else:
                chunks.append(article_text)
    else:
        chunks = text_splitter.split_text(text)

    return chunks


# process a PDF file (French data)
def process_pdf(file_path):
    text = extract_pdf_text(file_path)
    cleaned_text = clean_text(text)
    preamble, articles = extract_preamble_and_articles(cleaned_text)
    chunks = split_text(cleaned_text, articles)

    metadata = {
        "filename": os.path.basename(file_path),
        "language": "fr",
        "jurisdiction": "fr",
        "preamble": preamble,
        "articles": articles,
        "full_text": cleaned_text,
        "chunks": chunks
    }

    return metadata


# process a TXT file (French data)
def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    cleaned_text = clean_text(text)
    preamble, articles = extract_preamble_and_articles(cleaned_text)
    chunks = split_text(cleaned_text, articles)

    metadata = {
        "filename": os.path.basename(file_path),
        "language": "fr",
        "jurisdiction": "fr",
        "preamble": preamble,
        "articles": articles,
        "full_text": cleaned_text,
        "chunks": chunks
    }

    return metadata


# process a JSON file (English data)
def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Extract sections as articles
    articles = extract_english_sections(data)
    full_text = " ".join([article["content"] for article in articles])
    chunks = split_text(full_text, articles)

    metadata = {
        "filename": os.path.basename(file_path),
        "language": "en",
        "jurisdiction": "us",
        "preamble": "",
        "articles": articles,
        "full_text": full_text,
        "chunks": chunks
    }

    return metadata


# Main function to process all files
def preprocess_multilingual_legal_data(input_dir, output_json):
    processed_data = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        print(f"Processing  {filename}")
        if filename.endswith(".pdf"):
            data = process_pdf(file_path)
            processed_data.append(data)
        elif filename.endswith(".txt"):
            data = process_txt(file_path)
            processed_data.append(data)
        elif filename.endswith(".json"):
            data = process_json(file_path)
            processed_data.append(data)

    # Save processed data to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processed data saved to {output_json}")


if __name__ == "__main__":
    input_directory = "C:/sato/projects/Multilingual_Legal_Aid_Chatbot/data/fr&en_data"
    output_file = "processed_multilingual_legal_data.json"
    preprocess_multilingual_legal_data(input_directory, output_file)
