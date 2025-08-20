import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
import numpy as np


# Load the preprocessed data
def load_processed_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Prepare documents for vector store (convert chunks to LangChain Documents)
def prepare_documents(data):
    documents = []
    for entry in data:
        filename = entry["filename"]
        language = entry["language"]
        jurisdiction = entry["jurisdiction"]
        for i, chunk in enumerate(entry["chunks"]):
            metadata = {
                "filename": filename,
                "language": language,
                "jurisdiction": jurisdiction,
                "chunk_id": f"{filename}_chunk_{i}"
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
    return documents


# Set up the vector store with BGE-M3 embeddings
def setup_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )
    print("creating vector store.......")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("done")
    return vector_store


# Rerank retrieved results using BGE-Reranker-v2-M3
def rerank_results(query, retrieved_docs, top_k=3):
    cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs, batch_size=32)
    reranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [retrieved_docs[i] for i in reranked_indices]
    return reranked_docs


# Query the vector store with reranking
def query_vector_store(vector_store, query, top_k_initial=15, top_k_rerank=3, use_reranking=True):
    initial_results = vector_store.similarity_search(query, k=top_k_initial)
    if use_reranking:
        reranked_results = rerank_results(query, initial_results, top_k_rerank)
        return reranked_results
    return initial_results[:top_k_rerank]


# Main function to set up the vector store and test a query with reranking
def main():
    # Load preprocessed data
    processed_data = load_processed_data("processed_multilingual_legal_data.json")

    # Prepare documents for vector store
    documents = prepare_documents(processed_data)

    # Set up vector store with BGE-M3
    vector_store = setup_vector_store(documents)

    # Save the vector store for later use
    vector_store.save_local("faiss_index_bge_m3")
    print("Vector store saved to 'faiss_index_bge_m3'")

    # Example queries
    queries = [
        "What is the definition of county in U.S. law?",
        "Expliquez l'Article 1 de la Déclaration des droits de l'homme"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = query_vector_store(vector_store, query)
        for i, result in enumerate(results):
            print(f"Result {i + 1}:")
            print(f"Content: {result.page_content}")
            print(f"Metadata: {result.metadata}")
            print("-" * 50)


if __name__ == "__main__":
    main()
