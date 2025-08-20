import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
# from flask import Flask, request, jsonify
import time

# Define the custom prompt template
prompt_template = """
You are a multilingual legal assistant fluent in English and French.
 Based on the provided documents, answer the user's question accurately and concisely. 
 Use only the information from the documents and cite the source (e.g., filename) for each fact.
  If the answer is not in the documents, state 'I do not have sufficient information to answer this question.
  ' Respond in the same language as the query.

Question: {question}
Documents: {context}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Set global timeout for huggingface_hub downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# Load the pre-existing vector store with bge-m3 embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})

db = FAISS.load_local(
    "Multilingual_Legal_Aid_Chatbot/models/faiss_index_bge_m3", embeddings,
    allow_dangerous_deserialization=True)
vector_store = db.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance (MMR)
    search_kwargs={"k": 3, "lambda_mult": 0.1}
)

# Load the Qwen-1.5-7B-Chat model
model_path = "Multilingual_Legal_Aid_Chatbot/models/Qwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)

# Create a pipeline for text generation
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    batch_size=1,
    device_map="cpu",
    do_sample=True,
    top_k=50,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generator)

# Load the reranker model
# model_path = "Multilingual_Legal_Aid_Chatbot/models/bge-reranker-v2-m3"

# reranker = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)

# Set up the RetrievalQA chain with custom retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Interactive chatbot loop
def chatbot1(query):
    print("Welcome to the Multilingual Legal Aid Chatbot! (Type 'exit' to quit)")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            print("Goodbye!")
        break
        if not query.strip():  # Skip empty input
            print("Please enter a question.")
        continue
    start_time = time.time()
    result = qa_chain({"query": query})
    answer = result["result"].strip()
    if answer.startswith("You are a multilingual legal assistant"):
        answer = answer[answer.index("Answer:") + len("Answer:"):].strip()
    sources = "\n".join([f"{i}. {doc.page_content} (Source: {doc.metadata['filename']})" for i, doc in
                         enumerate(result["source_documents"], 1)])
    end_time = time.time()
    print("\nAnswer:", answer)
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    print("Source Documents:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"{i}. {doc.page_content} (Source: {doc.metadata['filename']})")
    print("-" * 50)
    return answer, sources


def chatbot(query):
    result = qa_chain({"query": query})
    answer = result["result"].strip()
    if answer.startswith("You are a multilingual legal assistant"):
        answer = answer[answer.index("Answer:") + len("Answer:"):].strip()
    sources = [f"{i}. {doc.page_content} (Source: {doc.metadata['filename']})" for i, doc in
               enumerate(result["source_documents"], 1)]
    return answer, "\n".join(sources)


interface = gr.Interface(fn=chatbot, inputs="text", outputs=["text", "text"], title="Multilingual Legal Aid Chatbot",
                         description="Ask legal questions in English or French.")
interface.launch()

# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)


# @app.route('/ask', methods=['POST'])
# def ask():
#    query = request.json.get('query')
#    start_time = time.time()
#    result = qa_chain({"query": query})
#    print("done")
#    answer = result["result"].strip()
#    if answer.startswith("You are a multilingual legal assistant"):
#        answer = answer[answer.index("Answer:") + len("Answer:"):].strip()
#    sources = [f"{i}. {doc.page_content} (Source: {doc.metadata['filename']})" for i, doc in
#               enumerate(result["source_documents"], 1)]
#    end_time = time.time()
#    return jsonify({"answer": answer, "sources": "\n".join(sources), "time": end_time - start_time})


# if __name__ == "__main__":
#    app.run(debug=True, port=5000, use_reloader=False)

