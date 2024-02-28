from Retriever import KNOWLEDGE_VECTOR_DATABASE, load_knowledge_base, create_vector_database
from Reader import initialize_reader, generate_answer
from ragatouille import RAGPretrainedModel
from typing import List, Optional

# Function to answer a question using the RAG system
def answer_with_rag(
    question: str,
    llm,
    knowledge_index,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
):
    # Step 1: Retrieve documents
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs_texts = [doc.page_content for doc in relevant_docs]  # Extract text for reranking or reading

    # Step 2: Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        reranked_docs = reranker.rerank(question, relevant_docs_texts, k=num_docs_final)
        relevant_docs_texts = [doc["content"] for doc in reranked_docs]
    else:
        relevant_docs_texts = relevant_docs_texts[:num_docs_final]

    # Step 3: Build the final prompt for the reader
    context = "\nExtracted documents:\n"
    context += "\n".join([f"Document {str(i)}:::\n{doc}" for i, doc in enumerate(relevant_docs_texts)])
    final_prompt = f"Question: {question}\n{context}\nAnswer:"

    # Step 4: Generate an answer
    print("=> Generating answer...")
    answer = generate_answer(llm, final_prompt)
    return answer, relevant_docs_texts

if __name__ == "__main__":
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    question = "What is the capital of France?"

    # Load the knowledge base and create the vector database
    knowledge_base = load_knowledge_base()
    vector_database = create_vector_database(EMBEDDING_MODEL_NAME, knowledge_base)

    # Initialize the reader model
    reader_llm = initialize_reader(READER_MODEL_NAME)

    # Optionally, initialize a reranker (this example uses a placeholder for demonstration)
    reranker = None  # Example: RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Answer the question
    answer, relevant_docs = answer_with_rag(
        question, reader_llm, vector_database, reranker=reranker
    )

    print("==================================Answer==================================")
    print(answer)
    print("==================================Relevant Documents==================================")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}:\n{doc}\n")
