embedding_manager = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(vector_store, embedding_manager)

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    response = simpleRag(query, retriever, llm)
    print(response)