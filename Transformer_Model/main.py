from Vector_db import VectorDatabase


# Create VectorDB
db = VectorDatabase()


# Documents
documents = [
    "Python is a popular programming language.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks.",
    "Transformers are powerful models for NLP.",
    "Vector databases store embeddings.",
    "ChromaDB is a vector database."
]


# Add documents
db.add_documents(documents)


# User query
query = "What is deep learning?"


# Search VectorDB
results = db.search(
    query,
    n_results=3
)


print("\nQuery:")
print(query)

print("\nRelevant Documents:")

for document in results["documents"][0]:
    print("-", document)