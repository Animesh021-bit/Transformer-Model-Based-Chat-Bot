import chromadb
from sentence_transformers import SentenceTransformer


class VectorDatabase:

    def __init__(self):

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add_documents(self, documents):

        embeddings = self.embedding_model.encode(
            documents
        ).tolist()

        ids = [
            f"doc_{i}"
            for i in range(len(documents))
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

        print("Documents added to VectorDB.")

    def search(self, query, n_results=3):

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results