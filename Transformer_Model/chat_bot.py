from Vector_db import VectorDatabase

db = VectorDatabase()

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    results = db.search(question, n_results=3)

    context = "\n".join(
        results["documents"][0]
    )

    print("\nAI:", context)