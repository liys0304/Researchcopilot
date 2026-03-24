from src.vectorstore import build_vectorstore
from src.graph import build_graph

def main():
    print("== Build / Load vector DB ==")
    try:
        build_vectorstore()
    except Exception as e:
        print(f"Vectorstore build skipped or failed: {e}")

    app = build_graph()

    print("Research Copilot started. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        state = {
            "user_query": query,
            "retrieved_docs": [],
            "memory_hits": [],
            "tool_outputs": [],
            "final_answer": None,
        }

        result = app.invoke(state)
        print("\nAssistant:")
        print(result["final_answer"])
        print()

if __name__ == "__main__":
    main()