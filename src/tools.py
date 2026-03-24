from langchain.tools import tool
from src.vectorstore import similarity_search
from src.memory import search_memory, save_memory


@tool
def retrieve_docs(query: str) -> str:
    """Retrieve relevant local documents for a user query."""
    docs = similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(docs)


@tool
def lookup_memory(query: str) -> str:
    """Search saved long-term memory for useful prior facts."""
    hits = search_memory(query)
    if not hits:
        return "No relevant memory found."
    return "\n".join(hits)


@tool
def write_memory(content: str) -> str:
    """Write useful facts or user preferences into memory."""
    save_memory(content)
    return f"Saved to memory: {content}"


TOOLS = [retrieve_docs, lookup_memory, write_memory]