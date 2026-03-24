from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState
from src.prompts import SYSTEM_PROMPT
from src.llm import get_chat_model
from src.vectorstore import similarity_search
from src.memory import search_memory, save_memory


llm = get_chat_model()


def memory_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    memory_hits = search_memory(query)
    state["memory_hits"] = memory_hits
    return state


def retrieve_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    docs = similarity_search(query, k=3)
    state["retrieved_docs"] = docs
    return state


def answer_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    memory_text = "\n".join(state.get("memory_hits", []))
    docs_text = "\n\n".join(state.get("retrieved_docs", []))

    prompt = f"""
User question:
{query}

Relevant memory:
{memory_text if memory_text else "None"}

Retrieved documents:
{docs_text if docs_text else "None"}

Please answer the user clearly. If the information is insufficient, say so explicitly.
Then on a new line output:
MEMORY_CANDIDATE: <text or NONE>
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    text = response.content
    state["tool_outputs"] = [text]

    if "MEMORY_CANDIDATE:" in text:
        answer, candidate = text.split("MEMORY_CANDIDATE:", 1)
        state["final_answer"] = answer.strip()
        candidate = candidate.strip()
        if candidate and candidate.upper() != "NONE":
            save_memory(candidate)
    else:
        state["final_answer"] = text.strip()

    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("memory", memory_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()