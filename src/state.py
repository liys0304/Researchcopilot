from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_query: str
    retrieved_docs: List[str]
    memory_hits: List[str]
    tool_outputs: List[str]
    final_answer: Optional[str]