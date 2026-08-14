"""Unified agent domain types.

These types are intentionally provider/tool agnostic. The LLM layer and tool
layer are adapted to these types at the edges, so the agent loop only depends
on a small, stable contract.
"""

from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(str, Enum):
    """Status of an agent run."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


@dataclass
class AgentStep:
    """One observable step inside an agent run."""

    name: str
    args: dict = field(default_factory=dict)
    status: AgentStatus = AgentStatus.SUCCESS
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": self.args,
            "status": self.status.value,
            "output": self.output,
        }


@dataclass
class AgentResult:
    """Final result returned by the agent loop."""

    status: AgentStatus
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)

    @classmethod
    def success(cls, content: str, **kwargs) -> "AgentResult":
        return cls(status=AgentStatus.SUCCESS, content=content, **kwargs)

    @classmethod
    def error(cls, content: str, **kwargs) -> "AgentResult":
        return cls(status=AgentStatus.ERROR, content=content, **kwargs)
