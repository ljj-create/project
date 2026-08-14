"""
Skill 抽象基类 — 所有工具/技能继承此类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SkillStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class SkillResult:
    """Skill 执行结果"""
    status: SkillStatus
    content: str  # 结果内容（文本）
    metadata: dict = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **metadata) -> "SkillResult":
        return cls(status=SkillStatus.SUCCESS, content=content, metadata=metadata)

    @classmethod
    def error(cls, content: str, **metadata) -> "SkillResult":
        return cls(status=SkillStatus.ERROR, content=content, metadata=metadata)

    @property
    def is_success(self) -> bool:
        return self.status == SkillStatus.SUCCESS


class BaseSkill(ABC):
    """
    Skill 抽象基类

    每个 Skill 需要定义:
    - name: 工具名称（英文，用于调用）
    - description: 工具描述（供 LLM 理解何时使用）
    - parameters: 参数 JSON Schema（供 LLM 生成调用参数）
    - execute(): 执行逻辑
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        """
        执行 Skill

        Args:
            **kwargs: 参数

        Returns:
            SkillResult 执行结果
        """
        ...

    def get_tool_definition(self) -> dict:
        """获取 OpenAI Function Calling 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
