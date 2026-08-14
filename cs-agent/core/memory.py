"""
对话记忆管理 — 短期记忆（当前对话）+ 长期记忆（历史摘要）
"""
from dataclasses import dataclass, field
from datetime import datetime
from llm.base import LLMMessage, MessageRole


@dataclass
class ConversationTurn:
    """一轮对话（用户消息 + 助手回复）"""
    user_message: str
    assistant_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """
    对话记忆管理器

    短期记忆: 当前对话的完整历史（最近 N 轮）
    长期记忆: 超出窗口的历史对话的摘要
    """

    def __init__(self, max_turns: int = 20, max_tokens: int = 8000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: list[ConversationTurn] = []
        self.summary: str = ""
        self.system_prompt: str = ""

    def set_system_prompt(self, prompt: str):
        """设置 system prompt"""
        self.system_prompt = prompt

    def add_turn(self, user_message: str, assistant_message: str, metadata: dict | None = None):
        """添加一轮对话"""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata or {},
        )
        self.turns.append(turn)

        # 如果超出窗口，压缩旧对话
        if len(self.turns) > self.max_turns:
            self._compress_old_turns()

    def get_messages(self, include_system: bool = True) -> list[LLMMessage]:
        """
        获取当前对话历史（LLM 格式）

        包含: system prompt + 长期记忆摘要 + 最近 N 轮对话
        """
        messages = []

        # System prompt
        if include_system and self.system_prompt:
            system_content = self.system_prompt
            if self.summary:
                system_content += f"\n\n## 历史对话摘要\n{self.summary}"
            messages.append(LLMMessage(role=MessageRole.SYSTEM, content=system_content))

        # 最近的对话轮次
        for turn in self.turns:
            messages.append(LLMMessage(role=MessageRole.USER, content=turn.user_message))
            messages.append(LLMMessage(role=MessageRole.ASSISTANT, content=turn.assistant_message))

        return messages

    def get_context_window_size(self) -> int:
        """估算当前上下文窗口的 token 数（粗略估计）"""
        total_chars = len(self.system_prompt) + len(self.summary)
        for turn in self.turns:
            total_chars += len(turn.user_message) + len(turn.assistant_message)
        # 粗略估计：1 个中文字符 ≈ 2 tokens
        return total_chars * 2

    def clear(self):
        """清空对话历史"""
        self.turns.clear()
        self.summary = ""

    def _compress_old_turns(self):
        """压缩超出窗口的旧对话为摘要"""
        # 只压缩超出 max_turns 的旧对话，保留最近 max_turns 轮
        excess = max(0, len(self.turns) - self.max_turns)
        if excess == 0:
            return

        # 一次最多压缩前 5 轮，避免单次操作过重
        compress_count = min(excess, 5)
        old_turns = self.turns[:compress_count]
        old_summary_parts = []
        for turn in old_turns:
            old_summary_parts.append(f"用户: {turn.user_message[:100]}...")
            old_summary_parts.append(f"助手: {turn.assistant_message[:100]}...")

        new_summary = "\n".join(old_summary_parts)
        if self.summary:
            self.summary = f"{self.summary}\n\n{new_summary}"
        else:
            self.summary = new_summary

        # 只移除被压缩的轮次
        self.turns = self.turns[compress_count:]

    def export_history(self) -> list[dict]:
        """导出对话历史（用于持久化存储）"""
        return [
            {
                "user": turn.user_message,
                "assistant": turn.assistant_message,
                "timestamp": turn.timestamp.isoformat(),
                "metadata": turn.metadata,
            }
            for turn in self.turns
        ]

    def import_history(self, history: list[dict]):
        """导入对话历史"""
        self.turns.clear()
        for item in history:
            self.turns.append(ConversationTurn(
                user_message=item["user"],
                assistant_message=item["assistant"],
                metadata=item.get("metadata", {}),
            ))
