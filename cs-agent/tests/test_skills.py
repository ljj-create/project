"""
Skill 系统测试
"""
import pytest
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.base import SkillResult, SkillStatus
from skills.skill_registry import SkillRegistry
from skills.code_executor import CodeExecutorSkill


class TestSkillRegistry:
    """Skill 注册表测试"""

    def test_register_and_list(self):
        registry = SkillRegistry()
        skill = CodeExecutorSkill()
        registry.register(skill)

        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "code_executor"

    def test_has_skill(self):
        registry = SkillRegistry()
        registry.register(CodeExecutorSkill())
        assert registry.has_skill("code_executor")
        assert not registry.has_skill("nonexistent")

    def test_execute_unknown(self):
        registry = SkillRegistry()
        result = asyncio.run(registry.execute("unknown"))
        assert result.status == SkillStatus.ERROR


class TestCodeExecutor:
    """代码执行测试"""

    @pytest.mark.asyncio
    async def test_python_hello(self):
        executor = CodeExecutorSkill()
        result = await executor.execute(code="print('Hello, World!')")
        assert result.is_success
        assert "Hello, World!" in result.content

    @pytest.mark.asyncio
    async def test_python_error(self):
        executor = CodeExecutorSkill()
        result = await executor.execute(code="print(1/0)")
        assert not result.is_success
        assert "ZeroDivisionError" in result.content or "退出码" in result.content

    @pytest.mark.asyncio
    async def test_empty_code(self):
        executor = CodeExecutorSkill()
        result = await executor.execute(code="")
        assert not result.is_success

    @pytest.mark.asyncio
    async def test_python_algorithm(self):
        executor = CodeExecutorSkill()
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

for i in range(10):
    print(fibonacci(i), end=' ')
"""
        result = await executor.execute(code=code.strip())
        assert result.is_success
        assert "0 1 1 2 3 5 8 13 21 34" in result.content


class TestSkillResult:
    """SkillResult 测试"""

    def test_success(self):
        result = SkillResult.success("ok", key="value")
        assert result.is_success
        assert result.content == "ok"
        assert result.metadata["key"] == "value"

    def test_error(self):
        result = SkillResult.error("fail")
        assert not result.is_success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
