"""
Skill 注册表 — 管理和调度所有可用的 Skill
"""
from loguru import logger
from skills.base import BaseSkill, SkillResult


class SkillRegistry:
    """
    Skill 注册表

    功能:
    - 注册/注销 Skill
    - 根据名称查找 Skill
    - 执行指定 Skill
    - 获取所有工具定义（供 LLM Function Calling）
    """

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册一个 Skill"""
        if not skill.name:
            raise ValueError(f"Skill 必须有 name 属性: {skill}")
        self._skills[skill.name] = skill
        logger.info(f"注册 Skill: {skill.name} — {skill.description}")

    def unregister(self, name: str):
        """注销一个 Skill"""
        if name in self._skills:
            del self._skills[name]
            logger.info(f"注销 Skill: {name}")

    def has_skill(self, name: str) -> bool:
        """检查是否存在指定 Skill"""
        return name in self._skills

    def get_skill(self, name: str) -> BaseSkill | None:
        """获取指定 Skill"""
        return self._skills.get(name)

    async def execute(self, name: str, **kwargs) -> SkillResult:
        """
        执行指定 Skill

        Args:
            name: Skill 名称
            **kwargs: 传递给 Skill 的参数

        Returns:
            SkillResult 执行结果
        """
        skill = self._skills.get(name)
        if not skill:
            return SkillResult.error(f"未知的工具: {name}")

        logger.info(f"执行 Skill: {name} | 参数: {kwargs}")
        try:
            result = await skill.execute(**kwargs)
            logger.info(f"Skill {name} 执行完成 | 状态: {result.status}")
            return result
        except Exception as e:
            logger.error(f"Skill {name} 执行异常: {e}")
            return SkillResult.error(f"工具执行失败: {str(e)}")

    def get_tool_definitions(self) -> list[dict]:
        """获取所有 Skill 的工具定义（OpenAI Function Calling 格式）"""
        return [skill.get_tool_definition() for skill in self._skills.values()]

    def list_skills(self) -> list[dict]:
        """列出所有已注册的 Skill"""
        return [
            {"name": skill.name, "description": skill.description}
            for skill in self._skills.values()
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
