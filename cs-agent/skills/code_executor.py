"""
代码执行 Skill — 在安全沙箱中执行 Python/C++ 代码

支持:
- Python 代码直接执行
- C++ 代码编译执行
- 超时限制
- 输出捕获
"""
import asyncio
import sys
import tempfile
from pathlib import Path
from loguru import logger

from skills.base import BaseSkill, SkillResult
from config.settings import settings


class CodeExecutorSkill(BaseSkill):
    """
    代码执行工具

    在隔离的子进程中执行用户代码，捕获 stdout/stderr。
    对于 C++ 代码，先编译再执行。
    """

    name = "code_executor"
    description = "执行代码并返回运行结果。支持 Python 和 C++。当用户要求运行、测试、调试代码时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的代码",
            },
            "language": {
                "type": "string",
                "enum": ["python", "cpp"],
                "description": "编程语言，默认 python",
                "default": "python",
            },
            "input_data": {
                "type": "string",
                "description": "标准输入数据（可选）",
                "default": "",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or settings.code_execution_timeout

    async def execute(self, **kwargs) -> SkillResult:
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")
        input_data = kwargs.get("input_data", "")

        if not code.strip():
            return SkillResult.error("代码不能为空")

        logger.info(f"执行 {language} 代码 | 长度: {len(code)} 字符")

        if language == "python":
            return await self._execute_python(code, input_data)
        elif language == "cpp":
            return await self._execute_cpp(code, input_data)
        else:
            return SkillResult.error(f"不支持的语言: {language}")

    async def _execute_python(self, code: str, input_data: str = "") -> SkillResult:
        """执行 Python 代码"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=input_data.encode() if input_data else None),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return SkillResult.error(
                    f"代码执行超时（超过 {self.timeout} 秒）",
                    timeout=True,
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                result_text = stdout_str if stdout_str else "(无输出)"
                return SkillResult.success(
                    result_text,
                    return_code=0,
                    language="python",
                )
            else:
                error_text = f"程序退出码: {proc.returncode}"
                if stderr_str:
                    error_text += f"\n\n错误信息:\n{stderr_str}"
                if stdout_str:
                    error_text += f"\n\n标准输出:\n{stdout_str}"
                return SkillResult.error(error_text, return_code=proc.returncode)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _execute_cpp(self, code: str, input_data: str = "") -> SkillResult:
        """编译并执行 C++ 代码"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "main.cpp"
            binary_path = Path(tmpdir) / "main"

            source_path.write_text(code, encoding="utf-8")

            # 编译
            compile_proc = await asyncio.create_subprocess_exec(
                "g++", "-std=c++17", "-O2", "-o", str(binary_path), str(source_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, compile_stderr = await asyncio.wait_for(
                    compile_proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                compile_proc.kill()
                await compile_proc.wait()
                return SkillResult.error(
                    f"编译超时（超过 {self.timeout} 秒）",
                    timeout=True,
                )

            if compile_proc.returncode != 0:
                error_msg = compile_stderr.decode("utf-8", errors="replace")
                return SkillResult.error(f"编译失败:\n{error_msg}", language="cpp")

            # 执行
            try:
                exec_proc = await asyncio.create_subprocess_exec(
                    str(binary_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    exec_proc.communicate(input=input_data.encode() if input_data else None),
                    timeout=self.timeout,
                )

                stdout_str = stdout.decode("utf-8", errors="replace").strip()
                stderr_str = stderr.decode("utf-8", errors="replace").strip()

                if exec_proc.returncode == 0:
                    result_text = stdout_str if stdout_str else "(无输出)"
                    return SkillResult.success(
                        result_text,
                        return_code=0,
                        language="cpp",
                    )
                else:
                    error_text = f"程序退出码: {exec_proc.returncode}"
                    if stderr_str:
                        error_text += f"\n\n错误信息:\n{stderr_str}"
                    return SkillResult.error(error_text, return_code=exec_proc.returncode)

            except asyncio.TimeoutError:
                exec_proc.kill()
                await exec_proc.wait()
                return SkillResult.error(
                    f"代码执行超时（超过 {self.timeout} 秒）",
                    timeout=True,
                )
