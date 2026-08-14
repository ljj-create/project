"""
CS-Agent 桌面端启动器

以桌面窗口方式运行 CS-Agent（pywebview + 内置中文聊天界面）。

后端服务由独立子进程运行（python scripts/run.py），与窗口进程隔离：
- 窗口进程崩溃不影响后端；后端崩溃会显示错误而非无声失败
- 已有一个服务在运行时（如单独启动过 run.py），直接复用，不会重复占用端口
- 窗口关闭时，只有本实例启动的后端才会被一并关闭

用法:
    python desktop_app.py              # 启动桌面端
    python desktop_app.py --port 8000  # 指定端口
"""
import sys
import time
import argparse
import subprocess
import urllib.request
from pathlib import Path

from loguru import logger

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WINDOW_TITLE = "CS-Agent 智能体"
HEALTH_URL = "http://127.0.0.1:{port}/api/health"
INSTANCE_COUNTER = ROOT / "storage" / ".desktop_instances"


def _read_count() -> int:
    try:
        return int(INSTANCE_COUNTER.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0


def _write_count(n: int) -> None:
    INSTANCE_COUNTER.parent.mkdir(parents=True, exist_ok=True)
    INSTANCE_COUNTER.write_text(str(n), encoding="utf-8")


def server_alive(port: int) -> bool:
    """检测本地服务是否已在运行"""
    try:
        with urllib.request.urlopen(HEALTH_URL.format(port=port), timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="CS-Agent 桌面端")
    parser.add_argument("--port", type=int, default=8000, help="服务端口（默认 8000）")
    args = parser.parse_args()

    port = args.port
    backend_proc = None
    _write_count(_read_count() + 1)

    try:
        if server_alive(port):
            logger.info(f"检测到服务已在 http://127.0.0.1:{port} 运行，直接复用")
        else:
            logger.info(f"启动后端服务 http://127.0.0.1:{port} ...")
            try:
                backend_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/run.py",
                        "--web-only",
                        "--port",
                        str(port),
                    ],
                    cwd=str(ROOT),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception as e:
                logger.error(f"后端进程启动失败: {e}")
                sys.exit(1)

            # 等待服务就绪
            for _ in range(120):
                if server_alive(port):
                    break
                if backend_proc.poll() is not None:
                    logger.error("后端进程异常退出，请检查 storage/logs 下的日志")
                    sys.exit(1)
                time.sleep(0.5)
            if not server_alive(port):
                logger.error("后端服务启动超时")
                sys.exit(1)

        import webview

        webview.create_window(
            WINDOW_TITLE,
            f"http://127.0.0.1:{port}",
            width=1100,
            height=760,
            min_size=(800, 600),
        )
        webview.start()
    finally:
        # 仅当最后一个窗口关闭时，才关闭由本实例启动的后端
        remaining = max(0, _read_count() - 1)
        _write_count(remaining)
        if remaining == 0 and backend_proc is not None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
            logger.info("已关闭本实例启动的后端服务")


if __name__ == "__main__":
    main()

