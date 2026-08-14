"""
CS-Agent 桌面端启动器

以桌面窗口方式运行 CS-Agent（pywebview + 内置中文聊天界面）。

用法:
    python desktop_app.py              # 启动桌面端
    python desktop_app.py --port 8000  # 指定端口
"""
import sys
import time
import argparse
import threading
import urllib.request
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

WINDOW_TITLE = "CS-Agent 智能体"


def server_alive(port: int) -> bool:
    """检测本地服务是否已在运行"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_backend(port: int) -> None:
    """在线程中启动 FastAPI 服务"""
    import uvicorn

    from scripts.run import create_app

    app = create_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    parser = argparse.ArgumentParser(description="CS-Agent 桌面端")
    parser.add_argument("--port", type=int, default=8000, help="服务端口（默认 8000）")
    args = parser.parse_args()

    port = args.port
    backend_thread = None

    if server_alive(port):
        logger.info(f"检测到服务已在 http://127.0.0.1:{port} 运行，直接打开窗口")
    else:
        logger.info(f"启动后端服务 http://127.0.0.1:{port} ...")
        backend_thread = threading.Thread(
            target=start_backend, args=(port,), daemon=True
        )
        backend_thread.start()

        # 等待服务就绪
        for _ in range(60):
            if server_alive(port):
                break
            time.sleep(0.5)
        if not server_alive(port):
            logger.error("后端服务启动失败，请检查日志")
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


if __name__ == "__main__":
    main()
