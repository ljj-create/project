"""
飞书机器人 — 消息接收与回复

基于飞书开放平台 API:
- 接收消息事件（Webhook）
- 发送消息回复（消息卡片）
- 文件下载与处理
"""
import json
import tempfile
from pathlib import Path
from loguru import logger

from config.settings import settings


class FeishuBot:
    """
    飞书机器人

    功能:
    1. 验证飞书 Webhook 请求
    2. 解析消息事件
    3. 调用 MessageHandler 处理消息
    4. 通过飞书 API 回复消息
    """

    def __init__(self, message_handler):
        """
        Args:
            message_handler: MessageHandler 实例，处理具体消息逻辑
        """
        self.message_handler = message_handler
        self.app_id = settings.feishu_app_id
        self.app_secret = settings.feishu_app_secret
        self.verification_token = settings.feishu_verification_token
        self.encrypt_key = settings.feishu_encrypt_key

        # 飞书 API 基础 URL
        self.api_base = "https://open.feishu.cn/open-apis"

        # Tenant Access Token
        self._tenant_token = ""
        self._token_expires = 0

        logger.info("飞书机器人初始化完成")

    async def handle_webhook(self, body: dict) -> dict:
        """
        处理飞书 Webhook 请求

        飞书事件回调格式:
        - URL 验证: {"challenge": "...", "type": "url_verification"}
        - 消息事件: {"header": {...}, "event": {...}}
        """
        # URL 验证
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge", "")}

        # 解析事件
        header = body.get("header", {})
        event = body.get("event", {})
        event_type = header.get("event_type", "")

        logger.info(f"收到飞书事件: {event_type}")

        if event_type == "im.message.receive_v1":
            return await self._handle_message_event(event)

        return {"code": 0, "msg": "ok"}

    async def _handle_message_event(self, event: dict) -> dict:
        """处理消息接收事件"""
        message = event.get("message", {})
        sender = event.get("sender", {})

        user_id = sender.get("sender_id", {}).get("open_id", "unknown")
        message_type = message.get("message_type", "")
        message_id = message.get("message_id", "")

        logger.info(f"收到消息 | 用户: {user_id} | 类型: {message_type}")

        try:
            if message_type == "text":
                # 文本消息
                content = json.loads(message.get("content", "{}"))
                text = content.get("text", "")
                reply_content = await self.message_handler.handle_text(user_id, text)

            elif message_type == "file":
                # 文件消息
                content = json.loads(message.get("content", "{}"))
                file_key = content.get("file_key", "")
                file_name = content.get("file_name", "unknown")

                # 下载文件
                file_path = await self._download_file(file_key, file_name)
                if file_path:
                    reply_content = await self.message_handler.handle_file(user_id, file_path, file_name)
                else:
                    from bot.card_builder import CardBuilder
                    reply_content = CardBuilder.error_card("文件下载失败，请重试。")

            else:
                from bot.card_builder import CardBuilder
                reply_content = CardBuilder.text_card(
                    "💡 提示",
                    f"暂不支持 {message_type} 类型的消息。\n\n请发送文本消息或文件。",
                    "orange",
                )

            # 回复消息
            await self._reply_message(message_id, reply_content)

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            from bot.card_builder import CardBuilder
            error_card = CardBuilder.error_card(f"处理消息时出错: {str(e)}")
            await self._reply_message(message_id, error_card)

        return {"code": 0, "msg": "ok"}

    async def _get_tenant_token(self) -> str:
        """获取 Tenant Access Token"""
        import httpx
        import time

        if self._tenant_token and time.time() < self._token_expires:
            return self._tenant_token

        url = f"{self.api_base}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") == 0:
            self._tenant_token = data["tenant_access_token"]
            self._token_expires = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        else:
            raise RuntimeError(f"获取 Token 失败: {data}")

    async def _reply_message(self, message_id: str, card_content: dict):
        """通过飞书 API 回复消息"""
        import httpx

        token = await self._get_tenant_token()
        url = f"{self.api_base}/im/v1/messages/{message_id}/reply"

        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "content": json.dumps(card_content),
            "msg_type": "interactive",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") == 0:
            logger.info(f"消息回复成功: {message_id}")
        else:
            logger.error(f"消息回复失败: {data}")

    async def _download_file(self, file_key: str, file_name: str) -> str | None:
        """下载飞书文件"""
        import httpx

        token = await self._get_tenant_token()
        url = f"{self.api_base}/im/v1/messages/resources/{file_key}"
        headers = {"Authorization": f"Bearer {token}"}

        # 保存到临时目录
        tmp_dir = Path(tempfile.mkdtemp())
        # 飞书返回的文件名可能包含路径分隔符，这里只保留文件名，避免路径穿越
        safe_name = Path(file_name).name or "uploaded_file"
        file_path = tmp_dir / safe_name

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                file_path.write_bytes(resp.content)
                logger.info(f"文件下载成功: {file_path}")
                return str(file_path)
        except Exception as e:
            logger.error(f"文件下载异常: {e}")
            return None
