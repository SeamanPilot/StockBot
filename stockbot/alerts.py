from __future__ import annotations

import json
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("stockbot.alerts")


class AlertManager:
    """Send trade notifications via email and/or webhook."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_pass: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_addr: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.smtp_host = smtp_host or os.environ.get("STOCKBOT_SMTP_HOST")
        self.smtp_port = int(os.environ.get("STOCKBOT_SMTP_PORT", smtp_port))
        self.smtp_user = smtp_user or os.environ.get("STOCKBOT_SMTP_USER")
        self.smtp_pass = smtp_pass or os.environ.get("STOCKBOT_SMTP_PASS")
        self.from_addr = from_addr or os.environ.get("STOCKBOT_EMAIL_FROM")
        self.to_addr = to_addr or os.environ.get("STOCKBOT_EMAIL_TO")
        self.webhook_url = webhook_url or os.environ.get("STOCKBOT_WEBHOOK_URL")

    def _format_message(self, event: dict) -> str:
        action = event.get("action", "?").upper()
        symbol = event.get("symbol", "?")
        price = event.get("price", 0)
        reason = event.get("reason", "")
        equity = event.get("equity", 0)
        return f"StockBot {action} {symbol} @ ${price:.2f} | Equity: ${equity:,.2f} | {reason}"

    def send_email(self, event: dict) -> bool:
        if not all([self.smtp_host, self.smtp_user, self.smtp_pass, self.from_addr, self.to_addr]):
            logger.debug("Email not configured, skipping")
            return False
        msg = EmailMessage()
        msg["Subject"] = f"StockBot Alert: {event.get('action', 'event').upper()} {event.get('symbol', '')}"
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg.set_content(self._format_message(event))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            logger.info("Email alert sent for %s", event.get("symbol"))
            return True
        except Exception as e:
            logger.error("Email alert failed: %s", e)
            return False

    def send_webhook(self, event: dict) -> bool:
        if not self.webhook_url:
            logger.debug("Webhook not configured, skipping")
            return False
        payload = json.dumps({
            "text": self._format_message(event),
            "symbol": event.get("symbol"),
            "action": event.get("action"),
            "price": event.get("price"),
            "equity": event.get("equity"),
            "reason": event.get("reason"),
        }).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=10) as resp:
                if resp.status < 300:
                    logger.info("Webhook alert sent for %s", event.get("symbol"))
                    return True
                logger.error("Webhook returned status %d", resp.status)
                return False
        except (URLError, Exception) as e:
            logger.error("Webhook alert failed: %s", e)
            return False

    def notify(self, event: dict) -> dict[str, bool]:
        results = {
            "email": self.send_email(event),
            "webhook": self.send_webhook(event),
        }
        return results
