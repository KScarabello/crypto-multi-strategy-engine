"""Email notification utility for dry-run/live trading activity.

Transferred from crypto-momentum-strategy and adapted for multi-strategy use.

This module is intentionally isolated from execution logic. It only provides a
small helper for sending trade-related notifications using SMTP credentials from
.env, so it can be called after a trade is submitted or filled.
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage

from dotenv import load_dotenv


@dataclass(frozen=True)
class EmailSettings:
    """SMTP/email configuration loaded from environment variables."""

    smtp_host: str
    smtp_port: int
    use_tls: bool
    username: str
    password: str
    from_email: str
    to_email: str


def _load_email_settings() -> EmailSettings:
    """Load email settings from .env and environment variables."""
    load_dotenv()

    smtp_host = os.getenv("EMAIL_SMTP_HOST") or os.getenv("SMTP_SERVER") or "smtp.gmail.com"
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT") or os.getenv("SMTP_PORT") or "587")
    use_tls = os.getenv("EMAIL_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}

    username = os.getenv("EMAIL_USERNAME") or os.getenv("SMTP_USER") or ""
    password = os.getenv("EMAIL_PASSWORD") or os.getenv("SMTP_PASS") or ""
    from_email = os.getenv("EMAIL_FROM", username)
    to_email = os.getenv("EMAIL_TO", "")

    if not username or not password or not from_email or not to_email:
        raise ValueError(
            "Missing email configuration. Set EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_FROM, and EMAIL_TO in .env"
        )

    return EmailSettings(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        use_tls=use_tls,
        username=username,
        password=password,
        from_email=from_email,
        to_email=to_email,
    )


def send_trade_notification(
    strategy_variant: str,
    timestamp: datetime | str,
    symbol: str,
    side: str,
    notional_or_quantity: float,
    status_text: str | None = None,
) -> bool:
    """Send a single trade activity email notification.

    Args:
        strategy_variant: Strategy label (for example: "momentum_baseline").
        timestamp: Trade-related timestamp (signal, submit, or fill time).
        symbol: Trading symbol (for example: "BTC/USD").
        side: Trade side (BUY/SELL).
        notional_or_quantity: Trade size as notional USD or quantity.
        status_text: Optional status note (for example: "submitted", "filled").

    Returns:
        True if email is sent successfully; raises exception on failure.

    Integration note:
        Call this from the execution pipeline immediately after a trade is
        submitted and/or when a fill confirmation is received.
    """
    settings = _load_email_settings()

    side_upper = side.strip().upper()
    status_line = status_text.strip() if status_text else "none"
    ts_text = str(timestamp)

    subject = f"Trade Notification | {side_upper} {symbol} | {strategy_variant}"
    body = (
        "Trade activity detected\n\n"
        f"strategy_variant: {strategy_variant}\n"
        f"timestamp: {ts_text}\n"
        f"symbol: {symbol}\n"
        f"side: {side_upper}\n"
        f"notional_or_quantity: {notional_or_quantity:.8f}\n"
        f"status: {status_line}\n"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.from_email
    msg["To"] = settings.to_email
    msg.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        if settings.use_tls:
            server.starttls()
        server.login(settings.username, settings.password)
        server.send_message(msg)

    return True
