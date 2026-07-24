# DERIV TOUCH BOT v1

A professional Deriv trading bot built in Python that trades ONETOUCH contracts on synthetic volatility indices using supertrend, market-session analysis, and fast ML confidence gating.

## Running the bot

- Workflow: `Start application` (`python exo_bot.py`)
- Main file: `exo_bot.py`

## Required secrets / environment variables

- `DERIV_TOKEN` — Deriv API token / PAT (Bearer token for the new API).
- `DERIV_APP_ID` — Deriv application ID.
  - Legacy API: numeric (e.g., `1089`).
  - New Options API: alphanumeric app ID from the Deriv app dashboard.
- `DERIV_ACCOUNT_ID` *(optional)* — Options trading account ID to use (e.g., `DOT93850888`). When not set, the bot auto-selects a demo account for safety.
- `DERIV_USE_NEW_API` *(optional)* — Force new API mode when set to `1`, `true`, or `yes`.
- `TG_BOT_TOKEN` — Telegram bot token (optional; fallback is hardcoded for development).
- `TG_CHAT_ID` — Telegram chat ID (optional; fallback is hardcoded for development).
- `NEON_DATABASE_URL` or `DATABASE_URL` — Optional Postgres URL; otherwise SQLite is used.

## User preferences

- Bot auto-detects the new Deriv Options API when `DERIV_APP_ID` is alphanumeric, falling back to the legacy v3 API for numeric IDs.
- When using the new API and no `DERIV_ACCOUNT_ID` is set, the bot prefers the demo account for safety.
