---
name: Deriv API Migration
description: Notes on migrating the Deriv integration from the legacy v3 WebSocket API to the new Options API.
---

# Deriv API Migration

## Legacy vs New API detection

- Legacy API app IDs are numeric (e.g., `1089`).
- New Options API app IDs are alphanumeric (e.g., `a1b2c3d4e5`).
- The bot auto-detects the new API when `DERIV_APP_ID` is non-numeric. Set `DERIV_USE_NEW_API=1` to force it.

## New auth flow

1. REST base URL: `https://api.derivws.com`.
2. Every authenticated REST call must include:
   - `Deriv-App-ID: {DERIV_APP_ID}`
   - `Authorization: Bearer {DERIV_TOKEN}`
3. List accounts: `GET /trading/v1/options/accounts`.
4. Get a one-time WebSocket URL: `POST /trading/v1/options/accounts/{accountId}/otp`.
5. Connect directly to the returned `wss://api.derivws.com/trading/v1/options/ws/{demo|real}?otp=...` URL. No `authorize` WebSocket message is needed.

## Breaking WebSocket message changes

- **`proposal`**: the field `symbol` was renamed to `underlying_symbol`.
- **`buy`**: `loginid` was removed; otherwise the request is unchanged.
- **Response types**: `ask_price`, `payout`, `buy_price`, `profit` may now be returned as strings instead of numbers. Always parse them with a tolerant float helper.
- Market-data messages (`ticks`, `candles`, `ticks_history`) are reported to be identical to the legacy API.

**Why:** Deriv is moving legacy API users to the new Options API, which requires app-scoped PAT tokens and account-bound OTP WebSocket connections. The old `app_id` query-param flow no longer works with the new app IDs.

**How to apply:** When adding or editing any Deriv request/response handling, check whether the endpoint is legacy or new, use the right symbol field, and parse numeric response fields defensively.
