---
title: Automatic Claude Code Routing (Discord)
sidebar_label: Claude Code Auto-Routing
description: "Configure a Discord channel so ordinary messages are delegated straight to a Claude Code worker instead of Hermes' own agent loop"
---

# Automatic Claude Code Routing

Lets you mark a specific Discord channel so that ordinary messages there are delegated straight to an authenticated [Claude Code](https://code.claude.com/docs/en/cli-reference) tmux worker instead of going through Hermes' own agent loop. It builds entirely on the existing [Claude Code tmux bridge](/reference/slash-commands#delegating-to-a-claude-code-worker-cc-delegate) — nothing here talks to tmux or Discord directly, and every message still goes through the same worker allowlisting, workspace containment, per-session locking, approval-required detection, response extraction, and sanitisation that `/cc-delegate` uses.

Unlike `/cc-delegate`, which is something *you* type explicitly, auto-routing fires on **plain messages** in a channel you've opted in — so a channel dedicated to one repo's ongoing work can feel like talking directly to Claude Code, without retyping `/cc-delegate` every time.

Delivery is asynchronous: the channel gets an immediate acknowledgement (worker, workspace, and a stable task ID), Claude Code runs in the background for as long as it needs (well past the bridge's normal 300-second default — see [Background-worker timeout](#background-worker-timeout)), and the result posts back to the same thread/channel whenever it's actually done. See [`/cc-tasks`](#checking-task-state-cc-tasks) to check on a task without waiting.

## Nothing is on by default

No channel is auto-routed until you add a `claude_routing` entry to `config.yaml` (below). Without that config, this feature does nothing — every message goes through Hermes exactly as it always has.

## Configuration

Add a `claude_routing` list to `~/.hermes/config.yaml`:

```yaml
claude_routing:
  - platform: discord
    channel_id: "123456789012345678"   # the Discord channel's ID (Developer Mode -> right-click -> Copy Channel ID)
    enabled: true
    worker: claude-momentum             # must be a session registered in agent/claude_code_tmux_bridge.py
    workspace: /home/michael/code/momentum-studio  # must be the worker's root or a subdirectory of it
    mode: claude_default
    standard_instructions: true         # true (default block), false (omit), or custom text
    background_timeout_seconds: 1800    # optional; defaults to 1800 (30 min)
```

| Field | Required | Meaning |
|---|---|---|
| `platform` | no (defaults to `discord`) | Currently only `discord` is supported. |
| `channel_id` | **yes** | The Discord channel's numeric ID. Messages in threads under this channel match too. |
| `enabled` | no (defaults to `true`) | Set `false` to turn a mapping off without deleting it. |
| `worker` | **yes** | A session name already registered with the tmux bridge (`claude-momentum` ships by default). Unregistered names are rejected, not guessed at. |
| `workspace` | **yes** | An absolute path that must resolve to the worker's configured workspace root or a subdirectory of it. Anything else is rejected before it reaches the worker. |
| `mode` | no (defaults to `claude_default`) | See [Routing modes](#routing-modes) below. |
| `standard_instructions` | no (defaults to `true`) | `true` includes the built-in requirements block (below); `false` omits it; any other string is used verbatim instead. |
| `background_timeout_seconds` | no (defaults to `1800`) | How long the background task waits for Claude Code before giving up and reporting a timeout. Must be a positive number. See [Background-worker timeout](#background-worker-timeout). |

Restart the gateway (`hermes gateway restart`) after editing `claude_routing` — it's read from the loaded config, the same as `quick_commands`.

**Do not hardcode a channel ID in source.** The registry only exists in your own `config.yaml`; nothing in the Hermes codebase references a specific channel.

### Momentum Studio example

The exact block to add for delegating the Momentum Studio channel to the `claude-momentum` worker:

```yaml
claude_routing:
  - platform: discord
    channel_id: "REPLACE_WITH_YOUR_CHANNEL_ID"
    enabled: true
    worker: claude-momentum
    workspace: /home/michael/code/momentum-studio
    mode: claude_default
```

## Routing modes

`claude_default` is the only mode implemented so far: **deterministic, unconditional** delegation. Every plain-text message in the channel that isn't a command and doesn't use the `chat:` bypass goes straight to the configured worker — there's no content-based classification deciding whether a given message "looks like a coding task." That's intentional: an operator opts a whole channel in, rather than Hermes guessing per-message.

## Background-worker timeout

`/cc-delegate` waits synchronously (up to the bridge's own 300-second default) because you're watching for the reply. Auto-routing doesn't have that constraint — it already acknowledged and moved on — so it uses a much longer, separately configurable ceiling: **1800 seconds (30 minutes) by default**, overridable per channel via `background_timeout_seconds`. A task that's still working at the 300-second mark is not abandoned; it's only reported as timed out if it's still running once `background_timeout_seconds` is reached. Set a shorter value for a channel where you want faster failure feedback, or longer for a workspace that routinely runs long test suites or builds.

## Checking task state (`/cc-tasks`)

`/cc-tasks [limit]` (default 10, max 50) lists recent auto-routed tasks: task ID, worker, workspace, **execution state** (`running` / `finished` / `interrupted`), and **delivery state** (`pending` / `delivered` / `failed`). It's read-only — it doesn't retry, cancel, or otherwise act on anything, just shows you what the registry currently has recorded. Useful while a long task is still running, or to confirm a result was actually delivered.

## Duplicate events

A stable task ID is derived deterministically from the triggering Discord message (not randomly), so if the exact same message is ever reprocessed — a gateway reconnect replay, a missed-message backfill, a retried dispatch — auto-routing recognises it and replies with a short "already dispatched" notice instead of starting a second Claude Code task or delivering the result twice. A genuinely new message, even with identical text, gets its own message ID and its own task, and is handled normally.

## Restart recovery

The gateway can restart at any point during a long-running auto-routed task. What happens depends on exactly when:

- **Claude Code finished, but the result wasn't confirmed-delivered yet** (crash between the bridge call returning and the Discord send succeeding) — recovered automatically. This reuses Hermes' existing delivery ledger (the same durable mechanism that protects any other final response), which redelivers the stored result on the next boot.
- **Claude Code was still working when the gateway died** — **not** automatically recovered. The underlying tmux session and Claude Code process are independent of the gateway and may keep working (or may already be done) in tmux, but nothing is watching for that specific completion anymore after a restart. The task shows as `interrupted` in `/cc-tasks` so this is visible rather than silently lost. Send a fresh message (or use `/cc-delegate` against the same worker to check in) to pick the work back up — it gets a new task ID and proceeds normally.

## Standard implementation instructions

Unless a mapping sets `standard_instructions: false`, every auto-routed prompt is prefixed with:

> Standard implementation requirements for this task: inspect the current repository state first; use a new branch or an isolated worktree; do not make unrelated changes; run relevant tests and checks; commit intentionally; push the branch; create or update a pull request; and report the branch, commit, PR link, files changed, and tests run.

This is advisory framing text, not a rule enforced in code — Claude Code is an agent in its own right and reconciles it against anything your message explicitly says (e.g. "just show me the diff, don't push yet" in your own message takes precedence). Set `standard_instructions` to a custom string on a mapping to replace the block entirely, or `false` to omit it.

## Bypass syntax: keep a message with Hermes

Prefix a message with `chat:` to keep that one message with Hermes instead of routing it to Claude Code:

```
@hermes chat: Help me think through the strategy before changing code.
```

The `chat:` prefix is stripped and the rest of the message goes to Hermes' normal agent loop, exactly as if auto-routing weren't configured for the channel at all. This is a per-message opt-out — every other message in the channel keeps auto-routing.

## What still works normally

- **Explicit slash commands** (`/status`, `/cc-delegate`, `/<skill-name>`, etc.) are never auto-routed — they dispatch exactly as they do everywhere else.
- **`/cc-delegate`** still works in an auto-routed channel for one-off delegations to a *different* worker or workspace than the channel's default mapping.
- **Bot messages** (including Hermes' own) are never auto-routed — no recursive delegation loop.
- **Duplicate Discord events** are filtered the same way they always are (the gateway's existing message dedup) *and* by task-ID dedup (see [Duplicate events](#duplicate-events)) — a second delegation attempt for a worker that's still busy on the first also gets a clear `busy` reply rather than a second Claude Code task.

## Failure handling

Auto-routing never silently falls back to substantial Hermes/OpenAI implementation work. Once a channel is configured and a message isn't a command or bypassed, the outcome is always reported back to the chat:

- ✅ success — Claude's response, sanitised, same as `/cc-delegate`
- ⏸️ approval required — Claude Code is waiting on an interactive approval prompt
- 🔒 busy — the worker is already handling another prompt
- 🔑 authentication failure — the worker needs to re-login
- ❌ / 💀 the worker's tmux session is missing or dead
- ⏱️ timeout, or ⚠️ a generic/extraction failure
- ⚠️ a misconfigured mapping (e.g. an unregistered `worker`, or a `workspace` outside the worker's allowed root) — this is reported the same way, never silently widened or ignored

The original request is never discarded on failure — retry it (or fix the `claude_routing` config, for a misconfiguration) and send it again.

## Testing automatic routing

1. Add a `claude_routing` entry for a test channel and restart the gateway.
2. Send a plain message in that channel (no leading `/`): `@hermes say hello`.
3. You should see an immediate **"🤖 Automatic routing activated"** acknowledgement naming the worker, workspace, and a task ID, followed by the worker's response (or a clear busy/approval/failure status) once it's ready — this can take a while for larger tasks, that's expected.
4. Run `/cc-tasks` to see the task listed, and check its execution/delivery state change from `running`/`pending` to `finished`/`delivered`.
5. Confirm `/cc-delegate` and other slash commands still work normally in the same channel.
6. Confirm `@hermes chat: ...` in that channel goes to Hermes instead.

## Disabling it quickly

Set `enabled: false` on the mapping (no need to delete it) and restart the gateway:

```yaml
claude_routing:
  - platform: discord
    channel_id: "123456789012345678"
    enabled: false   # <-- messages in this channel go back to normal Hermes handling
    worker: claude-momentum
    workspace: /home/michael/code/momentum-studio
    mode: claude_default
```

Or remove the `claude_routing` block (or the gateway config change) entirely and restart — with no entries, every channel behaves exactly as it did before this feature existed.
