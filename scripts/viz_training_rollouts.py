"""Gradio viewer for BabyAI training rollouts from wandb.

Turn-by-turn navigation with observation/response split view,
action timeline, thinking extraction, and episode-level stats.

Usage:
    uv run python scripts/viz_training_rollouts.py <wandb_run_path>
    uv run python scripts/viz_training_rollouts.py --local /path/to/final-samples.table.json
    uv run python scripts/viz_training_rollouts.py  # auto-discovers from /tmp/kv-eviction/
"""

import argparse
import html
import json
import re
from pathlib import Path

import gradio as gr

# Chat template tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IM_START_ID = 151644
IM_END_ID = 151645
NEWLINE_ID = 198

_tokenizer = None


def get_tokenizer():
    """Lazy-load the Qwen3 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    return _tokenizer


def compute_message_boundaries(messages_text: str) -> list[dict]:
    """Tokenize messages and return per-message token boundaries.

    Returns list of {role, start, end, len, preview} where start/end are
    token positions in the full sequence (exclusive end).
    """
    tok = get_tokenizer()
    full_ids = tok.encode(messages_text, add_special_tokens=False)

    boundaries = []
    i = 0
    while i < len(full_ids):
        if full_ids[i] == IM_START_ID:
            msg_start = i
            # Find im_end
            j = i + 1
            while j < len(full_ids) and full_ids[j] != IM_END_ID:
                j += 1
            msg_end = j + 1  # include im_end token
            # Decode role (tokens between im_start and first newline)
            role_end = i + 1
            while role_end < j and full_ids[role_end] != NEWLINE_ID:
                role_end += 1
            role = tok.decode(full_ids[i + 1 : role_end]).strip()
            # Decode full message content
            content_start = role_end + 1 if role_end < j else role_end
            content = tok.decode(full_ids[content_start:j]).strip()
            boundaries.append({
                "role": role,
                "start": msg_start,
                "end": msg_end,
                "len": msg_end - msg_start,
                "content": content,
            })
            # Skip past im_end + optional newline
            i = msg_end
            if i < len(full_ids) and full_ids[i] == NEWLINE_ID:
                i += 1
        else:
            i += 1

    return boundaries

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

ACTION_ICONS = {
    "turn left":  "↺  turn left",
    "turn right": "↻  turn right",
    "go forward": "⬆  go forward",
    "pick up":    "✋  pick up",
    "drop":       "⬇  drop",
    "toggle":     "🔓  toggle",
}

CSS = """
/* Message panels */
.msg-wrap { font-family: monospace; font-size: 13px; line-height: 1.5; }
.msg { margin: 6px 0; border-radius: 6px; padding: 8px 10px; }
.msg-role { font-size: 10px; font-weight: bold; letter-spacing: 1px; opacity: 0.7; margin-bottom: 3px; }
.msg-content { white-space: pre-wrap; word-break: break-word; }
.msg-system { background: #1a1a2e; border-left: 3px solid #4a4a8a; color: #ccc; }
.msg-user { background: #1a2a1a; border-left: 3px solid #27ae60; color: #ccc; }
.msg-assistant { background: #1a2a3a; border-left: 3px solid #2980b9; color: #ccc; }
.msg-tool { background: #2a1a1a; border-left: 3px solid #e74c3c; color: #ccc; }
.think { color: #888; font-style: italic; margin: 4px 0; padding: 4px 8px; border-left: 2px solid #555; }
.action-tag { color: #f39c12; font-weight: bold; font-size: 14px; margin-top: 6px; }
.tool-call { background: #2a2a1a; border: 1px dashed #f39c12; border-radius: 4px; padding: 4px 6px; margin: 4px 0; font-size: 12px; }

/* Action timeline */
.timeline-wrap { padding: 8px 4px; font-family: monospace; }
.timeline-title { font-size: 11px; color: #888; margin-bottom: 6px; }
.timeline-chips { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 8px; }
.timeline-chip {
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 36px; padding: 3px 6px; border-radius: 4px;
    border: 2px solid transparent; cursor: default; font-size: 13px;
    position: relative;
}
.timeline-chip.active { border-color: #ffcc00 !important; }
.chip-num { font-size: 10px; font-weight: bold; opacity: 0.7; }
.chip-compact { font-size: 8px; color: #e74c3c; font-weight: bold; margin-top: 1px; }

/* Compaction bar */
.compact-bar { font-family: monospace; font-size: 12px; padding: 6px 10px;
               background: #1a0a0a; border: 1px solid #4a1a1a; border-radius: 4px;
               margin: 4px 0; color: #e74c3c; }
.compact-bar .label { color: #888; font-size: 10px; }
.compact-bar .evict-count { color: #e74c3c; font-weight: bold; }
.compact-bar .prompt-count { color: #f39c12; }

/* Observation panel */
.obs-wrap { font-family: monospace; font-size: 14px; line-height: 1.6; padding: 10px;
            background: #111; border-radius: 6px; border: 1px solid #333; color: #ccc; }
.obs-item { margin: 2px 0; }
.obs-object { color: #f39c12; font-weight: bold; }
.obs-wall { color: #7f8c8d; }
.obs-direction { color: #2980b9; }

/* KV cache state visualization */
.kv-wrap { font-family: monospace; font-size: 11px; padding: 8px 10px;
           background: #0a0a0a; border: 1px solid #333; border-radius: 6px; margin: 4px 0; }
.kv-title { font-size: 10px; color: #888; margin-bottom: 6px; font-weight: bold; letter-spacing: 1px; }
.kv-source { font-size: 10px; color: #666; margin-bottom: 8px; font-style: italic; }
.kv-bar { display: flex; height: 28px; border-radius: 3px; overflow: hidden; margin-bottom: 6px; }
.kv-seg { display: flex; align-items: center; justify-content: center; overflow: hidden;
          font-size: 9px; font-weight: bold; color: #fff; min-width: 2px;
          border-right: 1px solid #0a0a0a; white-space: nowrap; }
.kv-seg.protected { background: #1a5c2a; }
.kv-seg.retained  { background: #1a3a5a; }
.kv-seg.evicted   { background: #5a1a1a; }
.kv-seg.future    { background: #222; color: #555; }
.kv-legend { display: flex; gap: 12px; color: #888; font-size: 10px; margin-bottom: 8px; }
.kv-legend span { display: inline-flex; align-items: center; gap: 3px; }
.kv-legend .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
.kv-msgs { margin-top: 6px; max-height: 500px; overflow-y: auto; }
.kv-msg-block { margin: 4px 0; border-radius: 4px; padding: 6px 8px; font-size: 12px; }
.kv-msg-block.evicted   { background: #2a0a0a; border-left: 3px solid #e74c3c; color: #e74c3c; opacity: 0.7; }
.kv-msg-block.retained  { background: #0a1a2a; border-left: 3px solid #2980b9; color: #7eb8da; }
.kv-msg-block.protected { background: #0a2a0a; border-left: 3px solid #27ae60; color: #5cb85c; }
.kv-msg-block.future    { background: #161616; border-left: 3px solid #333; color: #555; }
.kv-msg-header { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.kv-msg-role { font-weight: bold; font-size: 10px; letter-spacing: 1px; }
.kv-msg-tokens { font-size: 10px; opacity: 0.7; }
.kv-msg-range { font-size: 10px; opacity: 0.5; }
.kv-msg-status { font-size: 9px; font-weight: bold; padding: 1px 5px; border-radius: 3px; }
.kv-msg-status.evicted   { background: #5a1a1a; color: #e74c3c; }
.kv-msg-status.retained  { background: #1a3a5a; color: #7eb8da; }
.kv-msg-status.protected { background: #1a5c2a; color: #5cb85c; }
.kv-msg-content { white-space: pre-wrap; word-break: break-word; line-height: 1.4; }
.kv-msg-block.evicted .kv-msg-content { text-decoration: line-through; }
"""


def parse_messages(raw_text: str) -> list[dict]:
    """Parse chat-template formatted text into structured messages."""
    messages = []
    parts = raw_text.split(IM_START)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = part.replace(IM_END, "").strip()
        newline_idx = part.find("\n")
        if newline_idx == -1:
            role = part.strip()
            content = ""
        else:
            role = part[:newline_idx].strip()
            content = part[newline_idx + 1:].strip()
        messages.append({"role": role, "content": content})
    return messages


def extract_action(content: str) -> str | None:
    """Extract action from tool_call in assistant message."""
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return args.get("action")
    except (json.JSONDecodeError, AttributeError):
        return None


def extract_thinking(content: str) -> tuple[str, str]:
    """Split content into (thinking, rest)."""
    match = THINK_RE.search(content)
    if not match:
        return "", content
    thinking = match.group(1).strip()
    rest = content[:match.start()] + content[match.end():]
    return thinking, rest.strip()


def extract_tool_response(content: str) -> str | None:
    """Extract observation text from <tool_response> wrapper."""
    match = TOOL_RESPONSE_RE.search(content)
    return match.group(1).strip() if match else None


def messages_to_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert flat message list into structured turns.

    Returns (system_prompt, turns) where each turn has:
        obs: str, response: str, action: str|None, thinking: str
    """
    system_prompt = ""
    turns = []
    i = 0

    # Extract system prompt
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        i = 1

    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            obs_content = msg["content"]
            # Strip tool_response wrapper if present
            tool_resp = extract_tool_response(obs_content)
            if tool_resp:
                obs_content = tool_resp

            response = ""
            action = None
            thinking = ""

            # Look for the assistant response
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                response = messages[i + 1]["content"]
                action = extract_action(response)
                thinking, _ = extract_thinking(response)
                i += 2
            else:
                i += 1

            turns.append({
                "obs": obs_content,
                "response": response,
                "action": action,
                "thinking": thinking,
            })
        else:
            i += 1

    return system_prompt, turns


def render_observation_html(obs: str) -> str:
    """Render observation text with highlighted objects and directions."""
    lines = obs.strip().splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        escaped = html.escape(line)
        # Highlight object names (colored objects)
        for color in ("red", "green", "blue", "purple", "yellow", "grey"):
            for obj in ("key", "ball", "box", "door"):
                pattern = f"{color} {obj}"
                if pattern in escaped:
                    escaped = escaped.replace(
                        pattern,
                        f'<span class="obs-object">{pattern}</span>',
                    )
        # Highlight walls
        if "wall" in escaped:
            escaped = escaped.replace("wall", '<span class="obs-wall">wall</span>')
        # Highlight goal
        if "goal" in escaped.lower():
            escaped = re.sub(
                r"(goal)",
                r'<span class="obs-object">\1</span>',
                escaped,
                flags=re.IGNORECASE,
            )
        # Highlight directions
        for d in ("forward", "left", "right"):
            escaped = escaped.replace(d, f'<span class="obs-direction">{d}</span>')
        items.append(f'<div class="obs-item">{escaped}</div>')
    return f'<div class="obs-wrap">{"".join(items)}</div>'


def render_response_html(response: str) -> str:
    """Render assistant response with thinking and tool call sections."""
    thinking, rest = extract_thinking(response)
    action = extract_action(response)

    parts = ['<div class="msg-wrap">']

    if thinking:
        truncated = thinking[:800] + ("..." if len(thinking) > 800 else "")
        parts.append(f'<div class="think">{html.escape(truncated)}</div>')

    # Remove thinking and tool_call from the visible "reasoning" text
    rest_clean = rest
    rest_clean = TOOL_CALL_RE.sub("", rest_clean).strip()

    if rest_clean:
        parts.append(
            f'<div class="msg msg-assistant">'
            f'<div class="msg-role">REASONING</div>'
            f'<div class="msg-content">{html.escape(rest_clean)}</div>'
            f'</div>'
        )

    if action:
        icon = ACTION_ICONS.get(action, action)
        parts.append(f'<div class="action-tag">{icon}</div>')
    else:
        # Show raw tool_call if action parsing failed
        match = TOOL_CALL_RE.search(response)
        if match:
            parts.append(f'<div class="tool-call">{html.escape(match.group(1).strip())}</div>')

    parts.append('</div>')
    return "".join(parts)


def render_compaction_bar(step_events: list[dict] | None) -> str:
    """Render a compaction info bar for the current turn."""
    if not step_events:
        return ""
    total_evicted = sum(e.get("evicted", 0) for e in step_events)
    n_events = len(step_events)
    last_offset = step_events[-1].get("offset", 0)
    last_prompt = step_events[-1].get("prompt", 0)
    return (
        '<div class="compact-bar">'
        f'<span class="label">COMPACTION</span> '
        f'<span class="evict-count">{n_events} eviction{"s" if n_events != 1 else ""}, '
        f'{total_evicted} tokens removed</span> · '
        f'<span class="prompt-count">prompt after: {last_prompt} tokens</span> · '
        f'offset: {last_offset}'
        '</div>'
    )


def render_action_timeline(turns: list[dict], current_turn: int,
                           per_step_events: list | None = None) -> str:
    """Render a visual timeline of actions across the episode."""
    chips = []
    for i, t in enumerate(turns):
        action = t.get("action")
        active_cls = " active" if i == current_turn else ""

        if action in ("go forward",):
            bg, fg, icon = "#1a3a1a", "#27ae60", "⬆"
        elif action in ("turn left",):
            bg, fg, icon = "#1a1a3a", "#2980b9", "↺"
        elif action in ("turn right",):
            bg, fg, icon = "#1a1a3a", "#2980b9", "↻"
        elif action in ("pick up",):
            bg, fg, icon = "#3a2a10", "#f39c12", "✋"
        elif action in ("drop",):
            bg, fg, icon = "#3a2a10", "#f39c12", "⬇"
        elif action in ("toggle",):
            bg, fg, icon = "#2a1a3a", "#8e44ad", "🔓"
        else:
            bg, fg, icon = "#2a1a1a", "#e74c3c", "?"

        # Compaction indicator on the chip
        compact_label = ""
        if per_step_events and i < len(per_step_events) and per_step_events[i]:
            evicted = sum(e.get("evicted", 0) for e in per_step_events[i])
            compact_label = f'<span class="chip-compact">-{evicted}</span>'

        chips.append(
            f'<div class="timeline-chip{active_cls}" style="background:{bg};color:{fg};">'
            f'{icon}<span class="chip-num">{i + 1}</span>{compact_label}</div>'
        )

    return (
        '<div class="timeline-wrap">'
        '<div class="timeline-title">Action timeline — green: forward · blue: turn · orange: interact · purple: toggle · '
        '<span style="color:#e74c3c">red number</span>: tokens evicted · 🟡 = current</div>'
        f'<div class="timeline-chips">{"".join(chips)}</div>'
        '</div>'
    )


def get_eviction_window(per_step_events: list | None, turn: int) -> int:
    """Get cumulative eviction offset for a given turn.

    Returns the total tokens evicted from position evict_start during
    this turn's inference (0 if no compaction at this turn).
    """
    if not per_step_events or turn >= len(per_step_events):
        return 0
    events = per_step_events[turn]
    if not events:
        return 0
    # The last event's offset is the cumulative eviction for this request
    return events[-1].get("offset", 0)


def render_kv_cache_state(
    msg_boundaries: list[dict] | None,
    per_step_events: list | None,
    turn: int,
    total_turns: int,
) -> str:
    """Render a visual KV cache state showing retained vs evicted tokens.

    Maps compaction events back to message boundaries to show exactly
    which messages are in the KV cache when the model generates at this turn.
    Each message is shown in full with color-coded background:
      green = protected system prompt, blue = retained, red = evicted.
    """
    if not msg_boundaries:
        return ""

    offset = get_eviction_window(per_step_events, turn)

    # The first message is system prompt — always protected.
    # Block-aligned evict_start: round up system prompt end to block_size=16.
    sys_end = msg_boundaries[0]["end"] if msg_boundaries else 0
    evict_start = ((sys_end + 15) // 16) * 16
    evict_end = evict_start + offset  # original tokens [evict_start, evict_end) removed

    # How many messages are visible at this turn.
    # Each viewer "turn" = a user+assistant pair, plus the system message.
    msgs_visible = 1 + 2 * (turn + 1)
    msgs_visible = min(msgs_visible, len(msg_boundaries))

    total_tokens = msg_boundaries[-1]["end"] if msg_boundaries else 0

    # Classify each message and build bar + message blocks
    bar_segments = []
    msg_blocks = []

    for i, mb in enumerate(msg_boundaries):
        if i >= msgs_visible:
            status = "future"
        elif i == 0:
            status = "protected"
        elif offset == 0:
            status = "retained"
        else:
            msg_s, msg_e = mb["start"], mb["end"]
            overlap_s = max(msg_s, evict_start)
            overlap_e = min(msg_e, evict_end)
            if overlap_e <= overlap_s:
                status = "retained"
            elif overlap_s <= msg_s and overlap_e >= msg_e:
                status = "evicted"
            else:
                status = "evicted"  # partially evicted — mark as evicted

        # Bar segment
        label = f"SYS {mb['len']}t" if status == "protected" else f"{mb['len']}"
        if status == "future":
            label = ""
        bar_segments.append((status, mb["len"], label))

        # Also split bar for partial eviction
        if status == "evicted" and offset > 0 and i > 0:
            msg_s, msg_e = mb["start"], mb["end"]
            overlap_s = max(msg_s, evict_start)
            overlap_e = min(msg_e, evict_end)
            if overlap_s > msg_s or overlap_e < msg_e:
                # Replace last bar segment with split
                bar_segments.pop()
                if overlap_s > msg_s:
                    bar_segments.append(("retained", overlap_s - msg_s, ""))
                bar_segments.append(("evicted", overlap_e - overlap_s, ""))
                if overlap_e < msg_e:
                    bar_segments.append(("retained", msg_e - overlap_e, ""))

        # Full message block
        role = mb["role"].upper()
        content_escaped = html.escape(mb.get("content", ""))
        status_label = {
            "protected": "PROTECTED",
            "retained": "IN CACHE",
            "evicted": "EVICTED",
            "future": "FUTURE",
        }[status]
        msg_blocks.append(
            f'<div class="kv-msg-block {status}">'
            f'<div class="kv-msg-header">'
            f'<span class="kv-msg-role">{role}</span>'
            f'<span class="kv-msg-tokens">{mb["len"]} tokens</span>'
            f'<span class="kv-msg-range">[{mb["start"]}, {mb["end"]})</span>'
            f'<span class="kv-msg-status {status}">{status_label}</span>'
            f'</div>'
            f'<div class="kv-msg-content">{content_escaped}</div>'
            f'</div>'
        )

    # Render bar
    bar_html = []
    for cls, tok_len, label in bar_segments:
        pct = max(0.3, tok_len / total_tokens * 100)
        bar_html.append(
            f'<div class="kv-seg {cls}" style="width:{pct:.1f}%;" '
            f'title="{tok_len} tokens">{label}</div>'
        )

    # Summary line
    summary = ""
    if offset > 0:
        summary = (
            f' — <span style="color:#e74c3c">{offset} tokens evicted</span> '
            f'from token positions [{evict_start}, {evict_end})'
        )

    return (
        '<div class="kv-wrap">'
        f'<div class="kv-title">KV CACHE STATE AT TURN {turn + 1}{summary}</div>'
        '<div class="kv-source">Tokens from Qwen3 tokenizer applied to the full message sequence. '
        'Eviction removes the oldest conversation turns after the system prompt.</div>'
        f'<div class="kv-bar">{"".join(bar_html)}</div>'
        '<div class="kv-legend">'
        '<span><span class="dot" style="background:#1a5c2a"></span> Protected (system prompt)</span>'
        '<span><span class="dot" style="background:#1a3a5a"></span> Retained in KV cache</span>'
        '<span><span class="dot" style="background:#5a1a1a"></span> Evicted from KV cache</span>'
        '<span><span class="dot" style="background:#222"></span> Future turns</span>'
        '</div>'
        f'<div class="kv-msgs">{"".join(msg_blocks)}</div>'
        '</div>'
    )


def render_tokens_html(input_ids_str: str | list) -> str:
    """Render raw token IDs as flowing text with per-token hover tooltips.

    Special tokens appear as visible badges; newlines force a line break;
    regular tokens are shown as the decoded text with alternating faint
    background so boundaries are visible. Hover over any token to see its ID.
    """
    if not input_ids_str:
        return "<div style='color:#666;font-family:monospace;font-size:12px;'>No input_ids available.</div>"
    try:
        ids = json.loads(input_ids_str) if isinstance(input_ids_str, str) else list(input_ids_str)
    except (json.JSONDecodeError, TypeError):
        return f"<div style='color:#e74c3c;font-family:monospace;'>Could not parse input_ids: {html.escape(str(input_ids_str)[:200])}</div>"

    tok = get_tokenizer()
    decoded = [tok.decode([i], skip_special_tokens=False) for i in ids]

    parts = []
    alt = False  # alternating background for token boundaries
    for pos, (tid, text) in enumerate(zip(ids, decoded)):
        title = f"pos={pos} id={tid}"
        if tid == IM_START_ID:
            parts.append(
                f'<br><span style="display:inline-block;margin:2px 1px;padding:1px 6px;'
                f'border-radius:3px;background:#3a1a4a;color:#c084fc;font-family:monospace;'
                f'font-size:12px;font-weight:bold;" title="{title}">&lt;|im_start|&gt;</span>'
            )
            alt = False
            continue
        if tid == IM_END_ID:
            parts.append(
                f'<span style="display:inline-block;margin:2px 1px;padding:1px 6px;'
                f'border-radius:3px;background:#2a0a3a;color:#a855f7;font-family:monospace;'
                f'font-size:12px;font-weight:bold;" title="{title}">&lt;|im_end|&gt;</span><br>'
            )
            alt = False
            continue
        if text == "\n":
            parts.append(f'<span title="{title}" style="color:#444;font-size:11px;">↵</span><br>')
            alt = False
            continue
        if text.strip() == "":
            # Pure whitespace — show as faint dot so boundary is still visible
            esc = html.escape(text)
            parts.append(
                f'<span style="white-space:pre;color:#444;font-family:monospace;font-size:13px;" '
                f'title="{title}">{esc}</span>'
            )
            continue

        bg = "#1e2a38" if alt else "#141e28"
        alt = not alt
        esc = html.escape(text)
        parts.append(
            f'<span style="white-space:pre;background:{bg};color:#b0d0f0;font-family:monospace;'
            f'font-size:13px;border-radius:2px;padding:0 1px;" title="{title}">{esc}</span>'
        )

    total = len(ids)
    body = "".join(parts)
    return (
        f'<div style="margin-bottom:6px;font-family:monospace;font-size:11px;color:#888;">'
        f'{total} tokens &nbsp;·&nbsp; hover any token to see its position and ID</div>'
        f'<div style="max-height:500px;overflow-y:auto;padding:10px;background:#0d1117;'
        f'border:1px solid #2a2a2a;border-radius:6px;line-height:2;">{body}</div>'
    )


def render_system_prompt_html(system_prompt: str) -> str:
    """Render the system prompt (truncated)."""
    if not system_prompt:
        return ""
    truncated = system_prompt[:600] + ("\n... (truncated)" if len(system_prompt) > 600 else "")
    return (
        '<div class="msg-wrap">'
        '<div class="msg msg-system">'
        '<div class="msg-role">SYSTEM</div>'
        f'<div class="msg-content">{html.escape(truncated)}</div>'
        '</div></div>'
    )


def load_samples(run_path: str = None, local_path: str = None) -> list[dict]:
    """Load samples from wandb or local JSON."""
    if local_path:
        with open(local_path) as f:
            data = json.load(f)
    else:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        art_name = f"run-{run.id}-final-samples:latest"
        art = api.artifact(f"{run.entity}/{run.project}/{art_name}")
        path = art.download()
        table_file = Path(path) / "final-samples.table.json"
        with open(table_file) as f:
            data = json.load(f)

    columns = data["columns"]
    samples = []
    for row in data["data"]:
        sample = dict(zip(columns, row))
        sample["messages_parsed"] = parse_messages(sample.get("messages", ""))
        system_prompt, turns = messages_to_turns(sample["messages_parsed"])
        sample["system_prompt"] = system_prompt
        sample["turns"] = turns

        # Parse per-step compaction events if present.
        raw_ce = sample.get("compaction_events")
        if raw_ce and isinstance(raw_ce, str):
            try:
                sample["compaction_events_parsed"] = json.loads(raw_ce)
            except json.JSONDecodeError:
                sample["compaction_events_parsed"] = None
        else:
            sample["compaction_events_parsed"] = None

        # Compute per-message token boundaries for eviction visualization.
        try:
            sample["msg_boundaries"] = compute_message_boundaries(
                sample.get("messages", "")
            )
        except Exception:
            sample["msg_boundaries"] = None

        samples.append(sample)
    return samples


def build_app(samples: list[dict]) -> gr.Blocks:
    # Build rollout labels
    labels = []
    for i, s in enumerate(samples):
        reward = s.get("reward", 0)
        step = s.get("step", "?")
        n_turns = len(s["turns"])
        actions = [t["action"] for t in s["turns"] if t["action"]]
        pse = s.get("compaction_events_parsed")
        compact_tag = ""
        if pse:
            n_compact = sum(1 for e in pse if e)
            total_evicted = sum(sum(ev.get("evicted", 0) for ev in e) for e in pse if e)
            if n_compact:
                compact_tag = f" | compact={n_compact}x/{total_evicted}tok"
        labels.append(
            f"[step {step}] #{s.get('example_id', i)} | "
            f"reward={reward:.1f} | turns={n_turns} | "
            f"actions={len(actions)}{compact_tag}"
        )

    with gr.Blocks(title="BabyAI Training Rollout Viewer") as demo:
        gr.Markdown(
            "# BabyAI Training Rollout Viewer\n"
            "> **Observation** (left): what the agent sees each turn. "
            "**Response** (right): model reasoning + action. "
            "**Timeline** (bottom): full action sequence for the episode."
        )

        with gr.Row():
            rollout_dd = gr.Dropdown(
                choices=labels,
                value=labels[0] if labels else None,
                label="Rollout",
                scale=3,
            )
            turn_sl = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Turn", scale=2,
            )

        ep_bar = gr.Markdown("")

        compact_html = gr.HTML(label="Compaction")
        kv_html = gr.HTML(label="KV Cache State")

        with gr.Row():
            obs_html = gr.HTML(label="Observation")
            resp_html = gr.HTML(label="Model Response")

        timeline_html = gr.HTML(label="Action Timeline")
        system_html = gr.HTML(label="System Prompt")

        with gr.Accordion("Full conversation (all messages)", open=False):
            full_msgs_html = gr.HTML()

        with gr.Accordion("Raw tokens (input_ids)", open=False):
            tokens_html = gr.HTML()

        def on_rollout(choice):
            if not choice:
                return gr.update(maximum=0, value=0), "", "", "", "", "", "", "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            n = len(turns)
            pse = s.get("compaction_events_parsed")

            actions = [t["action"] for t in turns if t["action"]]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            action_summary = ", ".join(f"{v}x {k}" for k, v in action_counts.items())

            # Episode-level compaction summary
            compact_summary = ""
            if pse:
                total_evictions = sum(1 for step_e in pse if step_e)
                total_evicted = sum(
                    sum(e.get("evicted", 0) for e in step_e)
                    for step_e in pse if step_e
                )
                compact_summary = f" | **Compactions:** {total_evictions} turns, {total_evicted} tokens evicted"

            bar = (
                f"**Step:** {s.get('step', '?')} | "
                f"**Example:** {s.get('example_id', '?')} | "
                f"**Task:** {s.get('task', '?')} | "
                f"**Reward:** {s.get('reward', 0):.2f} | "
                f"**Turns:** {n} | "
                f"**Actions:** {action_summary or 'none'}"
                f"{compact_summary}"
            )

            max_turn = max(0, n - 1)

            # Render turn 0
            obs = ""
            resp = ""
            tl = ""
            cb = ""
            kv = ""
            if turns:
                obs = render_observation_html(turns[0]["obs"])
                resp = render_response_html(turns[0]["response"])
                tl = render_action_timeline(turns, 0, pse)
                if pse and len(pse) > 0:
                    cb = render_compaction_bar(pse[0])
                kv = render_kv_cache_state(s.get("msg_boundaries"), pse, 0, n)

            sys_html = render_system_prompt_html(s["system_prompt"])

            # Full conversation
            full = render_all_messages_html(s["messages_parsed"])

            # Raw tokens
            tokens = render_tokens_html(s.get("input_ids", ""))

            return (
                gr.update(maximum=max_turn, value=0),
                bar, cb, kv, obs, resp, tl, sys_html, full, tokens,
            )

        def on_turn(choice, turn):
            if not choice:
                return "", "", "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            pse = s.get("compaction_events_parsed")
            t = int(turn)
            if t >= len(turns):
                return "", "", "", "", ""

            obs = render_observation_html(turns[t]["obs"])
            resp = render_response_html(turns[t]["response"])
            tl = render_action_timeline(turns, t, pse)
            cb = ""
            if pse and t < len(pse):
                cb = render_compaction_bar(pse[t])
            kv = render_kv_cache_state(
                s.get("msg_boundaries"), pse, t, len(turns),
            )
            return cb, kv, obs, resp, tl

        rollout_dd.change(
            on_rollout, [rollout_dd],
            [turn_sl, ep_bar, compact_html, kv_html, obs_html, resp_html, timeline_html, system_html, full_msgs_html, tokens_html],
        )
        turn_sl.change(
            on_turn, [rollout_dd, turn_sl],
            [compact_html, kv_html, obs_html, resp_html, timeline_html],
        )
        demo.load(
            on_rollout, [rollout_dd],
            [turn_sl, ep_bar, compact_html, kv_html, obs_html, resp_html, timeline_html, system_html, full_msgs_html, tokens_html],
        )

    return demo


def render_all_messages_html(messages: list[dict]) -> str:
    """Render all messages as styled HTML (full conversation view)."""
    blocks = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        css_class = f"msg-{role}" if role in ("system", "user", "assistant", "tool") else "msg-user"

        escaped = html.escape(content)

        if role == "assistant":
            thinking, rest = extract_thinking(content)
            action = extract_action(content)

            parts = []
            if thinking:
                parts.append(
                    f'<div class="think">{html.escape(thinking[:500])}'
                    f'{"..." if len(thinking) > 500 else ""}</div>'
                )
            rest_escaped = html.escape(rest)
            rest_escaped = re.sub(
                r"&lt;tool_call&gt;(.*?)&lt;/tool_call&gt;",
                r'<div class="tool-call">\1</div>',
                rest_escaped,
                flags=re.DOTALL,
            )
            parts.append(rest_escaped)
            if action:
                parts.append(f'<div class="action-tag">Action: {html.escape(action)}</div>')
            escaped = "\n".join(parts)
        elif role == "system":
            if len(escaped) > 600:
                escaped = escaped[:600] + "\n... (truncated)"

        block = (
            f'<div class="msg {css_class}">'
            f'<div class="msg-role">{role.upper()}</div>'
            f'<div class="msg-content">{escaped}</div>'
            f'</div>'
        )
        blocks.append(block)

    return f'<div class="msg-wrap">{"".join(blocks)}</div>'


def main():
    parser = argparse.ArgumentParser(description="Visualize training rollouts from wandb")
    parser.add_argument("run_path", nargs="?", help="wandb run path (entity/project/run_id)")
    parser.add_argument("--local", help="Path to local .table.json file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.run_path and not args.local:
        # Auto-discover from kv-eviction output dirs
        patterns = [
            Path("/tmp/kv-eviction").glob("*/run_default/wandb/*/files/media/table/*.table.json"),
            Path("/tmp/kv-eviction").glob("*/wandb/*/files/media/table/*.table.json"),
        ]
        found = []
        for pat in patterns:
            found.extend(sorted(pat, key=lambda p: p.stat().st_mtime, reverse=True))
        if found:
            args.local = str(found[0])
            print(f"Auto-discovered: {args.local}")
        else:
            print("Usage: viz_training_rollouts.py <wandb_run_path>")
            print("   or: viz_training_rollouts.py --local /path/to/samples.table.json")
            return

    print("Loading samples...")
    samples = load_samples(run_path=args.run_path, local_path=args.local)
    print(f"Loaded {len(samples)} rollout samples")

    demo = build_app(samples)
    dark_theme = gr.themes.Base(primary_hue="blue").set(
        body_background_fill="#111",
        body_text_color="#eee",
        block_background_fill="#1a1a1a",
        block_border_color="#333",
        input_background_fill="#222",
        button_primary_background_fill="#2563eb",
    )
    demo.launch(server_port=args.port, share=args.share, css=CSS, theme=dark_theme)


if __name__ == "__main__":
    main()
