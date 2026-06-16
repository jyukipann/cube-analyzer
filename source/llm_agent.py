"""LLM cube-solving agent harness (M0).

Drives a local Ollama model (e.g. qwen3.5:4b) through the cube_tools kernel using
native function/tool calling.  The cube state is ground truth in the kernel; the
model observes and acts via tools.  One episode = scramble -> agent tries to
solve within a move/turn budget.

Usage
-----
    uv run python source/llm_agent.py --depth 1 --episodes 1 --max-turns 12
    uv run python source/llm_agent.py --host 192.168.10.62:11434 --model qwen3.5:4b

This M0 version exposes observe / apply / simulate / inverse.  Memory and the
learned-distance "intuition" tool are added in later milestones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cube_tools import CubeSession, MacroMemory, NOTATION, MoveParseError  # noqa: E402


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Ollama function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Show the current cube as an unfolded net plus how many "
                           "of the 20 pieces are already solved (home position+orientation).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply",
            "description": "Apply (commit) a sequence of moves to the cube and return "
                           "the new net and solved-piece count. Use space-separated "
                           "moves like \"R U R' U'\".",
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {"type": "string",
                              "description": "space-separated moves, e.g. R U2 F'"},
                },
                "required": ["moves"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Try a sequence on a scratch copy WITHOUT committing. "
                           "Returns the resulting net and the change in solved-piece "
                           "count, so you can test an idea safely before applying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {"type": "string"},
                },
                "required": ["moves"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inverse",
            "description": "Return the move sequence that exactly undoes the given moves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {"type": "string"},
                },
                "required": ["moves"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_moves",
            "description": "Learned INTUITION. Returns each possible next move ranked "
                           "by estimated_moves_to_solve (lower = closer to solved). The "
                           "FIRST move is the recommended one. would_solve=true means that "
                           "move solves the cube right now. Use this to choose moves — you "
                           "cannot read the net reliably, but this intuition can.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Macro-memory tools (M2). Appended to the intuition tools in 'memory' mode.
MACRO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_macro",
            "description": "Save a named move sequence (a 'macro') so you can reuse it "
                           "later, across cubes. Use this when a short sequence reliably "
                           "made progress or escaped a plateau.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "moves": {"type": "string", "description": "e.g. R U R' U'"},
                    "note": {"type": "string", "description": "what it does / when to use"},
                },
                "required": ["name", "moves"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_macros",
            "description": "List all saved macros (name, moves, note).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_macro",
            "description": "Preview a saved macro's effect on the current cube WITHOUT "
                           "committing (shows delta_pieces_solved).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_macro",
            "description": "Commit a saved macro's moves to the cube.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

_COMMON = (
    "You are solving a 3x3 Rubik's cube. The cube state is tracked for you.\n"
    "Notation: faces U D L R F B. Bare letter = 90 deg CW; +2 = 180 deg (R2); "
    "+' = 90 deg CCW (R'). Valid moves: " + ', '.join(NOTATION) + ".\n"
    "SOLVED = all 20 pieces home (pieces_solved = 20). Goal: reach pieces_solved = 20.\n"
)

INTUITION_PROMPT = _COMMON + (
    "\nYou CANNOT read the net reliably — do not reason about colors. Use the "
    "rank_moves INTUITION tool instead.\n"
    "Tools: rank_moves (next moves ranked by a learned intuition; lower "
    "estimated_moves_to_solve = closer; would_solve means it solves now), apply "
    "(commit moves), simulate (preview), observe, inverse.\n"
    "STRATEGY:\n"
    "1. Call rank_moves.\n"
    "2. If a move shows would_solve, apply it — you win.\n"
    "3. Otherwise apply the TOP-ranked move (lowest estimate).\n"
    "4. Repeat. Only apply changes the real cube.\n"
    "If the estimate stalls for several moves (plateau), try the 2nd/3rd ranked move.\n"
    "When pieces_solved reaches 20, say SOLVED."
)

BLIND_PROMPT = _COMMON + (
    "\nHOW THE CUBE WORKS (basics):\n"
    "- The cube has 6 faces. The observation shows them as colors: W=white, "
    "Y=yellow, G=green, B=blue, R=red, O=orange.\n"
    "- Each face has a fixed CENTER that never moves; the center is that face's "
    "target color. The cube is SOLVED when every face is one solid color matching "
    "its center (white up, yellow down, green front, blue back, red right, orange "
    "left).\n"
    "- A move turns ONE face 90 degrees. It only moves the stickers on that face "
    "and the strips touching it; centers stay put. So pick the move that brings a "
    "sticker toward the face whose center matches its color.\n"
    "- Order matters: R then U is NOT the same as U then R.\n"
    "- Every move is undone by its inverse: R then R' changes nothing; doing the "
    "same quarter-turn 4 times returns to start. Use this to back out of a bad line.\n"
    "- To place one piece you often must briefly disturb others and then restore "
    "them. Short repeatable sequences (like R U R' U') cycle a few pieces while "
    "leaving most intact.\n\n"
    "Tools: observe (see the colored net + how many stickers per face match its "
    "center), simulate (preview moves on a scratch copy without committing; shows "
    "delta_pieces_solved), apply (commit moves), inverse (undo a sequence).\n"
    "simulate does NOT change the real cube — only apply does. Strategy: read the "
    "cube, use simulate to find moves with delta_pieces_solved > 0, APPLY them, "
    "repeat. When pieces_solved reaches 20, say SOLVED."
)

MEMORY_PROMPT = INTUITION_PROMPT + (
    "\n\nYou ALSO have a persistent macro memory shared across cubes:\n"
    "- save_macro(name, moves, note): remember a useful short sequence.\n"
    "- list_macros / test_macro(name) / apply_macro(name).\n"
    "When the intuition plateaus (top estimate stops dropping), a short repeatable "
    "sequence (a 'macro', e.g. R U R' U') can shuffle pieces to escape it. If you "
    "find one that helps, SAVE it; on later cubes, list_macros and test_macro to "
    "reuse it. Building a good macro library makes you faster over time."
)

# Backwards-compatible default
SYSTEM_PROMPT = INTUITION_PROMPT


def tools_for_mode(mode: str) -> list[dict]:
    """intuition = rank_moves; blind = no rank_moves; memory = intuition + macros."""
    if mode == "blind":
        return [t for t in TOOLS if t["function"]["name"] != "rank_moves"]
    if mode == "memory":
        return TOOLS + MACRO_TOOLS
    return TOOLS


def prompt_for_mode(mode: str) -> str:
    return {"blind": BLIND_PROMPT, "memory": MEMORY_PROMPT}.get(mode, INTUITION_PROMPT)


# ---------------------------------------------------------------------------
# Ollama chat
# ---------------------------------------------------------------------------

def ollama_chat(messages: list[dict], tools: list[dict], model: str,
                host: str, think: bool = False, timeout: int = 600) -> dict:
    """One non-streaming /api/chat call. Returns the assistant message dict."""
    url = f"http://{host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "think": think,
        "options": {"temperature": 0.7},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body.get("message", {})


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def dispatch(session: CubeSession, name: str, args: dict) -> dict:
    try:
        if name == "observe":
            return session.observe()
        if name == "apply":
            return session.apply(args.get("moves", ""))
        if name == "simulate":
            return session.simulate(args.get("moves", ""))
        if name == "inverse":
            return session.inverse(args.get("moves", ""))
        if name == "rank_moves":
            return session.rank_moves()
        if name == "save_macro":
            return session.save_macro(args.get("name", ""), args.get("moves", ""),
                                      args.get("note", ""))
        if name == "list_macros":
            return session.list_macros()
        if name == "get_macro":
            return session.get_macro(args.get("name", ""))
        if name == "test_macro":
            return session.test_macro(args.get("name", ""))
        if name == "apply_macro":
            return session.apply_macro(args.get("name", ""))
        return {"error": f"unknown tool '{name}'"}
    except MoveParseError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _fmt_obs(d: dict) -> str:
    """Compact text form of a tool result (keeps the net readable, drops noise)."""
    # Special, token-light formatting for the rank_moves intuition tool.
    if "ranked_moves" in d:
        rows = d["ranked_moves"][:6]
        lines = [f"current_estimate={d.get('current_estimate')}  (top moves, lower=better):"]
        for r in rows:
            tag = " <-SOLVES" if r.get("would_solve") else ""
            lines.append(f"  {r['move']:>3}: est={r['estimated_moves_to_solve']} "
                         f"pieces={r['pieces_solved']}{tag}")
        return "\n".join(lines)
    net = d.pop("net", None) or d.pop("net_after", None)
    parts = [f"{k}={v}" for k, v in d.items()]
    s = "  ".join(parts)
    if net:
        s = net + "\n" + s
    return s


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(session: CubeSession, model: str, host: str,
                max_turns: int = 15, think: bool = False,
                mode: str = "intuition", log=lambda *_: None) -> dict:
    """Run one solve attempt. Returns a result dict."""
    tools = tools_for_mode(mode)
    obs = session.observe()
    messages = [
        {"role": "system", "content": prompt_for_mode(mode)},
        {"role": "user", "content":
            "Here is the scrambled cube. Solve it.\n" + _fmt_obs(dict(obs))},
    ]

    turns = 0
    tool_calls_made = 0
    t0 = time.time()
    while turns < max_turns:
        if session.observe()["is_solved"]:
            break
        turns += 1
        try:
            msg = ollama_chat(messages, tools, model, host, think=think)
        except Exception as exc:  # noqa: BLE001
            log(f"[turn {turns}] chat error: {exc}")
            return {"solved": False, "turns": turns, "error": str(exc),
                    "tool_calls": tool_calls_made, "secs": time.time() - t0}

        messages.append(msg)
        content = (msg.get("content") or "").strip()
        if content:
            log(f"[turn {turns}] assistant: {content[:300]}")

        tcs = msg.get("tool_calls") or []
        if not tcs:
            # No tool call. If it claims solved, verify; else nudge.
            if "SOLVED" in content.upper() and session.observe()["is_solved"]:
                break
            messages.append({"role": "user", "content":
                "Use a tool (observe/simulate/apply/inverse) to make progress. "
                "Apply moves to reduce the cube toward solved."})
            continue

        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"moves": args}
            result = dispatch(session, name, args)
            tool_calls_made += 1
            log(f"[turn {turns}] tool {name}({args}) -> "
                f"pieces_solved={result.get('pieces_solved', result.get('pieces_solved_after','?'))}"
                f"{' SOLVED' if result.get('is_solved') else ''}")
            messages.append({"role": "tool", "tool_name": name,
                             "content": _fmt_obs(dict(result))})
            if result.get("is_solved"):
                break

    solved = session.observe()["is_solved"]
    return {
        "solved": solved,
        "turns": turns,
        "tool_calls": tool_calls_made,
        "moves_made": len(session.history),
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.10.62:11434")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--depth", type=int, default=1, help="scramble depth")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="enable model thinking")
    ap.add_argument("--mode", choices=["intuition", "blind", "memory"],
                    default="intuition",
                    help="intuition = rank_moves; blind = none; memory = + macro store")
    ap.add_argument("--macro-file", default="runs/llm/macros.json",
                    help="persistent macro memory path (mode=memory)")
    ap.add_argument("--out", default="", help="optional transcript log file")
    args = ap.parse_args()

    logf = open(args.out, "a") if args.out else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    log(f"=== LLM cube agent | model={args.model} mode={args.mode} "
        f"depth={args.depth} episodes={args.episodes} ===")
    # Shared, persistent macro memory across episodes (memory mode only).
    memory = MacroMemory(args.macro_file) if args.mode == "memory" else None
    if memory is not None:
        log(f"macro memory: {args.macro_file} ({len(memory.macros)} macros loaded)")

    results = []
    for ep in range(args.episodes):
        session = CubeSession(seed=args.seed + ep, memory=memory)
        obs = session.scramble(args.depth)
        log(f"\n--- episode {ep} | scramble: {obs['scramble']} "
            f"(pieces_solved={obs['pieces_solved']}) ---")
        res = run_episode(session, args.model, args.host,
                          max_turns=args.max_turns, think=args.think,
                          mode=args.mode, log=log)
        log(f"--- episode {ep} result: {res} ---")
        results.append(res)

    n_solved = sum(1 for r in results if r["solved"])
    log(f"\n=== SUMMARY {args.model} mode={args.mode} depth={args.depth}: "
        f"solved {n_solved}/{len(results)} | "
        f"avg turns {sum(r['turns'] for r in results)/len(results):.1f} | "
        f"avg secs {sum(r['secs'] for r in results)/len(results):.1f} ===")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
