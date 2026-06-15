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

from cube_tools import CubeSession, NOTATION, MoveParseError  # noqa: E402


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

SYSTEM_PROMPT = (
    "You are solving a 3x3 Rubik's cube. The cube state is tracked for you — you "
    "do not need to remember it; just read it from tool results.\n\n"
    "Notation: faces are U (up), D (down), L (left), R (right), F (front), B (back). "
    "A bare letter = 90 deg clockwise; a letter+2 = 180 deg (e.g. R2); a letter+' = "
    "90 deg counter-clockwise (e.g. R'). Valid moves: " + ', '.join(NOTATION) + ".\n\n"
    "The cube is SOLVED when all 20 pieces are home (pieces_solved = 20, every face "
    "one color). Your goal: reach pieces_solved = 20.\n\n"
    "You CANNOT read the net reliably — do not try to reason about colors. Instead "
    "use the rank_moves INTUITION tool.\n\n"
    "Tools: rank_moves (ranked next-move suggestions from a learned intuition; lower "
    "estimated_moves_to_solve = closer to solved; would_solve means it solves now), "
    "apply (commit moves), simulate (preview without committing), observe, inverse.\n\n"
    "STRATEGY (follow this loop):\n"
    "1. Call rank_moves.\n"
    "2. If any move shows would_solve, apply that move — you win.\n"
    "3. Otherwise apply the TOP-ranked move (lowest estimate).\n"
    "4. Repeat. apply is the only tool that changes the real cube.\n"
    "If the estimate stops improving for several moves (a plateau), try the 2nd or "
    "3rd ranked move to escape it.\n"
    "When pieces_solved reaches 20, say SOLVED."
)


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
                log=lambda *_: None) -> dict:
    """Run one solve attempt. Returns a result dict."""
    obs = session.observe()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            msg = ollama_chat(messages, TOOLS, model, host, think=think)
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
    ap.add_argument("--out", default="", help="optional transcript log file")
    args = ap.parse_args()

    logf = open(args.out, "a") if args.out else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    log(f"=== LLM cube agent | model={args.model} host={args.host} "
        f"depth={args.depth} episodes={args.episodes} ===")
    results = []
    for ep in range(args.episodes):
        session = CubeSession(seed=args.seed + ep)
        obs = session.scramble(args.depth)
        log(f"\n--- episode {ep} | scramble: {obs['scramble']} "
            f"(pieces_solved={obs['pieces_solved']}) ---")
        res = run_episode(session, args.model, args.host,
                          max_turns=args.max_turns, think=args.think, log=log)
        log(f"--- episode {ep} result: {res} ---")
        results.append(res)

    n_solved = sum(1 for r in results if r["solved"])
    log(f"\n=== SUMMARY: solved {n_solved}/{len(results)} | "
        f"avg turns {sum(r['turns'] for r in results)/len(results):.1f} | "
        f"avg secs {sum(r['secs'] for r in results)/len(results):.1f} ===")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
