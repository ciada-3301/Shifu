"""
tools/python_repl.py — Python REPL tool for Shifu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop in tools/ alongside every other tool.
Auto-discovered by _load_tools() at boot.

Features
────────
• Persistent namespace across calls in the same Shifu session — variables
  set in one call are available in the next.
• Captures both stdout/stderr *and* the value of the last expression
  (Jupyter / IPython cell semantics).
• 30-second hard timeout per execution (configurable via REPL_TIMEOUT env var).
• Rich exception output with full tracebacks for easy debugging.
• Reset sentinel: pass code='__reset__' to wipe the namespace.
• Automation-friendly — usable from YAML action args as tool: python_repl.

Example automator instruction
──────────────────────────────
  "At 9 AM every weekday, run this snippet to check disk usage and
   append the result to Playground/disk_report.txt"

YAML usage
──────────
  actions:
    - tool: python_repl
      args:
        code: |
          import shutil
          t, u, f = shutil.disk_usage('/')
          print(f'Free: {f // 2**30} GB')
"""

from __future__ import annotations

import ast
import io
import os
import textwrap
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ── Persistent namespace ───────────────────────────────────────────────────────
# Lives for the lifetime of the Shifu process.  Thread-safe write lock.

_NAMESPACE: dict = {}
_NS_LOCK          = threading.Lock()

_REPL_TIMEOUT = int(os.getenv("REPL_TIMEOUT", "30"))


# ── Timeout helper ─────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _run_with_timeout(fn, timeout: int):
    result_box: list = [None]
    exc_box:    list = [None]

    def _target():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _TimeoutError(f"Execution exceeded the {timeout}s timeout")
    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


# ── Execution core ─────────────────────────────────────────────────────────────

def _execute(code: str) -> str:
    """
    Execute `code` in the persistent namespace.
    Returns stdout/stderr + final expression value (Jupyter-style).
    """
    code = textwrap.dedent(code).strip()

    # ── Reset sentinel ─────────────────────────────────────────────────────────
    if code == "__reset__":
        with _NS_LOCK:
            _NAMESPACE.clear()
        return "🔄  REPL namespace reset — all variables cleared."

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    last_expr_value: Optional[str] = None

    def _run():
        nonlocal last_expr_value

        # Parse the code — may raise SyntaxError
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            raise

        # If the final statement is a bare expression, pop and eval it
        # so its repr is returned automatically (Jupyter cell semantics)
        last_expr_src: Optional[str] = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_node = tree.body.pop()
            last_expr_src = ast.unparse(last_node.value)

        with _NS_LOCK:
            ns = _NAMESPACE

        # Execute the main body
        if tree.body:
            exec(compile(tree, "<repl>", "exec"), ns)  # noqa: S102

        # Evaluate the last expression
        if last_expr_src is not None:
            val = eval(compile(last_expr_src, "<repl_expr>", "eval"), ns)  # noqa: S307
            if val is not None:
                last_expr_value = repr(val)

    # ── Run with timeout ───────────────────────────────────────────────────────
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            _run_with_timeout(_run, _REPL_TIMEOUT)

    except _TimeoutError as e:
        return f"⏱  TimeoutError: {e}"

    except SyntaxError as e:
        return (
            f"🔴  SyntaxError: {e}\n"
            f"   line {e.lineno}: {(e.text or '').rstrip()}"
        )

    except Exception:
        tb         = traceback.format_exc()
        stdout_out = stdout_buf.getvalue()
        parts      = []
        if stdout_out:
            parts.append(stdout_out.rstrip())
        parts.append(f"🔴  Exception:\n{tb.rstrip()}")
        return "\n".join(parts)

    # ── Assemble clean output ──────────────────────────────────────────────────
    parts      = []
    stdout_out = stdout_buf.getvalue()
    stderr_out = stderr_buf.getvalue()

    if stdout_out:
        parts.append(stdout_out.rstrip())
    if stderr_out:
        parts.append(f"[stderr]\n{stderr_out.rstrip()}")
    if last_expr_value is not None:
        parts.append(f"→  {last_expr_value}")

    return "\n".join(parts) if parts else "✓  (no output)"


# ══════════════════════════════════════════════════════════════════════════════
#  LangChain BaseTool
# ══════════════════════════════════════════════════════════════════════════════

class _ReplInput(BaseModel):
    code: str = Field(
        description=(
            "Python code to execute. Multi-line is fine — use real newlines. "
            "Variables persist across calls in the same Shifu session. "
            "The value of the last expression is returned automatically (like Jupyter). "
            "Pass '__reset__' to wipe all session variables."
        )
    )


class PythonReplTool(BaseTool):
    """
    Execute Python code in a persistent in-process REPL.

    Use this for:
    • Running calculations, data transforms, or file manipulation.
    • Verifying snippets of code the user wants tested.
    • Multi-step computations across separate calls (variables persist).
    • Scripted automation actions that need full Python power.

    stdout and stderr are captured and returned. The last bare expression
    is returned automatically (Jupyter-cell semantics). Execution runs in a
    timeout-guarded thread (default 30 s, set REPL_TIMEOUT env var to override).

    To wipe all session variables between runs, pass code='__reset__'.
    """

    name:          str             = "python_repl"
    description:   str             = (
        "Execute Python code and return the output. "
        "Variables persist across calls in the same session. "
        "Last expression value is returned automatically (Jupyter-style). "
        "Use for calculations, file I/O, data work, or any Python task. "
        "Pass code='__reset__' to clear all session state."
    )
    args_schema:   Type[BaseModel] = _ReplInput
    return_direct: bool            = False

    def _run(self, code: str) -> str:
        return _execute(code)

    async def _arun(self, code: str) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, code)


# ── Singleton export (auto-discovered by _load_tools) ─────────────────────────

python_repl = PythonReplTool()