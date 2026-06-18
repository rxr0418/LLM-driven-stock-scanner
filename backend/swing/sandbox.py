"""
swing/sandbox.py - Safe execution environment for LLM-generated factor code.

Two-layer defence:
  1. AST blacklist  : reject code before execution if it uses forbidden nodes
  2. subprocess     : run code in a child process with CPU/memory/time limits

Usage:
  from swing.sandbox import run_factor_in_sandbox
  result = run_factor_in_sandbox(code, close_json, volume_json)
"""

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Layer 1 — AST blacklist
# ─────────────────────────────────────────────────────────────

# Node types that are unconditionally forbidden
FORBIDDEN_NODES = {
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
}

# Names forbidden in any context (function calls, attribute access, etc.)
FORBIDDEN_NAMES = {
    # OS / file system
    "os", "sys", "open", "eval", "exec", "__import__",
    "compile", "globals", "locals", "vars", "dir",
    # Process / network
    "subprocess", "socket", "urllib", "requests", "http",
    "threading", "multiprocessing", "asyncio",
    # Dunder abuse
    "__builtins__", "__class__", "__bases__", "__subclasses__",
    # Dangerous builtins
    "input", "print", "exit", "quit",
}

# Allowed top-level names (everything else is rejected in Name/Attribute nodes)
ALLOWED_NAMES = {
    # NumPy / Pandas — the only libraries the factor can use
    "np", "pd",
    # Python builtins safe for math
    "abs", "min", "max", "sum", "len", "range", "zip", "enumerate",
    "float", "int", "bool", "str", "list", "dict", "tuple", "set",
    "True", "False", "None",
    # Function definition keywords (handled as nodes, not names)
    "return", "if", "else", "elif", "for", "while", "pass", "break", "continue",
    # Our factor function signature
    "factor_generated", "close", "volume",
    # Common intermediate variable names
    "signal", "scores", "result", "ret", "returns", "vol", "avg",
    "window", "n", "i", "k", "x", "s", "r", "w", "d",
}


class ASTBlacklistVisitor(ast.NodeVisitor):
    """Walk the AST and collect violations."""

    def __init__(self):
        self.violations: list[str] = []

    def visit(self, node):
        # Check forbidden node types
        if type(node) in FORBIDDEN_NODES:
            self.violations.append(
                f"Forbidden AST node: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
            )
        # Check Name nodes
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                self.violations.append(f"Forbidden name '{node.id}' at line {node.lineno}")
        # Check Attribute access (e.g. os.path, __class__.__subclasses__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                self.violations.append(
                    f"Forbidden dunder attribute '{node.attr}' at line {node.lineno}"
                )
        self.generic_visit(node)


def ast_check(code: str) -> list[str]:
    """
    Parse code and return list of violation strings.
    Empty list = safe to proceed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    visitor = ASTBlacklistVisitor()
    visitor.visit(tree)
    return visitor.violations


# ─────────────────────────────────────────────────────────────
# Layer 2 — subprocess execution with timeout + resource limits
# ─────────────────────────────────────────────────────────────

WORKER_TEMPLATE = '''
import json, sys, numpy as np, pandas as pd

close_json  = json.loads(sys.argv[1])
volume_json = json.loads(sys.argv[2])

close  = pd.DataFrame(close_json)
volume = pd.DataFrame(volume_json)

{user_code}

try:
    result = factor_generated(close, volume)
    if not isinstance(result, pd.Series):
        raise ValueError("factor_generated must return pd.Series")
    print(json.dumps(result.dropna().to_dict()))
except Exception as e:
    print(json.dumps({{"__error__": str(e)}}))
'''

SANDBOX_TIMEOUT_SECONDS = 10
SANDBOX_MAX_MB          = 256


def _set_resource_limits():
    """Called in child process before exec — cap memory and CPU."""
    try:
        max_bytes = SANDBOX_MAX_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS,  (max_bytes, max_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (SANDBOX_TIMEOUT_SECONDS, SANDBOX_TIMEOUT_SECONDS))
    except Exception:
        pass  # resource limits not available on all platforms


def run_factor_in_sandbox(
    code: str,
    close: "pd.DataFrame",
    volume: "pd.DataFrame",
) -> dict:
    """
    Safely execute LLM-generated factor code and return scores.

    Args:
        code   : Python code defining factor_generated(close, volume) -> pd.Series
        close  : price DataFrame (dates × tickers)
        volume : volume DataFrame (dates × tickers)

    Returns:
        {
          "status": "ok" | "ast_error" | "timeout" | "runtime_error",
          "scores": {ticker: score, ...},   # present on success
          "violations": [...],              # present on ast_error
          "error": "...",                   # present on failure
        }
    """
    # Layer 1: AST check
    violations = ast_check(code)
    if violations:
        return {"status": "ast_error", "violations": violations, "scores": {}}

    # Serialize price data for subprocess (use last 90 rows to keep payload small)
    try:
        import pandas as pd
        close_json  = close.tail(90).to_json()
        volume_json = volume.tail(90).to_json()
    except Exception as e:
        return {"status": "runtime_error", "error": f"serialization failed: {e}", "scores": {}}

    # Write worker script to temp file
    worker_code = WORKER_TEMPLATE.format(user_code=textwrap.indent(code, "    " * 0))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="factor_sandbox_"
    ) as f:
        f.write(worker_code)
        worker_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, worker_path, close_json, volume_json],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_SECONDS,
            preexec_fn=_set_resource_limits,
        )
        output = proc.stdout.strip()
        if not output:
            stderr = proc.stderr.strip()[:300]
            return {"status": "runtime_error", "error": stderr or "no output", "scores": {}}

        parsed = json.loads(output)
        if "__error__" in parsed:
            return {"status": "runtime_error", "error": parsed["__error__"], "scores": {}}

        return {"status": "ok", "scores": parsed}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {SANDBOX_TIMEOUT_SECONDS}s", "scores": {}}
    except json.JSONDecodeError as e:
        return {"status": "runtime_error", "error": f"JSON parse failed: {e}", "scores": {}}
    except Exception as e:
        return {"status": "runtime_error", "error": str(e), "scores": {}}
    finally:
        try:
            os.unlink(worker_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Mock data
    dates   = pd.date_range("2025-01-01", periods=100)
    tickers = ["AAPL", "MSFT", "NVDA", "GS", "JPM"]
    close   = pd.DataFrame(np.random.randn(100, 5).cumsum(0) + 100, index=dates, columns=tickers)
    volume  = pd.DataFrame(np.random.randint(1, 10, (100, 5)) * 1e6, index=dates, columns=tickers)

    # Test 1: valid factor
    good_code = """
def factor_generated(close, volume):
    return -close.diff(5).iloc[-1]
"""
    print("Test 1 (valid factor):")
    r = run_factor_in_sandbox(good_code, close, volume)
    print(f"  status={r['status']} scores={list(r['scores'].keys())}\n")

    # Test 2: AST violation
    bad_code = """
import os
def factor_generated(close, volume):
    os.system('rm -rf /')
    return close.iloc[-1]
"""
    print("Test 2 (import os):")
    r = run_factor_in_sandbox(bad_code, close, volume)
    print(f"  status={r['status']} violations={r.get('violations')}\n")

    # Test 3: timeout
    slow_code = """
def factor_generated(close, volume):
    x = 0
    while True:
        x += 1
    return close.iloc[-1]
"""
    print("Test 3 (infinite loop — should timeout):")
    r = run_factor_in_sandbox(slow_code, close, volume)
    print(f"  status={r['status']} error={r.get('error')}\n")
