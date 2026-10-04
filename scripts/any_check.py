"""Flag local names whose type is Any/Unknown (shown white in the editor), e.g. `logits = model(x)`.

basedpyright's reportAny also flags returns, arguments and assignments to declared names; those are
fine, so only `Type of "name" is Any` is kept, and only when `name` has no annotation in its scope.
A line with `# any:ok` in its comment is skipped: an Any that's fine (`std *= n ** -0.5  # any:ok`).
Usage: uv run python scripts/any_check.py FILE.pyn|FILE.py
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

RULES = "reportAny=true, reportUnknownVariableType=true"
HIT = re.compile(r':(\d+):(\d+) - error: Type of "(\w+)" is (Any|partially unknown|unknown)')
Func = ast.FunctionDef | ast.AsyncFunctionDef
SKIP = re.compile(r"#.*\bany:ok\b")


def source_tree(path: Path) -> ast.Module:
    """AST with the file's own line numbers."""
    if path.suffix != ".pyn":
        return ast.parse(path.read_text())
    from byname import source_ast

    return source_ast(path.read_text(), str(path))


def declared(tree: ast.Module, line: int, name: str) -> bool:
    """`name` is annotated (parameter or `name: T`) in the innermost function around `line`."""
    scope: ast.Module | Func = tree
    for node in ast.walk(tree):
        if (
            isinstance(node, Func)
            and node.lineno <= line <= (node.end_lineno or 0)
            and (isinstance(scope, ast.Module) or node.lineno >= scope.lineno)
        ):
            scope = node
    if isinstance(scope, Func):
        a = scope.args
        if any(p.arg == name and p.annotation for p in [*a.posonlyargs, *a.args, *a.kwonlyargs]):
            return True
    return any(
        isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name
        for n in ast.walk(scope)
    )


def main() -> int:
    path = Path(sys.argv[1]).resolve()
    # a copy in the same folder, so imports resolve the same, with the rules switched on
    tmp = path.with_name(f"_anycheck_{path.name}")
    _ = tmp.write_text(f"# pyright: {RULES}\n" + path.read_text())
    try:
        cmd = ["byname", "tool", "basedpyright"] if path.suffix == ".pyn" else ["basedpyright"]
        out = subprocess.run([*cmd, str(tmp)], capture_output=True, text=True, check=False).stdout
    finally:
        tmp.unlink()
    tree = source_tree(path)
    lines = path.read_text().splitlines()
    hits: set[tuple[int, int, str, str]] = set()
    for line_s, col_s, name, kind in HIT.findall(out):
        line, col = int(line_s) - 1, int(col_s)  # -1: the rules comment shifted every line down
        if not lines[line - 1][col - 1 :].startswith(name):
            continue  # `logits.shape`, reported at `logits`: an attribute of an Any value, not a name
        if SKIP.search(lines[line - 1]):
            continue
        if not declared(tree, line, name):
            hits.add((line, col, name, kind))
    for line, col, name, kind in sorted(hits):
        print(f"{path}:{line}:{col} - {name} is {kind}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
