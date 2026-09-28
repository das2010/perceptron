"""Validación estática del código experto (RF-ARC-06, SPEC §13.2; ADR-0025).

Nunca ejecuta el código: lo parsea y recorre el AST. Solo admite imports de una lista
blanca y rechaza los nombres y atributos que abren caminos a E/S, red, procesos o
introspección (`eval`, `open`, `getattr`, dunders, `torch.load`, `torch.hub`…). Es la
primera capa; la segunda es el proceso sandbox (`perceptron.sandbox.guard`).
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field

MAX_SOURCE_BYTES = 200_000
ENTRYPOINT = "build_model"

ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "functools",
        "itertools",
        "math",
        "numbers",
        "typing",
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.nn.init",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)
# Atributos que dan acceso a disco, red, compilación o carga de código nativo/pickle.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "_C",
        "_dynamo",
        "_library",
        "_ops",
        "classes",
        "compile",
        "cpp_extension",
        "distributed",
        "from_file",
        "hub",
        "jit",
        "library",
        "load",
        "load_library",
        "multiprocessing",
        "onnx",
        "ops",
        "package",
        "save",
        "storage",
        "testing",
        "utils",
    }
)
ALLOWED_DUNDERS = frozenset({"__init__", "__name__", "__future__"})


class CodeIssue(BaseModel):
    line: int
    col: int
    message: str


class StaticReport(BaseModel):
    valid: bool
    issues: list[CodeIssue] = Field(default_factory=list)


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDERS


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[CodeIssue] = []

    def _bad(self, node: ast.AST, message: str) -> None:
        self.issues.append(
            CodeIssue(
                line=getattr(node, "lineno", 0), col=getattr(node, "col_offset", 0), message=message
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                self._bad(node, f"import no permitido: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            self._bad(node, "imports relativos no permitidos")
        elif module not in ALLOWED_IMPORTS:
            self._bad(node, f"import no permitido: {module}")
        for alias in node.names:
            if alias.name == "*":
                self._bad(node, "import * no permitido")
            elif alias.name in FORBIDDEN_ATTRIBUTES or _dunder(alias.name):
                self._bad(node, f"import no permitido: {module}.{alias.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or _dunder(node.id):
            self._bad(node, f"nombre no permitido: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES or _dunder(node.attr):
            self._bad(node, f"atributo no permitido: .{node.attr}")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._bad(node, "`global` no permitido")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bad(node, "funciones async no permitidas")


def check_source(source: str) -> StaticReport:
    """Revisa el código experto sin ejecutarlo."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return StaticReport(
            valid=False,
            issues=[CodeIssue(line=0, col=0, message=f"el código supera {MAX_SOURCE_BYTES} bytes")],
        )
    try:
        tree = ast.parse(source, filename="model.py", mode="exec")
    except SyntaxError as e:
        issue = CodeIssue(line=e.lineno or 0, col=(e.offset or 1) - 1, message=f"sintaxis: {e.msg}")
        return StaticReport(valid=False, issues=[issue])
    checker = _Checker()
    checker.visit(tree)
    issues = checker.issues
    defines = any(isinstance(n, ast.FunctionDef) and n.name == ENTRYPOINT for n in tree.body)
    if not defines:
        issues.append(CodeIssue(line=0, col=0, message=f"falta la función `{ENTRYPOINT}(config)`"))
    return StaticReport(valid=not issues, issues=issues)
