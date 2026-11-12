"""Source-only import-policy evidence, never a Python sandbox or admission check."""

from __future__ import annotations

import ast
from collections.abc import Set
from dataclasses import dataclass

from .execution import validate_code

# Candidate P1 math/stdlib allowlist for static source triage (not a security boundary).
CANDIDATE_MATH_MODULES: frozenset[str] = frozenset(
    {
        "array",
        "bisect",
        "cmath",
        "collections",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
)


@dataclass(frozen=True)
class ImportSite:
    module: str
    line: int


@dataclass(frozen=True)
class ImportInventory:
    direct: tuple[ImportSite, ...]
    unresolved: tuple[ImportSite, ...]

    def unlisted(self, allowed_modules: Set[str]) -> tuple[ImportSite, ...]:
        """Compare known direct imports only; unresolved/runtime imports remain unknown."""
        return tuple(
            site
            for site in self.direct
            if not any(
                site.module == allowed or site.module.startswith(allowed + ".")
                for allowed in allowed_modules
            )
        )


@dataclass(frozen=True)
class ImportPolicyReport:
    inventory: ImportInventory
    disallowed: tuple[ImportSite, ...]
    unresolved: tuple[ImportSite, ...]
    compliant: bool


def evaluate_import_policy(
    code: str,
    allowed_modules: Set[str] = CANDIDATE_MATH_MODULES,
    *,
    allow_unresolved: bool = False,
) -> ImportPolicyReport:
    """Evaluate static AST imports against a candidate module allowlist."""
    inventory = inspect_imports(code)
    disallowed = inventory.unlisted(allowed_modules)
    unresolved = inventory.unresolved
    compliant = not disallowed and (allow_unresolved or not unresolved)
    return ImportPolicyReport(
        inventory=inventory,
        disallowed=disallowed,
        unresolved=unresolved,
        compliant=compliant,
    )


def inspect_imports(code: str) -> ImportInventory:
    """Inventory all syntactic branches without executing/importing the source.

    Dynamic sites are conservative hints, not a complete bypass detector. Imports
    inside libraries, indirect calls and reflective Python behavior need separate
    runtime enforcement; an empty inventory is never proof of safety.
    """
    validate_code(code)
    tree = ast.parse(code, filename="<bridge-import-inventory>")
    direct = []
    unresolved = []
    dynamic_names = {"__import__", "exec", "eval", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "importlib",
            "builtins",
        ):
            for alias in node.names:
                if alias.name in (
                    "import_module",
                    "__import__",
                    "exec",
                    "eval",
                    "compile",
                ):
                    dynamic_names.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct.extend(ImportSite(alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                unresolved.append(
                    ImportSite("." * node.level + (node.module or ""), node.lineno)
                )
            elif node.module:
                direct.append(ImportSite(node.module, node.lineno))
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in dynamic_names
        ):
            unresolved.append(ImportSite(node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in (
            "import_module",
            "__import__",
        ):
            unresolved.append(ImportSite(node.attr, node.lineno))

    def key(site: ImportSite) -> tuple[int, str]:
        return site.line, site.module

    return ImportInventory(
        tuple(sorted(direct, key=key)), tuple(sorted(unresolved, key=key))
    )
