"""Import-policy planning fixtures; no analyzed code is ever executed."""

from __future__ import annotations

import pytest

from neural_compiler.bridge.imports import ImportSite, inspect_imports


def test_inventory_preserves_aliases_relative_imports_and_source_lines():
    inventory = inspect_imports(
        "import math as m, os.path\n"
        "from fractions import Fraction as F\n"
        "from . import local\n"
        "from ..helpers import solve\n"
    )
    assert inventory.direct == (
        ImportSite("math", 1),
        ImportSite("os.path", 1),
        ImportSite("fractions", 2),
    )
    assert inventory.unresolved == (ImportSite(".", 3), ImportSite("..helpers", 4))
    assert inventory.unlisted({"math", "fractions"}) == (ImportSite("os.path", 1),)


def test_inventory_does_not_adopt_a_branch_as_runtime_truth():
    inventory = inspect_imports("if False:\n    import os\nelse:\n    import math\n")
    assert inventory.direct == (ImportSite("os", 2), ImportSite("math", 4))
    assert inventory.unlisted({"math"}) == (ImportSite("os", 2),)


@pytest.mark.parametrize(
    "code,unknown",
    [
        ("__import__('os')", "__import__"),
        ("loader = __import__\nloader('os')", "__import__"),
        ("import importlib as lib\nlib.import_module('os')", "import_module"),
        ("from importlib import import_module as load\nload('os')", "load"),
        ("from builtins import __import__ as load\nload('os')", "load"),
        ("exec('import os')", "exec"),
    ],
)
def test_dynamic_import_routes_stay_unresolved(code, unknown):
    inventory = inspect_imports(code)
    assert any(site.module == unknown for site in inventory.unresolved)
    assert not any(site.module == "os" for site in inventory.direct)


def test_inventory_does_not_execute_source(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    code = f"open({str(sentinel)!r}, 'w').write('not executed')\nimport math"
    assert inspect_imports(code).direct == (ImportSite("math", 2),)
    assert not sentinel.exists()


def test_module_prefix_comparison_respects_package_boundaries():
    inventory = inspect_imports("import os.path, os.pathology")
    assert inventory.unlisted({"os.path"}) == (ImportSite("os.pathology", 1),)


def test_invalid_and_oversized_source_is_explicit():
    with pytest.raises(SyntaxError):
        inspect_imports("import")
    with pytest.raises(ValueError, match="UTF-8"):
        inspect_imports("#" + "\u00e9" * (64 * 1024))
