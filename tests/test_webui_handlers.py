"""Static guards for the Gradio event handlers in ``webui.py``.

These run on the source alone: importing ``webui.py`` requires Gradio (an
optional extra) and downloads model checkpoints, so it cannot be imported in the
``not gpu`` CI job. This mirrors ``tests/test_webui_syntax.py``, which likewise
only reads the file.

Both guards cover regressions in the preset feature:

* ``confirm_save_preset_from_modal`` reads ``result[0]``/``result[1]`` from
  ``on_preset_save``. One of its branches returned a bare ``gr.update()`` --
  Gradio's "skip" sentinel, which is just the dict ``{"__type__": "update"}`` --
  so saving a preset with an empty name raised ``KeyError: 0`` (index-tts#787).
* ``on_preset_load`` returned ``{}`` on its early-exit paths. Gradio wraps a lone
  non-tuple value into a one-element list, so an empty dict fails its output
  arity check with "didn't return enough output values" for a 24-output event.
"""

import ast
from pathlib import Path

WEBUI_PATH = Path(__file__).resolve().parents[1] / "webui.py"

# Gradio event triggers, i.e. the methods that accept `fn=` and `outputs=`.
GRADIO_EVENTS = frozenset({
    "blur", "change", "clear", "click", "delete", "edit", "error", "focus",
    "input", "key_up", "like", "load", "release", "select", "stop", "stream",
    "submit", "tick", "unload", "upload",
})


def _parse(path=WEBUI_PATH):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _module_functions(tree):
    return {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }


def _list_lengths(tree):
    """Lengths of every `name = [...]` / `name = (...)` assignment in the module."""
    lengths = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lengths[target.id] = len(node.value.elts)
    return lengths


def _output_count(node, list_lengths):
    """Number of components in an `outputs=` argument, resolving list variables."""
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        return list_lengths.get(node.id)
    return None


def _is_skip_sentinel(value):
    """``gr.update()`` with no arguments -- Gradio's dict meaning "update nothing".

    Gradio compares returned dicts against this sentinel, so the no-argument form
    is special-cased and expanded to skip every output. It is only meaningful as a
    return value, never as something a caller may index.
    """
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "update"
        and not value.args
        and not value.keywords
    )


def _subscripted_helpers(tree):
    """Module-level functions whose result a caller indexes (`x = f(...)`; `x[0]`)."""
    functions = _module_functions(tree)
    helpers = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, call = node.targets[0], node.value
        if not (isinstance(target, ast.Name) and isinstance(call, ast.Call)):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id not in functions:
            continue
        for inner in ast.walk(tree):
            if (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == target.id
            ):
                helpers[call.func.id] = functions[call.func.id]
                break
    return helpers


def test_indexed_helpers_never_return_a_skip_sentinel():
    """A helper whose result is indexed must return a value, never a skip dict."""
    offenders = set()
    for name, function in _subscripted_helpers(_parse()).items():
        for node in ast.walk(function):
            if not isinstance(node, ast.Return):
                continue
            if _is_skip_sentinel(node.value):
                offenders.add(
                    f"webui.py:{node.lineno} {name}() returns a bare gr.update(), "
                    "but a caller indexes its result"
                )
            elif isinstance(node.value, ast.Dict):
                offenders.add(
                    f"webui.py:{node.lineno} {name}() returns a dict, "
                    "but a caller indexes its result"
                )

    assert not offenders, "\n".join(sorted(offenders))


def test_multi_output_handlers_never_return_an_empty_value():
    """Handlers wired to several outputs must not return `{}` or bare `None`."""
    tree = _parse()
    list_lengths = _list_lengths(tree)
    functions = _module_functions(tree)
    offenders = set()

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in GRADIO_EVENTS:
            continue

        handler = outputs = None
        for keyword in node.keywords:
            if keyword.arg == "fn":
                handler = keyword.value
            elif keyword.arg == "outputs":
                outputs = keyword.value
        if node.args:
            handler = handler or node.args[0]
        if len(node.args) > 1:
            outputs = outputs or node.args[1]

        if not isinstance(handler, ast.Name) or handler.id not in functions:
            continue
        count = _output_count(outputs, list_lengths)
        if count is None or count <= 1:
            continue

        for ret in ast.walk(functions[handler.id]):
            if not isinstance(ret, ast.Return):
                continue
            if ret.value is None:
                offenders.add(
                    f"webui.py:{ret.lineno} {handler.id}() returns None for a "
                    f"{count}-output event"
                )
            elif isinstance(ret.value, ast.Dict) and not ret.value.keys:
                offenders.add(
                    f"webui.py:{ret.lineno} {handler.id}() returns {{}} for a "
                    f"{count}-output event"
                )

    assert not offenders, "\n".join(sorted(offenders))
