import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os
import time

import pytest


def _webui_source():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    return webui_path, webui_path.read_text(encoding="utf-8")


def test_webui_python_source_parses():
    webui_path, source = _webui_source()

    assert compile(source, str(webui_path), "exec") is not None


def test_duration_factor_is_only_forwarded_to_v25():
    webui_path, source = _webui_source()
    tree = ast.parse(source, filename=str(webui_path))
    gen_single = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "gen_single"
    )

    infer_kwargs = next(
        node.value for node in ast.walk(gen_single)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "infer_kwargs" for target in node.targets)
    )
    assert isinstance(infer_kwargs, ast.Call)
    assert "duration_factor" not in {
        keyword.arg for keyword in infer_kwargs.keywords if keyword.arg is not None
    }

    v25_branch = next(
        node for node in gen_single.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "IS_V25"
    )
    forwarded_keys = {
        target.slice.value
        for statement in v25_branch.body
        for target in getattr(statement, "targets", [])
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "infer_kwargs"
        and isinstance(target.slice, ast.Constant)
    }
    assert {"lang", "duration_factor"} <= forwarded_keys


def test_duration_factor_control_is_hidden_for_v2():
    webui_path, source = _webui_source()
    tree = ast.parse(source, filename=str(webui_path))

    def assigned_component(statements, component_name):
        return any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "duration_factor"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == component_name
            for statement in statements
        )

    duration_control = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "IS_V25"
        and assigned_component(node.body, "Slider")
    )
    assert assigned_component(duration_control.orelse, "State")


@pytest.mark.parametrize("is_v25", [False, True])
@pytest.mark.parametrize("target_duration", [None, 1.23])
def test_generation_callback_forwards_target_duration_in_both_versions(is_v25, target_duration):
    webui_path, source = _webui_source()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "gen_single")
    tts = SimpleNamespace(infer=Mock(return_value="output.wav"))
    namespace = {
        "os": os, "time": time, "tts": tts, "IS_V25": is_v25,
        "cmd_args": SimpleNamespace(verbose=False),
        "gr": SimpleNamespace(Progress=lambda: None, update=lambda **kwargs: kwargs),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(webui_path), "exec"), namespace)
    result = namespace["gen_single"](
        0, "reference.wav", "text", "EN", None, 1.0,
        *([0.0] * 8), "", False, 120, 1.5, target_duration,
        False, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500,
    )
    kwargs = tts.infer.call_args.kwargs
    assert kwargs["target_duration"] == target_duration
    assert kwargs["do_sample"] is False
    if is_v25:
        assert kwargs["duration_factor"] == 1.5
        assert kwargs["lang"] == "EN"
    else:
        assert "duration_factor" not in kwargs
        assert "lang" not in kwargs
    assert result == {"value": "output.wav", "visible": True}
