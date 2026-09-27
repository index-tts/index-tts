import ast
from pathlib import Path


def _webui_source():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    return webui_path, webui_path.read_text(encoding="utf-8")


def test_webui_python_source_parses():
    webui_path, source = _webui_source()

    assert compile(source, str(webui_path), "exec") is not None


def test_webui_gr_error_is_never_constructed_without_raise():
    # `gr.Error` is an exception class: calling it as a statement builds an exception and
    # discards it, so the user never sees the message. Raise it, or use gr.Warning / gr.Info.
    webui_path, source = _webui_source()
    bare = [
        node.lineno
        for node in ast.walk(ast.parse(source, str(webui_path)))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "Error"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "gr"
    ]
    assert not bare, f"gr.Error(...) is constructed but never raised at webui.py lines {bare}"
