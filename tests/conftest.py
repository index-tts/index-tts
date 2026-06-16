# padding_test.py is a manual script, not a pytest test module.
collect_ignore = ["padding_test.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--use-modelscope", action="store_true", default=False,
        help="Force download tests to use ModelScope path (for CN network testing)",
    )


def pytest_configure(config):
    if config.getoption("--use-modelscope"):
        import indextts.utils.model_download as md
        md._USING_MODELSCOPE = True
