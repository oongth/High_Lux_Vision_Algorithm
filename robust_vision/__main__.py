# 실행 진입점: python -m robust_vision

import sys

from .bootstrap import require_packages, use_project_python


def main(argv=None):
    exit_code = use_project_python(argv)
    if exit_code is not None:
        return exit_code

    require_packages()
    from .app.cli import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    sys.exit(main())
