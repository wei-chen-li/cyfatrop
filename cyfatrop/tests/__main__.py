"""Run with python -m cyfatrop.tests or python cyfatrop/tests/__main__.py."""

import unittest

from cyfatrop.tests import test_solver


def main():
    unittest.main(module=test_solver)


if __name__ == "__main__":
    main()
