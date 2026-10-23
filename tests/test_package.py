import cvxpy
import matplotlib
import numpy
import osqp
import scipy

import simts_optimization


def test_package_version() -> None:
    assert simts_optimization.__version__ == "0.1.0"


def test_dependencies_are_importable() -> None:
    assert all(
        dependency.__version__
        for dependency in (numpy, scipy, cvxpy, osqp, matplotlib)
    )
