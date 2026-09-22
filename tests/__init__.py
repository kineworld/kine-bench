"""Make every ``test_*`` in this directory visible to the standard runner.

**This is the organisation-level copy.** Copy it into a repository's test directory as
``__init__.py``. Together with ``check_collection.py`` from this directory and three lines
of workflow YAML, that is the whole adoption -- see ``guards/README.md``.

Why it exists: ``unittest`` collects ``TestCase`` methods and ignores functions named
``test_*`` that sit at module level. So a test file written to be run directly

    $ python tests/test_windows.py
    PASS test_windows

is invisible to the standard runner

    $ python -m unittest discover -s tests
    Ran 0 tests in 0.000s
    NO TESTS RAN

The file is not broken. The loader cannot see it, and an exit code of 5 next to a green
hand-run is easy to read as "nothing wrong here". Four self-authored repositories in this
organisation hold 47 such functions -- every one of them correct, none of them ever
executed. The audit is in kineworld/.github#3. This module implements the documented
``load_tests`` protocol so they are collected without changing a single test.

Use either of:

    python -m unittest discover -s tests -t .     # note -t .
    python -m unittest tests

``-t .`` is load-bearing. Without a top-level directory, ``discover`` treats the test
directory as a set of top-level modules, never imports it as a package, and this module is
never reached. ``check_collection.py`` and
``kineworld/.github/.github/workflows/python-tests.yml`` both branch on the same rule, so
the guard, the local command and CI always agree on what "collected" means.
"""

import importlib
import inspect
import pkgutil
import unittest


def _wrapper(module_name, function_name, function):
    """Return a one-method ``TestCase`` whose only job is to call ``function``."""

    class _BareFunctionTest(unittest.TestCase):
        def runTest(self):  # unittest's own protocol name; it is not misspelled
            function()

    _BareFunctionTest.__name__ = "Test_%s_%s" % (module_name, function_name)
    _BareFunctionTest.__qualname__ = _BareFunctionTest.__name__
    return _BareFunctionTest()


def _iter_test_modules():
    """Yield ``(name, module)`` for every ``test_*.py`` beside this file.

    Enumerated at import time rather than listed in a literal: a file added later must not
    need an edit here. A hand-maintained list drifts, and a drifted list silently stops
    covering the thing it was written to cover -- which is this defect one level up.
    """
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("test_"):
            yield info.name, importlib.import_module("%s.%s" % (__name__, info.name))


def load_tests(loader, tests, pattern):
    """Collect ``TestCase`` classes *and* module-level ``test_*`` functions.

    A package-level ``load_tests`` **replaces** the default collection rather than adding
    to it, so both kinds have to be handled here. Handling only the bare functions drops
    every ``TestCase`` method in the directory -- and still reports success, which is how
    the first attempt at this fix in kineworld/kine-jepa#4 would have quietly removed five
    passing tests.
    """
    suite = unittest.TestSuite()
    for module_name, module in _iter_test_modules():
        # 1. TestCase subclasses, through the standard loader, so that the usual rules
        #    (including a module-level load_tests) still apply to them.
        suite.addTests(loader.loadTestsFromModule(module))
        # 2. Module-level test_* functions, which that loader passes over.
        for name, value in sorted(vars(module).items()):
            if not name.startswith("test_") or not inspect.isfunction(value):
                continue
            # Only functions *defined in* the module. ``check_collection.py`` counts
            # top-level ``def test_*`` with ast, and the two numbers have to mean the same
            # thing; an imported name would be collected here and not counted there.
            if value.__name__ != name or value.__module__ != module.__name__:
                continue
            suite.addTest(_wrapper(module_name, name, value))
    return suite
