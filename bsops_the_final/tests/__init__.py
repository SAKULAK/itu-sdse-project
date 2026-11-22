# The test modules are listed here instead of importing their classes: unittest
# discovery also collects tests from this package's namespace, so importing the
# TestCase classes here would run every test twice.
__all__ = [
    "test_export",
    "test_pipeline",
    "test_promote",
    "test_selection",
]
