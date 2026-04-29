"""Compatibility wrapper for the legacy stealth-data generator.

The original implementation lives in scripts/archive/generate_stealth_answers.py.
This wrapper keeps the README path valid and reflects that stealth data is now
treated as an active dataset rather than an archived artifact.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = Path(__file__).resolve().parents[1] / "archive" / "generate_stealth_answers.py"
    runpy.run_path(str(legacy_script), run_name="__main__")
