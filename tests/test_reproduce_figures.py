from pathlib import Path
import hashlib
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.integration
def test_publication_figure_command_reproduces_archived_pngs():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run the archived-figure integration test")
    sys.path.insert(0, str(ROOT / "src"))
    from e1.make_publication_figures import FIG_DIR

    expected = sorted(FIG_DIR.glob("*.png"))
    if len(expected) != 6:
        pytest.skip("archived publication PNG files are not present")
    before = {path.name: _sha256(path) for path in expected}
    subprocess.run(
        [sys.executable, "src/e1/make_publication_figures.py"],
        cwd=ROOT,
        check=True,
    )
    after = {path.name: _sha256(path) for path in expected}
    assert after == before
