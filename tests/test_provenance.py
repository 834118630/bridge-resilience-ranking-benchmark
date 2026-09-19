from pathlib import Path
import json
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.provenance import write_run_metadata


def test_run_metadata_records_command_hashes_and_status():
    root = Path(__file__).resolve().parents[1] / "tmp" / "q1_check"
    work = root / f"provenance_test_{uuid.uuid4().hex}"
    work.mkdir(parents=True)
    try:
        input_path = work / "input.csv"
        input_path.write_text("value\n1\n", encoding="utf-8")
        output_dir = work / "output"
        output_dir.mkdir()
        (output_dir / "result.csv").write_text("candidate\nL4\n", encoding="utf-8")

        metadata_path = write_run_metadata(
            output_dir,
            command="python -m e1.run_example",
            parameters={"samples": 10},
            input_paths=(input_path,),
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "complete"
        assert metadata["failure_status"] == "none"
        assert metadata["command"] == "python -m e1.run_example"
        assert input_path.as_posix() in metadata["input_hashes"]
        assert "result.csv" in next(iter(metadata["output_hashes"]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
