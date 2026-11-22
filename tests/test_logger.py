import json
import logging

import pytest

from core import logger as L


def test_json_file_logging_and_log_step(tmp_path):
    log_file = tmp_path / "logs" / "clipgod.log"
    L.setup_logging(level="DEBUG", log_file=log_file, console=False, force=True)
    log = L.get_logger("modules.scan")
    assert log.name == "clipgod.scan"
    with L.log_step(log, "transcribe", video_id="abc", name="reserved") as ctx:
        ctx["segments"] = 3
    with pytest.raises(ValueError), L.log_step(log, "render"):
        raise ValueError("x")
    for h in logging.getLogger("clipgod").handlers:
        h.flush()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    done = next(e for e in lines if e.get("event") == "done")
    assert done["video_id"] == "abc" and done["segments"] == 3 and done["ctx_name"] == "reserved"
    err = next(e for e in lines if e.get("event") == "error")
    assert "ValueError" in err["exc"]
    L.setup_logging(console=False, log_file="", force=True)
