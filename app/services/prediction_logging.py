import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

LOGGER = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def append_prediction_log(
    path: Path,
    features: Mapping[str, object],
    prediction: str,
) -> bool:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": dict(features),
        "prediction": prediction,
    }
    try:
        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")
    except OSError:
        LOGGER.warning("Could not write an experimental prediction log entry.", exc_info=True)
        return False
    return True
