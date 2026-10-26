import json
import os
from datetime import datetime, timezone
from pathlib import Path


class Ledger:
    """Append-only event stream; each record is sufficient to replay an episode's decisions."""

    def __init__(self, path, episode_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.sequence = 0

    def write(self, kind, **data):
        self.sequence += 1
        record = {"schema": 1, "episode_id": self.episode_id,
                  "sequence": self.sequence, "at": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record
