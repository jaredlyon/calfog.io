"""Content identities and completion-last publication for study artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

RECORD_SCHEMA_VERSION = 1


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_text(path, text: str) -> None:
    atomic_write_bytes(path, text.encode('utf-8'))


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def completion_path(family_dir, family) -> Path:
    return Path(family_dir) / f'{family}_run_v2.json'


def load_completion(family_dir, family) -> dict | None:
    try:
        record = json.loads(completion_path(family_dir, family).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    required = {'schema_version', 'family', 'sites', 'inputs', 'metrics_sha256', 'prediction_sha256'}
    if not isinstance(record, dict) or set(record) != required or record['schema_version'] != RECORD_SCHEMA_VERSION:
        return None
    if (not isinstance(record['family'], str) or not isinstance(record['sites'], list) or
            not all(isinstance(site, str) for site in record['sites']) or
            record['sites'] != sorted(set(record['sites'])) or not isinstance(record['inputs'], dict) or
            not isinstance(record['metrics_sha256'], str) or not isinstance(record['prediction_sha256'], dict) or
            not all(isinstance(name, str) and isinstance(digest, str)
                    for name, digest in record['prediction_sha256'].items())):
        return None
    return record


def read_verified(path, expected_sha256: str) -> bytes:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f'{path}: sha256 mismatch')
    return data
