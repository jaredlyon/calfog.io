"""Honesty-clause checks over every final serving metadata artifact."""
from conftest import RETRAIN, SITES, load_json


def test_not_marked_faithful():
    """Schema compatibility must never be represented as feed faithfulness."""
    for site in SITES:
        path = RETRAIN / "serving" / site / "metadata.json"
        metadata = load_json(path)
        assert isinstance(metadata, dict), f"{path}: metadata must be a JSON object"
        assert "faithful" in metadata, f"{site}: serving metadata omits faithful"
        assert metadata["faithful"] is False, (
            f"{site}: faithful must be the JSON boolean false, not a truthy/string value"
        )
        assert metadata.get("train_source") == "era5_archive", (
            f"{site}: train_source must disclose ERA5 Archive reanalysis"
        )
