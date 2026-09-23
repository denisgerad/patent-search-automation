from unittest.mock import MagicMock

from services.ai_invention_structure_service import generate_invention_structure


def test_generate_invention_structure_uses_generate_api():
    client = MagicMock()
    client.generate.return_value = '{"application": ["autonomous driving"], "core_system": ["vehicle guidance system"], "sensors": ["camera"], "functions": ["detect"], "objects": ["pedestrian"], "search_paths": [{"name": "Sensor + Function", "roles": ["sensors", "functions"], "expression": "(camera OR \'camera sensor\') AND (detect OR detection)", "purpose": "Find prior art of sensing and detection."}]}'

    result = generate_invention_structure(
        "A camera-based pedestrian detection system for autonomous vehicles.",
        client,
    )

    assert result["application"] == ["autonomous driving"]
    assert result["core_system"] == ["vehicle guidance system"]
    assert result["sensors"] == ["camera"]
    assert result["functions"] == ["detect"]
    assert result["objects"] == ["pedestrian"]
    assert result["search_paths"][0]["name"] == "Sensor + Function"
    client.generate.assert_called_once()
