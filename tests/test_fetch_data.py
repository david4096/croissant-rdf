# import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

# sys.path.append('.')
from croissant_rdf.huggingface.fetch_data import (
    croissant_dataset,
    fetch_datasets,
    get_datasets,
)


@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {"HF_API_KEY": "fake_api_key"}):
        yield


@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.json.return_value = {"name": "test_dataset", "description": "A test dataset"}
    return mock


def test_mock_croissant_dataset(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        result = croissant_dataset("test_dataset")

        mock_get.assert_called_once_with("https://huggingface.co/api/datasets/test_dataset/croissant", headers=ANY)
        assert result == {"name": "test_dataset", "description": "A test dataset"}


def test_mock_fetch_datasets(mock_response):
    mock_dataset = MagicMock()
    mock_dataset.id = "test_dataset"

    with patch("croissant_rdf.huggingface.fetch_data.get_datasets", return_value=[mock_dataset]) as mock_get:
        with patch(
            "croissant_rdf.huggingface.fetch_data.croissant_dataset",
            return_value=mock_response.json(),
        ):
            result = fetch_datasets(limit=1)

            mock_get.assert_called_once_with(1, None)
            assert len(result) == 1
            assert result[0] == {
                "name": "test_dataset",
                "description": "A test dataset",
            }


def test_mock_fetch_datasets_empty():
    with patch("croissant_rdf.huggingface.fetch_data.get_datasets", return_value=[]):
        result = fetch_datasets(limit=0)
        assert result == []


def test_get_datasets():
    datasets = get_datasets(5)
    assert len(datasets) == 5


def test_fetch_croissant_dataset():
    result = croissant_dataset("fka/awesome-chatgpt-prompts", False)
    assert len(result) > 0
    assert "https://schema.org/" in result["@context"]["@vocab"]
    assert "http://mlcommons.org/croissant/" in result["@context"]["cr"]


def test_fetch_data_workflow():
    croissant_dataset = fetch_datasets(5)
    assert len(croissant_dataset) == 5
    for dataset in croissant_dataset:
        if "error" not in dataset.keys():
            assert "https://schema.org/" in dataset["@context"]["@vocab"]
            assert "http://mlcommons.org/croissant/" in dataset["@context"]["cr"]
