import os
from unittest.mock import ANY, MagicMock, patch

import pytest
from rdflib import Graph

from croissant_rdf.providers import HuggingfaceHarvester


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
        harvester = HuggingfaceHarvester()
        result = harvester.fetch_dataset_croissant("test_dataset")

        mock_get.assert_called_once_with(
            "https://huggingface.co/api/datasets/test_dataset/croissant", headers=ANY, timeout=30
        )
        assert result == {"name": "test_dataset", "description": "A test dataset"}


def test_mock_fetch_datasets(mock_response):
    mock_dataset = MagicMock()
    mock_dataset.id = "test_dataset"

    with patch(
        "croissant_rdf.providers.HuggingfaceHarvester.fetch_dataset_croissant",
        return_value=mock_response.json(),
    ):
        harvester = HuggingfaceHarvester(limit=1)
        result = harvester.fetch_datasets_croissant()
        assert len(result) == 1
        assert result[0] == {
            "name": "test_dataset",
            "description": "A test dataset",
        }


def test_mock_fetch_datasets_empty():
    with patch("croissant_rdf.providers.HuggingfaceHarvester.fetch_dataset_croissant", return_value=[]):
        harvester = HuggingfaceHarvester(limit=0)
        result = harvester.fetch_datasets_croissant()
        assert result == []


def test_get_datasets():
    harvester = HuggingfaceHarvester(limit=5)
    datasets = harvester.fetch_datasets_croissant()
    assert len(datasets) == 5


def test_fetch_croissant_dataset():
    harvester = HuggingfaceHarvester(use_api_key=False)
    result = harvester.fetch_dataset_croissant("fka/awesome-chatgpt-prompts")
    assert len(result) > 0
    assert "https://schema.org/" in result["@context"]["@vocab"]
    assert "http://mlcommons.org/croissant/" in result["@context"]["cr"]


def test_fetch_data_workflow():
    harvester = HuggingfaceHarvester(limit=5)
    croissant_dataset = harvester.fetch_datasets_croissant()
    assert len(croissant_dataset) == 5
    for dataset in croissant_dataset:
        if "error" not in dataset:
            assert "https://schema.org/" in dataset["@context"]["@vocab"]
            assert "http://mlcommons.org/croissant/" in dataset["@context"]["cr"]


OUTPUT_FILEPATH = "./tests/test_output.ttl"


def test_generate_ttl():
    """Test the complete generate_ttl workflow."""
    harvester = HuggingfaceHarvester(fname=OUTPUT_FILEPATH, limit=3, use_api_key=False)
    file_ttl = harvester.generate_ttl()
    assert os.path.isfile(OUTPUT_FILEPATH)
    assert os.path.isfile(file_ttl)
    g = Graph().parse(OUTPUT_FILEPATH, format="ttl")
    assert len(g) > 0
    os.remove(OUTPUT_FILEPATH)
