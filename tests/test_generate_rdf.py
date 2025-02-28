import os

import pytest
from rdflib import Graph

from croissant_rdf.generate_rdf import convert_to_rdf
from croissant_rdf.huggingface.fetch_data import fetch_datasets

OUTPUT_FILEPATH = "./tests/test_output.ttl"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.isfile(OUTPUT_FILEPATH):
        os.remove(OUTPUT_FILEPATH)


def test_convert_to_rdf_mock_data():
    """Test with mock data"""
    data = [
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description",
            },
            "name": "test_dataset",
        },
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description",
            },
            "name": "test_dataset_2",
        },
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description",
            },
            "name": "test_dataset_3",
        },
    ]
    file_ttl = convert_to_rdf(data, OUTPUT_FILEPATH)
    # assert there is a file named test_output.ttl in the data directory
    assert os.path.isfile(OUTPUT_FILEPATH)
    assert os.path.isfile(file_ttl)
    # assert there are 9 triples in the graph
    g = Graph().parse(OUTPUT_FILEPATH, format="ttl")
    assert len(g) == 3


def test_convert_to_rdf_mock_data_empty():
    """Test with empty data"""
    data = []
    convert_to_rdf(data, OUTPUT_FILEPATH)
    assert os.path.isfile(OUTPUT_FILEPATH)
    g = Graph().parse(OUTPUT_FILEPATH, format="ttl")
    assert len(g) == 0


def test_convert_to_rdf_real_data():
    """Test data from HuggingFace, does not require API key"""
    data = fetch_datasets(limit=5)
    convert_to_rdf(data, OUTPUT_FILEPATH)
    g = Graph().parse(OUTPUT_FILEPATH, format="ttl")
    assert len(g) > 0
    # asset there is a file named test_output.ttl in the data directory
    assert os.path.isfile
