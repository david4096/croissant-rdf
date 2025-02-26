import os

from rdflib import Graph

from croissant_rdf.generate_rdf import convert_to_rdf
from croissant_rdf.huggingface.fetch_data import fetch_datasets


def test_convert_to_rdf_mock_data():
    # Test data
    data = fetch_datasets(limit=1)
    data = [
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description"
            },
            "name": "test_dataset",
        },
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description"
            },
            "name": "test_dataset_2",
        },
        {
            "@context": {
                "name": "http://schema.org/name",
                "description": "http://schema.org/description",
            },
            "name": "test_dataset_3",
        }
    ]
    print(data)
    #print current folder where the test is running
    filename = "./tests/data/test_output.ttl"
    file_ttl = convert_to_rdf(data, filename)
    # asset there is a file named test_output.ttl in the data directory
    assert os.path.isfile(filename)
    # assert there are 9 triples in the graph
    g = Graph().parse(filename, format='ttl')
    assert len(g) == 3
    # clean up
    os.remove(filename)

def test_convert_to_rdf_mock_data_empty():
    # Test data
    data = []
    filename = "./tests/data/test_output.ttl"
    file_ttl = convert_to_rdf(data, filename)
    # asset there is a file named test_output.ttl in the data directory
    assert os.path.isfile(filename)
    # assert there are 9 triples in the graph
    g = Graph().parse(filename, format='ttl')
    assert len(g) == 0
    # clean up
    os.remove(filename)

def test_convert_to_rdf_real_data():
    # Test data
    data = fetch_datasets(limit=5)
    filename = "./tests/data/test_output.ttl"
    file_ttl = convert_to_rdf(data, filename)
    g = Graph().parse(filename, format='ttl')
    assert len(g)>0
    # asset there is a file named test_output.ttl in the data directory
    assert os.path.isfile

    # clean up
    os.remove(filename)