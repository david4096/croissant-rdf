import json

from rdflib import Graph, URIRef
from rich.progress import track

from croissant_rdf.utils import logger


def chunk_data(data, chunk_size):
    """Chunking data"""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def convert_to_rdf(data, output_file: str, base: str = "http://fakebase"):
    """Take a JSON-serializable data structure, converts it to RDF using
    JSON-LD format, and serializes it into Turtle format, saving it to the specified file.

    Args:
        data (list|dict): The JSON-serializable data structure to convert to RDF.
        output_file (str): The file path where the Turtle (.ttl) formatted RDF data will be saved.
        base (str): The base URI for the RDF graph, used as a prefix in generated RDF triples.
                    Defaults to "http://fakebase".

    Returns:
        str: A string representation of the RDF graph in Turtle format.
    """
    total_items = len(data)
    chunk_size = total_items // 100 if total_items > 100 else 1
    logger.info(f"Starting RDF conversion. Total items: {total_items}, Chunk size: {chunk_size}")
    g = Graph()

    for chunk in track(chunk_data(data, chunk_size), "Parsing data", total_items):
        for item in chunk:
            item_json_ld = json.dumps(item)
            g.parse(data=item_json_ld, format="json-ld", base=URIRef(base))

    logger.info(f"Writing {len(g)} RDF triples to file {output_file}")
    # Implementation for generating RDF
    g.serialize(destination=output_file, format="ttl")
    return output_file
