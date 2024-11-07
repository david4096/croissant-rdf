from rdflib import Graph, Namespace, URIRef, Literal
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def show_progress(current, total):
    """Display progress of completion."""
    progress = (current / total) * 100
    print(f"Progress: {current}/{total} ({progress:.2f}%)")

def convert_to_rdf(data, output_file,base="http://fakebase"):
    """
    This function takes a JSON-serializable data structure, converts it to RDF using 
    JSON-LD format, and serializes it into Turtle format, saving it to the specified file.

    Args:
        data (dict): The JSON-serializable data structure to convert to RDF.
        output_file (str): The file path where the Turtle (.ttl) formatted RDF data will be saved.
        base (str): The base URI for the RDF graph, used as a prefix in generated RDF triples.
                    Defaults to "http://fakebase".

    Returns:
        str: A string representation of the RDF graph in Turtle format.
    """
    json_ld_data = json.dumps(data)
    
    total_items = len(data) if isinstance(data, list) else 1
    logging.info(f"Starting RDF conversion. Total items: {total_items}")

    for i, item in enumerate(data if isinstance(data, list) else [data], start=1):
            show_progress(i, total_items)
        
    g = Graph().parse(
        data=json_ld_data,
        format='json-ld',
        base=URIRef(base)
    )

    logging.info(f"RDF data successfully saved to {output_file}")
    # Implementation for generating RDF
    return g.serialize(destination=output_file, format='ttl')
    
