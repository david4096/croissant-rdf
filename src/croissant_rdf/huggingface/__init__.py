import argparse
from typing import Optional

from croissant_rdf.generate_rdf import convert_to_rdf
from croissant_rdf.huggingface.fetch_data import fetch_datasets
from croissant_rdf.utils import logger

# Define package-level metadata
__version__ = "0.1.0"
__author__ = "David Steinberg"


def generate_ttl(fname: str, limit: int, use_api_key: bool = True, search: Optional[str] = None):
    """
    Generate a Turtle (.ttl) file from datasets fetched from HuggingFace.

    Args:
        fname (str): The filename for the output Turtle file.
        limit (int): The maximum number of datasets to fetch.

    Returns:
        str: The path to the generated Turtle file.

    Raises:
        ValueError: If the fname or limit parameters are invalid.
    """
    logger.info(f"Retrieving datasets metadata from HuggingFace{' for ' + search if search else ''}.")
    try:
        logger.debug("Fetching datasets with a limit of %d", limit)
        datasets = fetch_datasets(limit, use_api_key, search)

        logger.debug("Converting fetched datasets to RDF format.")
        ttl_path = convert_to_rdf(datasets, fname)
        return ttl_path

    except Exception as e:
        logger.error("An error occurred while generating the Turtle file: %s", e)
        raise

def main():
    """
    Parse command-line arguments and generate a Turtle file.
    """
    parser = argparse.ArgumentParser(description="Generate a Turtle file from HuggingFace datasets.")
    parser.add_argument("search", type=str, nargs="?", default=None, help="Search keywords to filter datasets.")
    # To get a list of search: nargs="*",
    parser.add_argument("--fname", type=str, required=False, default="huggingface.ttl", help="The filename for the output Turtle file.")
    parser.add_argument("--limit", type=int, required=False, default=10, help="The maximum number of datasets to fetch.")
    parser.add_argument("--use_api_key", type=bool, required=False, default=True, help="Use API key for HuggingFace API requests.")

    args = parser.parse_args()
    generate_ttl(args.fname, args.limit, args.use_api_key, args.search)

if __name__ == "__main__":
    main()

__all__ = [
    "convert_to_rdf",
    "fetch_datasets",
    "generate_ttl"
]
