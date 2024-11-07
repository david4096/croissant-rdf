import argparse
import logging

from .fetch_data import fetch_datasets
from .generate_rdf import convert_to_rdf


# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define package-level metadata
__version__ = "0.1.0"
__author__ = "David Steinberg,Nelson Quinones"

def generate_ttl(fname, limit, use_api_key=True):
    """
    Generate a Turtle (.ttl) file from datasets fetched from Hugging Face.

    Args:
        fname (str): The filename for the output Turtle file.
        limit (int): The maximum number of datasets to fetch.

    Returns:
        str: The path to the generated Turtle file.

    Raises:
        ValueError: If the fname or limit parameters are invalid.
    """
    logger.info("Starting the process to generate a Turtle file.")
    try:
        logger.debug("Fetching datasets with a limit of %d", limit)
        datasets = fetch_datasets(limit)

        logger.debug("Converting fetched datasets to RDF format.")
        ttl_path = convert_to_rdf(datasets, fname)

        logger.info("Turtle file generated successfully: %s", ttl_path)
        return ttl_path

    except Exception as e:
        logger.error("An error occurred while generating the Turtle file: %s", e)
        raise

def main():
    """
    Parse command-line arguments and generate a Turtle file.
    """
    parser = argparse.ArgumentParser(description="Generate a Turtle file from Hugging Face datasets.")
    parser.add_argument("--fname", type=str, required=False, default="huggingface.ttl", help="The filename for the output Turtle file.")
    parser.add_argument("--limit", type=int, required=False, default=10, help="The maximum number of datasets to fetch.")
    parser.add_argument("--use_api_key", type=bool, required=False, default=True, help="Use API key for Hugging Face API requests.")

    args = parser.parse_args()
    
    generate_ttl(args.fname, args.limit, args.use_api_key)

if __name__ == "__main__":
    main()

__all__ = [
    "generate_ttl",
    "fetch_datasets",
    "convert_to_rdf"
]
