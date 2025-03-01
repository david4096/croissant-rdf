import argparse
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from rdflib import Graph, URIRef
from rich.progress import track

from croissant_rdf.utils import chunk_data, logger


class CroissantHarvester(ABC):
    """Abstract base class for harvesting and processing Croissant metadata for datasets.

    This class defines a common interface for dataset harvesters targeting various providers.
    Child classes should override the abstract methods to implement custom fetching logic.
    The `api_url` attribute can be overridden either via subclass definition or during instantiation.
    """

    api_url: str = ""  # intended to be overridden in child classes

    def __init__(
        self,
        fname: str = "croissant_metadata.ttl",
        limit: int = 10,
        use_api_key: bool = True,
        search: Optional[str] = None,
        base_url: str = "https://w3id.org/croissant-rdf/data/",
        api_url: Optional[str] = None,
    ):
        """Initialize a Croissant metadata Harvester instance for a specific provider.

        Args:
            fname (str): The filename for the output Turtle file.
            limit (int): The maximum number of datasets to fetch.
            use_api_key (bool): Use API key for API requests.
            search (str): Search keywords to filter datasets.
            base_url (str): The base URL for the RDF graph, used as a prefix in generated RDF triples.
            api_url (str): The base URL for the API endpoint to fetch dataset metadata.
        """
        self.fname = fname
        self.limit = limit
        self.search = search
        self.base_url = base_url
        self.use_api_key = use_api_key
        self.api_url = api_url if api_url is not None else self.__class__.api_url

    @abstractmethod
    def fetch_datasets_ids(self) -> List[str]:
        """Fetch a list of dataset identifiers from the provider."""
        pass

    @abstractmethod
    def fetch_dataset_croissant(self, dataset_id: str) -> Optional[Union[Dict, List]]:
        """Fetch the croissant metadata for a specific dataset from the provider.

        Args:
            dataset_id (str): The ID of the dataset."""
        pass

    def fetch_datasets_croissant(self) -> List[Dict]:
        """Fetch metadata for multiple datasets, using threading where applicable."""
        try:
            datasets = self.fetch_datasets_ids()
            logger.info(f"Retrieved {len(datasets)} datasets ID.")
        except Exception as e:
            logger.error(f"Error fetching datasets: {e}")
            return []

        results = []
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.fetch_dataset_croissant, dataset): dataset for dataset in datasets}
                for future in track(as_completed(futures), "Fetching datasets metadata", len(futures)):
                    dataset_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Error processing dataset {dataset_id}: {e}")
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user. Shutting down...")
            executor.shutdown(wait=False)
            raise
        return results

    def convert_to_rdf(self, data) -> str:
        """Take a JSON-serializable data structure, converts it to RDF using
        JSON-LD format, and serializes it into Turtle format, saving it to the specified file.

        Args:
            data (list|dict): The JSON-serializable data structure to convert to RDF.

        Returns:
            str: The path to the generated turtle file.
        """
        total_items = len(data)
        chunk_size = total_items // 100 if total_items > 100 else 1
        logger.info(
            f"Loading Croissant metadata JSON-LD to RDF graph. Total items: {total_items}, Chunk size: {chunk_size}"
        )
        g = Graph()
        g.bind("cr", "http://mlcommons.org/croissant/")
        g.bind("crdf", self.base_url)
        start_time = time.time()
        for chunk in track(chunk_data(data, chunk_size), "Parsing data", total_items):
            for item in chunk:
                item_json_ld = json.dumps(item)
                g.parse(data=item_json_ld, format="json-ld", base=URIRef(self.base_url))

        logger.info(
            f"Parsing completed in {time.time() - start_time:.2f}s, writing {len(g)} RDF triples to file {self.fname}"
        )
        start_time = time.time()
        g.serialize(destination=self.fname, format="ttl")
        logger.info(f"Serialization completed in {time.time() - start_time:.2f}s")
        return self.fname

    def generate_ttl(self) -> str:
        """Fetch datasets and generate a Turtle file.

        Returns:
            str: The path to the generated turtle file.

        Raises:
            Exception: If there was an error generating the turtle file.
        """
        logger.info(f"Searching {self.limit} datasets metadata{f' for `{self.search}`' if self.search else ''}.")
        try:
            start_time = time.time()
            datasets = self.fetch_datasets_croissant()
            logger.info(
                f"Retrieved Croissant metadata JSON-LD for {len(datasets)} datasets in {time.time() - start_time:.2f}s"
            )
            ttl_path = self.convert_to_rdf(datasets)

            return ttl_path
        except Exception as e:
            logger.error(f"Error generating TTL file: {e}")
            raise

    @classmethod
    def cli(cls):
        """Parse command-line arguments and generate a turtle file from harvested Croissant metadata."""
        parser = argparse.ArgumentParser(description="Generate a Turtle file from datasets.")
        parser.add_argument(
            "search",
            type=str,
            nargs="?",
            default=None,
            help="Search keywords to filter datasets.",
        )
        parser.add_argument(
            "--fname",
            type=str,
            required=False,
            default="croissant_metadata.ttl",
            help="The filename for the output Turtle file.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            required=False,
            default=10,
            help="The maximum number of datasets to fetch.",
        )
        parser.add_argument(
            "--use_api_key",
            type=bool,
            required=False,
            default=True,
            help="Use API key for API requests.",
        )
        args = parser.parse_args()

        harvester = cls(
            fname=args.fname,
            limit=args.limit,
            use_api_key=args.use_api_key,
            search=args.search,
        )
        harvester.generate_ttl()
