import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from huggingface_hub import list_datasets
from rich.progress import track

from croissant_rdf.utils import logger

headers = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"} if os.environ.get("HF_API_KEY") else {}

API_URL = "https://huggingface.co/api/datasets/"


def croissant_dataset(dsid, use_api_key=True):
    """Retrieve the 'croissant' metadata file for a specified dataset from HuggingFace.

    Args:
        dsid (str): The unique identifier of the dataset from which to retrieve the 'croissant' metadata file.
        use_api_key (bool): A boolean determining if an API Key will be used to make the requests to Huggingface.

    Returns:
        dict: A JSON response containing metadata and details from the 'croissant' file for the specified dataset.

    """
    if use_api_key:
        response = requests.get(API_URL + dsid + "/croissant", headers=headers, timeout=30)
    else:
        response = requests.get(API_URL + dsid + "/croissant", timeout=30)

    return response.json()


def get_datasets(limit: int, search: Optional[str] = None):
    """Retrieve a list of datasets hosted on HuggingFace, up to the specified limit.

    Args:
        limit (int): The maximum number of datasets to retrieve.

    Returns:
        list: A list of dataset objects, each containing metadata for a HuggingFace dataset.
    """
    return list(list_datasets(limit=limit, search=search))


def fetch_datasets(limit: int, use_api_key: bool = True, search: Optional[str] = None):
    """Fetch metadata for multiple datasets from HuggingFace, including the 'croissant' metadata file for each.

    This is a wrapper function that retrieves a limited list of datasets using `get_datasets`
    and then fetches the 'croissant' metadata for each dataset.

    Args:
        limit (int): The maximum number of datasets to retrieve and process.
        use_api_key (bool): A boolean determining if an API Key will be used to make the requests to Huggingface.

    Returns:
        list: A list of dictionaries, each containing the 'croissant' metadata for a dataset.
    """
    logger.info(f"Fetching {limit} datasets from HuggingFace.")

    try:
        datasets = get_datasets(limit, search)
        logger.info(f"Got {len(datasets)} datasets from HuggingFace.")
    except Exception as e:
        logger.error(f"Error fetching datasets: {e}")
        return []

    # Create a thread pool
    results = []
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit each dataset to the thread pool
            futures = {
                executor.submit(croissant_dataset, dataset.id): dataset.id for dataset in datasets
            }  # Use tqdm to show progress
            for future in track(as_completed(futures), "Fetching datasets", len(futures)):
                dataset_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_id}: {e}")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Shutting down...")
        executor.shutdown(wait=False)  # Cancel remaining futures
        raise  # Reraise the exception to exit immediately

    return results
