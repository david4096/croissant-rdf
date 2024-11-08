import logging
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# You need to use an API Key to make requests from Kaggle api.

API_URL = "https://www.kaggle.com/datasets/"    

def croissant_dataset(dsid):
    """
    Retrieves the 'croissant' metadata file for a specified dataset from Kaggle.

    Args:
        dsid (str): The unique identifier of the dataset from which to retrieve the 'croissant' metadata file.

    Returns:
        dict: A JSON response containing metadata and details from the 'croissant' file for the specified dataset.
        
    """
    request_url = API_URL + str(dsid) + "/croissant/download"
    response = requests.get(request_url)  
    if response.status_code == 200:
        return response.json()
    else:
        logging.info(f"Error downloading: {dsid} with {response.status_code}")
        return None

def get_datasets(limit):
    """
    Retrieves a list of datasets hosted on Hugging Face, up to the specified limit.

    Args:
        limit (int): The maximum number of datasets to retrieve.

    Returns:
        list: A list of dataset objects, each containing metadata for a Hugging Face dataset.
    """
    api = KaggleApi()
    api.authenticate()
    final_datasets_list = []
    page_num = 1

    # Initialize tqdm with the total number of items to fetch
    with tqdm(total=limit, desc="Fetching datasets") as pbar:
        while len(final_datasets_list) < limit:
            curr_dataset_list = api.dataset_list(page=page_num)

            if not curr_dataset_list:  # Stop if no more datasets are returned
                break

            final_datasets_list.extend(curr_dataset_list)
            pbar.update(len(curr_dataset_list))  # Update the progress bar

            page_num += 1

    return final_datasets_list[:limit]

def fetch_datasets(limit):
    """
    Fetches metadata for multiple datasets from Hugging Face, including the 'croissant' metadata file for each.

    This is a wrapper function that retrieves a limited list of datasets using `get_datasets` 
    and then fetches the 'croissant' metadata for each dataset.

    Args:
        limit (int): The maximum number of datasets to retrieve and process.

    Returns:
        list: A list of dictionaries, each containing the 'croissant' metadata for a dataset.
    """
    datasets = get_datasets(limit)
    return [croissant_dataset(dataset) for dataset in tqdm(datasets)]
