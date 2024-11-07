import requests
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


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
    response = requests.get(API_URL + dsid + "/croissant")
        
    return response.json()

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
    dataset_list = api.dataset_list()
    return dataset_list[:limit]

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
    return [croissant_dataset(dataset.id) for dataset in datasets]
