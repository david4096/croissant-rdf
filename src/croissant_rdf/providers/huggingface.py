import os

import requests
from huggingface_hub import list_datasets

from croissant_rdf.croissant_harvester import CroissantHarvester

__author__ = "David Steinberg"


class HuggingfaceHarvester(CroissantHarvester):
    api_url = "https://huggingface.co/api/datasets/"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"} if os.environ.get("HF_API_KEY") else {}

    def fetch_datasets_ids(self):
        return [dataset.id for dataset in list(list_datasets(limit=self.limit, search=self.search))]

    def fetch_dataset_croissant(self, dataset_id: str):
        response = requests.get(
            self.api_url + dataset_id + "/croissant", headers=self.headers if self.use_api_key else {}, timeout=30
        )
        return response.json()


def main():
    HuggingfaceHarvester.cli()


__all__ = ["HuggingfaceHarvester"]
