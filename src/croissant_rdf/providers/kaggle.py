import requests
from kaggle.api.kaggle_api_extended import KaggleApi

from croissant_rdf.croissant_harvester import CroissantHarvester

__author__ = "David Steinberg,Nelson Quinones"


class KaggleHarvester(CroissantHarvester):
    api_url = "https://www.kaggle.com/datasets/"

    def fetch_datasets_ids(self):
        api = KaggleApi()
        api.authenticate()
        return api.dataset_list(search=self.search)[: self.limit]

    def fetch_dataset_croissant(self, dataset_id: str):
        response = requests.get(self.api_url + str(dataset_id) + "/croissant/download", timeout=30)
        return response.json() if response.status_code == 200 else None


def main():
    KaggleHarvester.cli()


__all__ = ["KaggleHarvester"]
