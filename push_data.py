from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import certifi
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()
mongo_password = os.getenv("MONGO_PASSWORD")

uri = f"mongodb+srv://pranavjoshi2210_db_user:{mongo_password}@cluster0.twde3nj.mongodb.net/?appName=Cluster0"


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records

        except Exception as e:
            NetworkSecurityException(e, sys)

    def insert_data_to_mongo(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongoclient = pymongo.MongoClient(uri)
            self.database = self.mongoclient[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)

        except Exception as e:
            NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = r"Network_Data\phisingData.csv"
    DATABASE = "PRANAV"
    COLLECTION = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongo(
        records=records, database=DATABASE, collection=COLLECTION
    )
    print(no_of_records)
