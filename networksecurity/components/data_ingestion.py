from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig
import os
import sys
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import logging

logger = logging.getLogger(__name__)

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("DataIngestion class initialized successfully")
        except Exception as e:
            logger.exception("Error during DataIngestion initialization")
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        try:
            logger.info("Starting export_collection_as_dataframe")

            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logger.info(
                f"Connecting to MongoDB database: {db_name}, collection: {collection_name}"
            )

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[db_name][collection_name]

            logger.info("Fetching data from MongoDB")
            df = pd.DataFrame(list(collection.find()))

            logger.info(f"Data fetched successfully with shape: {df.shape}")
            logger.info(f"Columns before dropping _id: {df.columns.tolist()}")

            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], axis=1, inplace=True)
                logger.info("_id column dropped successfully")

            logger.info(f"Columns after preprocessing: {df.columns.tolist()}")

            df.replace({"na": np.nan}, inplace=True)

            logger.info("Replaced 'na' values with np.nan")
            logger.info("Completed export_collection_as_dataframe")

            return df

        except Exception as e:
            logger.exception("Error occurred in export_collection_as_dataframe")
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            logger.info("Starting export_data_into_feature_store")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logger.info(f"Saving feature store file at: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logger.info("Feature store file saved successfully")
            logger.info(f"Feature store dataframe shape: {dataframe.shape}")

            return dataframe

        except Exception as e:
            logger.exception("Error occurred in export_data_into_feature_store")
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logger.info("Starting split_data_as_train_test")

            logger.info(f"Input dataframe shape before split: {dataframe.shape}")

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logger.info("Performed train test split")
            logger.info(f"Train set shape: {train_set.shape}")
            logger.info(f"Test set shape: {test_set.shape}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logger.info(
                f"Saving train file at: {self.data_ingestion_config.training_file_path}"
            )
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True,
            )

            logger.info(
                f"Saving test file at: {self.data_ingestion_config.testing_file_path}"
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True,
            )

            logger.info("Train and Test files exported successfully")
            logger.info("Completed split_data_as_train_test")

        except Exception as e:
            logger.exception("Error occurred in split_data_as_train_test")
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logger.info("Started initiate_data_ingestion pipeline")

            dataframe = self.export_collection_as_dataframe()

            dataframe = self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logger.info("Data ingestion pipeline completed successfully")

            return data_ingestion_artifact

        except Exception as e:
            logger.exception("Error occurred during initiate_data_ingestion")
            raise NetworkSecurityException(e, sys)
