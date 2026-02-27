import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import (
    DataTransformationConfig,
)
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object
import logging

logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logger.info("DataTransformation class initialized successfully")
        except Exception as e:
            logger.exception("Error during DataTransformation initialization")
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with shape {df.shape}")
            return df
        except Exception as e:
            logger.exception("Failed while reading data")
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logger.info("Entered get_data_transformer_object method")
        try:
            logger.info(
                f"Initializing KNNImputer with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor: Pipeline = Pipeline([("imputer", imputer)])
            logger.info("Pipeline object created successfully")
            return processor
        except Exception as e:
            logger.exception("Error while creating data transformer object")
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Entered initiate_data_transformation method")
        try:
            logger.info("Starting data transformation process")

            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            logger.info(f"Train dataframe shape: {train_df.shape}")
            logger.info(f"Test dataframe shape: {test_df.shape}")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            logger.info(
                f"Input train shape: {input_feature_train_df.shape}, Target train shape: {target_feature_train_df.shape}"
            )
            logger.info(
                f"Input test shape: {input_feature_test_df.shape}, Target test shape: {target_feature_test_df.shape}"
            )

            logger.info(
                f"Input feature columns: {input_feature_train_df.columns.tolist()}"
            )
            logger.info(f"Input train dtypes:\n{input_feature_train_df.dtypes}")

            preprocessor = self.get_data_transformer_object()

            logger.info("Fitting preprocessor on training data")
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            logger.info("Transforming training data")
            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df
            )

            logger.info("Transforming test data")
            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df
            )

            logger.info(
                f"Transformed train feature shape: {transformed_input_train_feature.shape}"
            )
            logger.info(
                f"Transformed test feature shape: {transformed_input_test_feature.shape}"
            )

            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df),
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df),
            ]

            logger.info(f"Final train array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")

            logger.info("Saving transformed train array")
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            logger.info("Saving transformed test array")
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            logger.info("Saving preprocessor object")
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )

            logger.info("Data transformation completed successfully")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            return data_transformation_artifact

        except Exception as e:
            logger.exception("Error occurred during data transformation")
            raise NetworkSecurityException(e, sys)
