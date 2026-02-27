from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys
import logging

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            logger.info("Initializing DataValidation class")

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logger.info(f"Schema loaded from path: {SCHEMA_FILE_PATH}")
            logger.info(
                f"Expected columns count from schema: {len(self._schema_config['columns'])}"
            )

        except Exception as e:
            logger.exception("Error during DataValidation initialization")
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with shape {df.shape}")
            logger.info(f"Columns present: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.exception("Failed while reading data")
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            logger.info("Starting column count validation")

            expected_columns = len(self._schema_config["columns"])

            if "_id" in dataframe.columns:
                logger.info("_id column found during validation, dropping temporarily")
                dataframe = dataframe.drop(columns=["_id"])

            actual_columns = len(dataframe.columns)

            logger.info(f"Expected number of columns: {expected_columns}")
            logger.info(f"Actual number of columns: {actual_columns}")

            if expected_columns != actual_columns:
                logger.error("Column count mismatch detected")
                return False

            logger.info("Column count validation passed")
            return True

        except Exception as e:
            logger.exception("Error during column count validation")
            raise NetworkSecurityException(e, sys)

    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            logger.info("Starting numerical column validation")

            expected_numerical_columns = self._schema_config["numerical_columns"]
            logger.info(f"Expected numerical columns: {expected_numerical_columns}")

            missing_columns = []

            for column in expected_numerical_columns:
                if column not in dataframe.columns:
                    missing_columns.append(column)

            if missing_columns:
                logger.error(f"Missing numerical columns: {missing_columns}")
                return False

            logger.info("Numerical column validation passed")
            return True

        except Exception as e:
            logger.exception("Error during numerical column validation")
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            logger.info("Starting dataset drift detection")
            logger.info(f"Drift detection threshold: {threshold}")

            status = True
            report = {}

            logger.info(f"Base dataset shape: {base_df.shape}")
            logger.info(f"Current dataset shape: {current_df.shape}")

            for column in base_df.columns:

                if column not in current_df.columns:
                    logger.warning(f"Column {column} not found in current dataset")
                    status = False
                    continue

                d1 = base_df[column]
                d2 = current_df[column]

                logger.info(f"Performing KS test for column: {column}")

                ks_test = ks_2samp(d1, d2)

                drift_detected = ks_test.pvalue < threshold

                if drift_detected:
                    logger.warning(
                        f"Drift detected in column: {column}, p-value: {ks_test.pvalue}"
                    )
                    status = False
                else:
                    logger.info(
                        f"No drift in column: {column}, p-value: {ks_test.pvalue}"
                    )

                report[column] = {
                    "p_value": float(ks_test.pvalue),
                    "drift_detected": drift_detected,
                }

            drift_report_path = self.data_validation_config.drift_report
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

            write_yaml_file(drift_report_path, content=report)

            logger.info(f"Drift report saved at: {drift_report_path}")
            logger.info(f"Overall drift detection status: {status}")

            return status

        except Exception as e:
            logger.exception("Error during drift detection")
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Data validation process started")

            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            logger.info(f"Train file path: {train_path}")
            logger.info(f"Test file path: {test_path}")

            train_df = self.read_data(train_path)
            test_df = self.read_data(test_path)

            validation_status = True
            error_messages = []

            logger.info("Validating train dataset column count")
            if not self.validate_number_of_columns(train_df):
                validation_status = False
                error_messages.append("Train dataset column count mismatch")

            logger.info("Validating test dataset column count")
            if not self.validate_number_of_columns(test_df):
                validation_status = False
                error_messages.append("Test dataset column count mismatch")

            logger.info("Validating train dataset numerical columns")
            if not self.validate_numerical_columns(train_df):
                validation_status = False
                error_messages.append("Train dataset missing numerical columns")

            logger.info("Validating test dataset numerical columns")
            if not self.validate_numerical_columns(test_df):
                validation_status = False
                error_messages.append("Test dataset missing numerical columns")

            logger.info("Running dataset drift detection")
            drift_status = self.detect_dataset_drift(train_df, test_df)

            if not drift_status:
                validation_status = False
                error_messages.append("Dataset drift detected")

            os.makedirs(
                os.path.dirname(self.data_validation_config.valid_train_file_path),
                exist_ok=True,
            )

            logger.info("Saving validated train dataset")
            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )

            logger.info("Saving validated test dataset")
            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            logger.info("Validated datasets saved successfully")

            if error_messages:
                logger.error(f"Validation errors encountered: {error_messages}")
            else:
                logger.info("No validation errors detected")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None if validation_status else train_path,
                invalid_test_file_path=None if validation_status else test_path,
                drift_report_file_path=self.data_validation_config.drift_report,
            )

            logger.info(f"Final validation status: {validation_status}")
            logger.info("Data validation process completed successfully")

            return data_validation_artifact

        except Exception as e:
            logger.exception("Data validation failed")
            raise NetworkSecurityException(e, sys)
