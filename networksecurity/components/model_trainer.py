from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
import os, sys
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metrics.classification_metric import (
    get_classification_score,
)
from networksecurity.utils.ml_utils.models.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import logging
import dagshub

dagshub.init(repo_owner="Pranav4op", repo_name="network-security", mlflow=True)

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            logger.info("Initializing ModelTrainer class")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logger.info("ModelTrainer initialized successfully")
        except Exception as e:
            logger.exception("Error during ModelTrainer initialization")
            raise NetworkSecurityException(e, sys)

    def log_metrics(self, prefix, classificationmetric):
        logger.info(f"Logging {prefix} metrics to MLflow")
        mlflow.log_metric(f"{prefix}_F1", classificationmetric.f1_score)
        mlflow.log_metric(f"{prefix}_Precision", classificationmetric.precision_score)
        mlflow.log_metric(f"{prefix}_Recall", classificationmetric.recall_score)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model training process")

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
            }

            logger.info(f"Models initialized: {list(models.keys())}")

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "loss": ["log_loss", "exponential"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "euclidean", "manhattan"],
                },
            }

            with mlflow.start_run():
                logger.info("MLflow run started")

                model_report = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    param=params,
                )

                logger.info(f"Model evaluation report: {model_report}")

                best_model_name = max(model_report, key=model_report.get)
                best_model = models[best_model_name]

                logger.info(f"Best model selected: {best_model_name}")

                mlflow.log_param("Best Model", best_model_name)
                mlflow.log_params(best_model.get_params())

                logger.info("Predicting on training data")
                y_train_pred = best_model.predict(X_train)
                train_metric = get_classification_score(y_train_pred, y_train)
                self.log_metrics("Train", train_metric)

                logger.info("Predicting on test data")
                y_test_pred = best_model.predict(X_test)
                test_metric = get_classification_score(y_test_pred, y_test)
                self.log_metrics("Test", test_metric)

                mlflow.sklearn.log_model(best_model, "Model")
                logger.info("Model logged to MLflow successfully")

            logger.info("MLflow run completed")

            logger.info("Loading preprocessor object")
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)

            logger.info("Wrapping model with preprocessor")
            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)

            logger.info("Saving trained model object locally")
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=Network_Model,
            )

            logger.info("Model training completed successfully")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

            return model_trainer_artifact

        except Exception as e:
            logger.exception("Error occurred during model training")
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Initiating model trainer pipeline")

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            logger.info("Loading transformed training and testing arrays")

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logger.info(f"Train array shape: {train_arr.shape}")
            logger.info(f"Test array shape: {test_arr.shape}")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logger.info("Calling train_model method")

            model_trainer_artifact = self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            logger.info("Model trainer pipeline completed successfully")
            return model_trainer_artifact

        except Exception as e:
            logger.exception("Error occurred in initiate_model_trainer")
            raise NetworkSecurityException(e, sys)
