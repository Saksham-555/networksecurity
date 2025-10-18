import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='Saksham-555', repo_name='networksecurity', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Saksham-555/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Saksham-555"
os.environ["MLFLOW_TRACKING_PASSWORD"]="69ded5713d6ec6f52453bf6cf60201fcc4fcd9a6"





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, train_metric, test_metric):
        """
        Track model training metrics in MLflow
        Logs metrics, parameters, and model as artifact (DagsHub compatible)
        """
        # mlflow.set_registry_uri("https://dagshub.com/Saksham-555/networksecurity.mlflow")
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log train metrics
            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            # Log test metrics
            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)

            # Log model name as a parameter for reference
            mlflow.log_param("model_class", type(best_model).__name__)

            # Try to log model hyperparameters
            try:
                params = best_model.get_params()
                for key in ['n_estimators', 'learning_rate', 'max_depth', 'criterion', 'subsample']:
                    if key in params:
                        mlflow.log_param(key, params[key])
            except Exception as e:
                logging.info(f"Could not log all model parameters: {e}")

            # Save model locally first, then log as artifact
            try:
                import tempfile
                import joblib

                # Create temporary file
                with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix='.pkl') as f:
                    joblib.dump(best_model, f.name)
                    temp_model_path = f.name

                # Log the model file as an artifact (not as mlflow model)
                mlflow.log_artifact(temp_model_path, artifact_path="model")

                # Clean up temp file
                os.remove(temp_model_path)
                logging.info(f"✅ Model artifact logged successfully")
            except Exception as e:
                logging.warning(f"⚠️ Could not log model artifact to DagsHub: {e}")
                logging.info("Model is saved locally in artifacts folder")

            logging.info(f"✅ Metrics logged to MLflow for {type(best_model).__name__}")      

                            
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        # Get train predictions and metrics
        y_train_pred=best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        # Get test predictions and metrics
        y_test_pred = best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
        ## Track Both train and test metrics in ONE mlflow run
        self.track_mlflow(best_model, classification_train_metric, classification_test_metric)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


        # y_test_pred=best_model.predict(x_test)
        # classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        # self.track_mlflow(best_model,classification_test_metric)

        # preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)

        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        


       
    
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)