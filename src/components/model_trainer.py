import pandas as pd
import os
import sys
import mlflow
import pickle
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'models', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train):
        try:
            logging.info('Model Training Started.')

            # params for CatBoostRegressor
            catboost_params = {
            'iterations': 1485,
            'learning_rate': 0.09566208338756467,
            'depth': 8,
            'l2_leaf_reg': 4,
            'border_count': 149,
            'random_strength': 0.3221392635228121,
            'bagging_temperature': 0.16171642890213989
            }

            tuned_CBR = CatBoostRegressor(**catboost_params, random_state=42, verbose=False)

            # params for XGBRegressor
            xgb_params = {
            'n_estimators': 535,
            'learning_rate': 0.057582817878155085,
            'max_depth': 10,
            'min_child_weight': 10,
            'subsample': 0.7124624923071757,
            'colsample_bytree': 0.7561466714445289,
            'gamma': 0.0433184458126587,
            'reg_alpha': 0.934442472471591,
            'reg_lambda': 2.07724971668887
            }

            tuned_XGB = XGBRegressor(**xgb_params, random_state=42)

            # params for LGBMRegressor
            lgbm_params = {
            'n_estimators': 214,
            'learning_rate': 0.10979784611692994,
            'max_depth': 9,
            'num_leaves': 75,
            'min_child_samples': 33,
            'subsample': 0.939496220559003,
            'colsample_bytree': 0.8775803119044167,
            'reg_alpha': 0.8615502636228614,
            'reg_lambda': 0.8363793385956172
            }

            tuned_LGBM = LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)

            # ensemble of tuned models
            ensemble = StackingRegressor(
                estimators=[
                    ('catboost', tuned_CBR),
                    ('lgbm', tuned_LGBM),
                    ('xgb', tuned_XGB)
            ], final_estimator=Ridge(), cv=5)

            mlflow.set_experiment('House Price Prediction')
            with mlflow.start_run(run_name='model_training'):

                mlflow.log_params({
                    'catboost_params': catboost_params,
                    'xgb_parmas': xgb_params,
                    'lgbm_params': lgbm_params,
                    'final_estimator': 'Ridge',
                    'cv_folds': 5
                })

                logging.info('Training Stacking Ensemble model...')
                ensemble.fit(X_train, y_train)

                mlflow.sklearn.log_model(ensemble, name='stacking_ensemble_model')

                # saving the trained model
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

                with open(self.model_trainer_config.trained_model_path, 'wb') as f:
                    pickle.dump(ensemble, f)

                logging.info('Model training completed.')


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':

    # path for processed training dataset
    df_train_processed_path = 'artifacts/processed/train.csv'
    
    # loading the processed training dataset
    df_train = pd.read_csv(df_train_processed_path)

    # separating the independent and dependent variables
    X_train = df_train.drop('MedHouseVal', axis=1)
    y_train = df_train['MedHouseVal']

    # initiating model training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train, y_train)