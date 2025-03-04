import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import logging
from sklearn.metrics import (r2_score, roc_auc_score, mean_squared_error, mean_absolute_error)
from sklearn.model_selection import GridSearchCV

def save_obj(obj, file_path):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
        
    except Exception as e:
        logging.info(f'Cant load the {file} file')
        raise CustomException(e, sys)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Create GridSearchCV for this model
            param_grid = params.get(model_name, {})
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,  # 3-fold cross-validation
                    scoring='r2',
                    n_jobs=-1  # Use all available cores
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model from grid search
                best_model = grid_search.best_estimator_
                y_preds = best_model.predict(X_test)
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # No hyperparameters to tune
                model.fit(X_train, y_train)
                y_preds = model.predict(X_test)
            
            r2 = r2_score(y_test, y_preds)
            mse = mean_squared_error(y_test, y_preds)
            mae = mean_absolute_error(y_test, y_preds)
            
            report[model_name] = {
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'model': model if not param_grid else best_model
            }
        
        return dict(sorted(report.items(), key=lambda x: x[1]['r2_score'], reverse=True))
        
    except Exception as e:
        logging.error(f'Evaluation failed: {str(e)}')
        raise CustomException(e, sys)
    
