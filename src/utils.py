import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import logging
from sklearn.metrics import (r2_score, roc_auc_score, mean_squared_error, mean_absolute_error)

def save_obj(obj, file_path):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_preds = model.predict(X_test)
            
            # Use regression metrics only
            r2 = r2_score(y_test, y_preds)
            mse = mean_squared_error(y_test, y_preds)
            mae = mean_absolute_error(y_test, y_preds)
            
            report[model_name] = {
                'r2_score': r2,
                'mse': mse,
                'mae': mae
            }
        
        # Sort models by R2 score (higher is better)
        sorted_report = dict(sorted(
            report.items(), 
            key=lambda item: item[1]['r2_score'], 
            reverse=True
        ))
        
        return sorted_report
        
    except Exception as e:
        logging.info('Evaluation failed')
        raise CustomException(e, sys)