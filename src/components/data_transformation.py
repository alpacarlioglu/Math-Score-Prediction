import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_obj
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTraining


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    """
    Data transformation component responsible for preprocessing features.
    
    This class handles the preprocessing of both numerical and categorical features
    using scikit-learn pipelines. It creates a preprocessing object that can:
    
    - Handle missing values in numerical features using median imputation
    - Handle missing values in categorical features using most frequent value imputation
    - Scale numerical features using StandardScaler
    - Encode categorical features using OneHotEncoder

    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        """
        Creates and returns a preprocessing object for feature transformation.
        
        This method defines the preprocessing steps for both numerical and categorical
        columns in the dataset. It creates separate pipelines for each type and 
        combines them using a ColumnTransformer.
        
        Returns:
        A preprocessor object that can transform
        raw features into a format suitable for machine learning models.
                
        """
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                                'gender','race_ethnicity',
                                'parental_level_of_education',
                                'lunch','test_preparation_course',
                                ]
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )
            
            logging.info('Preprocessing has completed')
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name= "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_features = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train = train_df[target_column_name]

            test_features=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test=test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_features_preprocessed = preprocessing_obj.fit_transform(train_features)
            test_features_preprocessed = preprocessing_obj.transform(test_features)

            train_arr = np.c_[
                train_features_preprocessed, np.array(target_feature_train)
            ]
            test_arr = np.c_[test_features_preprocessed, np.array(target_feature_test)]

            logging.info(f"Saved preprocessing object.")

            save_obj (
                obj=preprocessing_obj,
                file_path=self.data_transformation_config.preprocessor_obj_path
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    
if __name__ == '__main__':
    data = DataIngestion()
    train_data, test_data = data.initiate_data_ingestion()
    
    preprocessor = DataTransformation()
    train_arr, test_arr, preprocessor_path = preprocessor.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTraining()
    print(f"{model_trainer.initiate_model_trainer(train_arr, test_arr):.4f}")