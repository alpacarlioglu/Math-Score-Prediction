import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

def save_obj(obj, file_path):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)