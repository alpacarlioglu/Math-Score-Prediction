

# Math Score Prediction


A machine learning application that predicts student math scores based on various demographic and academic factors. The project implements a complete ML pipeline from data ingestion to model deployment with a web interface.



## Features

- Data Preprocessing: Automated pipeline for handling categorical and numerical features
- Model Training: Implements multiple regression algorithms with hyperparameter tuning
- Model Evaluation: Comprehensive metrics for model performance assessment
- Web Interface: User-friendly Flask application for real-time predictions
- Modular Design: Well-structured codebase for easy maintenance and extension


## Tech Stack

**ML Libraries**: scikit-learn, XGBoost, CatBoost

**Data Processing**: Pandas, NumPy

**Web Framework**: Flask

**Visualization**: Matplotlib, Seaborn

**Serialization**: Dill/Pickle

```bash
Math Score Prediction/
│
├── artifacts/               # Trained models and data artifacts
├── logs/                    # Application logs
├── notebooks/  # Exploratory data analysis notebooks  
│   ├── EDA STUDENT PERFORMANCE.ipynb     
│   ├── MODEL TRAINING.ipynb       
├── src/                     
│   ├── components/          # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Pipelines for training and prediction
│   │   └── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py         # Custom exception handling
│   └── utils.py             # Utility functions
├── templates/               # HTML templates
│   ├── home.html
│   └── index.html
├── application.py           # Flask application entry point
├── requirements.txt         # Project dependencies
└── setup.py                 # Package configuration
```
# Installation

**1- Clone the repository:**

```bash
  git clone https://github.com/alpacarlioglu/Math Score Prediction.git
  cd Math Score Prediction
```
**2- Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```
**3- Install dependencies:**
```bash
pip install -r requirements.txt
```
## Usage

### Web Interface

**1- Run the Flask application:**
```
python application.py
```

**2- Open your browser and navigate to:**
```
http://localhost:5000
```

**3- Fill in the student information form and submit to get the predicted math score.**

## Running the ML Pipeline
To train the model from scratch:

This will:

Ingest and process the data
Train multiple regression models
Select the best performing model
Save the model and preprocessor artifacts
Dataset
The project uses a student performance dataset with the following features:

Gender
Race/Ethnicity
Parental level of education
Lunch type
Test preparation course completion
Reading scores
Writing scores
Target variable: Math scores

Model Details
The system evaluates multiple regression models:

Random Forest Regressor
Decision Tree Regressor
Gradient Boosting Regressor
Linear Regression
K-Neighbors Regressor
XGBoost Regressor
CatBoost Regressor
AdaBoost Regressor
Models are evaluated based on R² score, MSE, and MAE metrics.

Author
Alp Acarlıoğlu - alpacarlioglu@gmail.com