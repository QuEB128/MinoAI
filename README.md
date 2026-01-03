# MinoAI: Intelligent Price Prediction with Machine Learning


## Overview

MinoAI is an end-to-end machine learning system designed to predict Airbnb listing prices using structured data and supervised learning techniques. The project follows industry-standard data science and machine learning practices, covering the full lifecycle from exploratory data analysis to deployment.

The system is deployed as a Streamlit web application, allowing users to make real-time price predictions through an intuitive and professional user interface.


## Problem Statement

Pricing short-term rental properties accurately is a challenging task influenced by multiple variables such as location, room type, availability, and guest engagement. Many hosts rely on intuition or manual comparisons, which often leads to underpricing, overpricing, or inconsistent revenue outcomes.

Traditional pricing approaches fail to capture nonlinear relationships and interactions between features. There is a clear need for a data-driven and automated pricing system that can learn from historical data and generalize to new listings effectively.


## Solution

MinoAI solves this problem by leveraging a supervised machine learning approach using a Random Forest Regressor. The model is trained on a cleaned and engineered Airbnb dataset to learn complex relationships between listing attributes and price.

The trained model is serialized and deployed via a Streamlit web application, enabling users to input listing details and instantly receive accurate price predictions. This ensures usability, scalability, and real-world applicability.


## Key Features

- End-to-end machine learning pipeline  
- Comprehensive exploratory data analysis (EDA)  
- Robust data cleaning and preprocessing  
- Feature engineering with categorical encoding  
- Random Forest regression model  
- Model evaluation using MAE, RMSE, and R²  
- Model persistence using Joblib  
- Interactive Streamlit-based prediction interface  


## Project Structure
```
MinoAI_ML_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── MinoAI_dataset.csv
│
├── notebooks/
│   └── eda_and_model.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   ├── random_forest_model.pkl
│   └── feature_columns.pkl
│
├── results/
│   └── figures/
│
├── report/
│
├── app.py
│
└── venv/
```

## Dataset Description

The dataset contains Airbnb listing information, including:

- Neighbourhood group  
- Room type  
- Minimum nights  
- Number of reviews  
- Reviews per month  
- Availability over 365 days  
- Price (target variable)  

The data is preprocessed to handle missing values, outliers, and categorical variables before modeling.


## Machine Learning Workflow

### Exploratory Data Analysis (EDA)

- Dataset inspection and summary statistics  
- Visualization of distributions and categorical counts  
- Correlation analysis and pattern discovery  

### Data Cleaning

- Handling missing values  
- Removing duplicates  
- Treating outliers using the IQR method  
- Dropping irrelevant or inconsistent columns  

### Feature Engineering

- One-hot encoding of categorical variables  
- Feature alignment to ensure consistency during inference  

### Model Training

- Algorithm: Random Forest Regressor  
- Train-test split: 80/20  
- Fixed random state for reproducibility  

### Model Evaluation

The model is evaluated using industry-standard regression metrics:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

Sample performance results:

- MAE ≈ 37  
- RMSE ≈ 53  
- R² ≈ 0.59  


## Deployment: Streamlit Application

The trained Random Forest model is deployed using Streamlit, allowing users to:

- Enter listing details via a clean UI  
- Generate instant price predictions  
- Interact with a professional ML-powered application  

To run the application locally:

streamlit run app.py

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/QuEB128/MinoAI.git 
cd MinoAI_ML_Project
```
Install dependencies:

pip install -r requirements.txt  

Activate the virtual environment 

Train Model:
```bash
jupyter notebook
# Open: notebooks/eda_and_model.ipynb
# Run all cells sequentially
```

and run the Streamlit application.
```bash
streamlit run app.py
```

## Technologies Used

### Programming and Machine Learning

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

### Visualization and Deployment

- Matplotlib  
- Seaborn  
- Streamlit  


## Results and Insights

- Location and room type are strong predictors of price  
- Entire homes/apartments tend to have higher predicted prices  
- Availability and review frequency significantly influence pricing  
- Random Forest effectively captures nonlinear feature interactions  


## Future Improvements

- Hyperparameter tuning using GridSearch or RandomizedSearch  
- Cross-validation for improved generalization  
- Prediction confidence intervals  
- Batch predictions using CSV uploads  
- Cloud deployment (Streamlit Cloud or Docker)  


## Author

Emmanuel Quartey  

MinoAI is a complete demonstration of a real-world machine learning workflow, from raw data to a deployed predictive system.
