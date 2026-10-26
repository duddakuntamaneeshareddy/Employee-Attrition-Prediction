# Employee Attrition Prediction using Apache Spark MLlib

An end-to-end machine learning project for predicting employee attrition using Apache Spark MLlib, AWS S3, AWS EC2, and Streamlit.

## Overview

This project analyzes employee data to identify attrition patterns and predict employees who may be at risk of leaving the organization.

The project combines cloud storage, distributed data processing, machine learning, and interactive visualization.

## Technologies

- Python
- Apache Spark
- PySpark MLlib
- AWS S3
- AWS EC2
- Streamlit
- Pandas
- Plotly
- Google Colab

## Machine Learning Workflow

1. Load employee data from AWS S3
2. Clean and preprocess the dataset
3. Encode categorical features
4. Assemble numerical and categorical features
5. Split the dataset into training and testing data
6. Handle class imbalance using class weights
7. Train a Logistic Regression model using Spark MLlib
8. Apply a classification threshold of 0.4
9. Generate predictions and evaluation results
10. Store model outputs in AWS S3
11. Analyze results using the Streamlit dashboard

## Key Result

Accuracy: 75.59% | Attrition Recall: 77.78%

## Dashboard

The Streamlit dashboard provides:

- Employee attrition overview
- Department analysis
- Job role analysis
- Gender analysis
- Overtime analysis
- Workforce insights
- Model evaluation
- Confusion matrix
- Interactive data exploration
- Data filtering and download

## Project Structure

Employee-Attrition-Prediction/
+-- app.py
+-- README.md
+-- requirements.txt
+-- src/
¦   +-- employee_attrition_spark.py
+-- notebooks/
    +-- Employee_Attrition_Prediction_Spark_MLlib.ipynb

## AWS Architecture

Employee Dataset
        ?
      AWS S3
        ?
     AWS EC2
        ?
  Apache Spark
        ?
   Spark MLlib
        ?
    Predictions
        ?
      AWS S3
        ?
Streamlit Dashboard

## Running the Dashboard

Install the required packages:

pip install -r requirements.txt

Run the dashboard:

streamlit run app.py

## Security

No AWS credentials, private keys, passwords, environment files, or other sensitive authentication information are included in this repository.

## Objective

The objective of this project is to build a practical employee attrition prediction system using cloud computing, distributed processing, machine learning, and interactive analytics.
