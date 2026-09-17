# 30-Day Patient Readmission Prediction

A machine learning case study that uses **Logistic Regression with L2 regularization** to predict whether a patient is likely to be readmitted to the hospital within 30 days.

## Objective

The goal of this project is to build a binary classification model that identifies patients at higher risk of 30-day readmission using historical patient information.

## Features

The model uses patient-related features such as:

* Diagnosis codes
* Vital signs
* Previous hospital visits
* Other relevant patient record attributes

## Machine Learning Approach

* Data preprocessing and cleaning
* Feature preparation
* Logistic Regression
* **L2 regularization** to reduce overfitting
* Model evaluation using **ROC-AUC**
* Analysis of classification errors

## Evaluation

The model is evaluated primarily using **ROC-AUC**, which measures its ability to distinguish between patients who are readmitted and those who are not.

The project also considers the clinical consequences of prediction errors:

* **False Negative:** A high-risk patient is predicted as low-risk, potentially causing the patient to miss additional monitoring or intervention.
* **False Positive:** A low-risk patient is predicted as high-risk, potentially resulting in unnecessary monitoring, testing, or healthcare resource usage.

## Key Learning

This case study demonstrates how machine learning can be applied to healthcare risk prediction while considering that different types of classification errors can have different real-world costs.
