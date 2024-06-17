# Arpit_Datahack
## Predicting Flu Vaccination Uptake

Welcome to my project repository for the DataHack hackathon hosted by IIT Guwahati on the Geeks for Geeks platform. The goal of this project is to predict the likelihood of individuals receiving their xyz and seasonal flu vaccines based on various demographic, behavioral, and opinion-based features.

### Problem Statement

The objective is to create a model that predicts the probabilities for two outcomes:

* **xyz_vaccine**: Whether an individual received the xyz flu vaccine.
* **seasonal_vaccine**: Whether an individual received the seasonal flu vaccine.

This is a multilabel classification problem because respondents could receive none, one, or even both vaccines.

### Dataset Overview

The dataset includes 36 features covering various aspects like respondents' health beliefs, behaviors, demographics, and opinions about flu vaccines. Each respondent is identified by a unique **respondent_id**.

### Approach

#### Data Preprocessing

* **Handling Missing Values**: I used a SimpleImputer to fill missing values with the most frequent ones.
* **Encoding Categorical Variables**: Categorical columns were encoded using LabelEncoder to convert them into a numeric format suitable for modeling.

#### Model Building

* **Feature Scaling**: Features were standardized using StandardScaler to ensure all variables contributed equally to the model.
* **Model Selection**: I chose RandomForestClassifier wrapped in MultiOutputClassifier to handle the multilabel nature of the problem efficiently.

#### Model Evaluation

* **Evaluation Metric**: Evaluated model performance using ROC AUC score, which measures the ability to distinguish between positive and negative classes effectively.

### Results and Submission

After training on the provided data, predictions were made on the test set and formatted according to submission guidelines. Results for both vaccines are saved in **submission.csv**, included in this repository.

### Files Included

* **Code Files**: Python scripts (predict_flu_vaccination.py and Jupyter Notebook predict_flu_vaccination.ipynb) encompassing the entire workflow from data preprocessing to model evaluation.
* **Datasets**: The Dataset folder contains the original CSV files provided for training and testing.

I hope you find this repository insightful! For any questions or feedback, please reach out :>
