#First of all, importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

#Reading/Loading data from csv files provided
# Reading/Loading data from csv files provided
train_features = pd.read_csv('Dataset/training_set_features.csv')
train_labels = pd.read_csv('Dataset/training_set_labels.csv')
test_features = pd.read_csv('Dataset/test_set_features.csv')
submission_format = pd.read_csv('Dataset/submission_format.csv')

#Just previewing stuffs
print("Training features preview:")
print(train_features.head())

print("Training labels preview:")
print(train_labels.head())

#Imputing missing values : Here i am using simple imputer to replace missing values with most frequent one
imputer = SimpleImputer(strategy='most_frequent')
train_features_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns)

#Here i am identifying categorical columns and encoding them using LabelEncoder to convert them into numeric form
categorical_cols = train_features_imputed.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_features_imputed[col] = le.fit_transform(train_features_imputed[col].astype(str))
    label_encoders[col] = le

#making sure that no missing values left after imputation
if train_features_imputed.isnull().sum().sum() == 0:
    print("No missing values left after imputation in training features.")
else:
    print("Warning: There are still missing values in training features!")

#Defining feature and target variables as X and y
X = train_features_imputed
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

#Standardizing features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7) #CR7

#Training model | handling multiple target labels
base_model = RandomForestClassifier(random_state=42)
model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_train, y_train)

#Predicting Probability of test set
y_pred = model.predict_proba(X_test)

#finding probabilities for xyz and seasonal vaccine
y_pred_xyz = np.array([pred[:, 1] for pred in y_pred]).T[:, 0]
y_pred_seasonal = np.array([pred[:, 1] for pred in y_pred]).T[:, 1]

#Additionaly calculation Mean ROC SUc score, just in case it is asked 
roc_auc_xyz = roc_auc_score(y_test['xyz_vaccine'], y_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_test['seasonal_vaccine'], y_pred_seasonal)
mean_roc_auc = np.mean([roc_auc_xyz, roc_auc_seasonal])
print(f"ROC AUC for xyz_vaccine: {roc_auc_xyz}")
print(f"ROC AUC for seasonal_vaccine: {roc_auc_seasonal}")
print(f"Mean ROC AUC: {mean_roc_auc}")

#Doing similiar process for test set as done in training set
test_features_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns)
for col in categorical_cols:
    le = label_encoders[col]
    test_features_imputed[col] = le.transform(test_features_imputed[col].astype(str))

test_features_scaled = scaler.transform(test_features_imputed)

y_pred_test = model.predict_proba(test_features_scaled)

y_pred_xyz_test = np.array([pred[:, 1] for pred in y_pred_test]).T[:, 0]
y_pred_seasonal_test = np.array([pred[:, 1] for pred in y_pred_test]).T[:, 1]

#Creating Submission file and saving it
submission = pd.DataFrame({
    'respondent_id': submission_format['respondent_id'],
    'xyz_vaccine': y_pred_xyz_test,
    'seasonal_vaccine': y_pred_seasonal_test
})

submission.to_csv('submission.csv', index=False)
print("Submission file created :>")