

import pandas as pd
from statistics import mean
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib





#Load datasets for all possible combinations & for individual disease's respective symptoms
df_combination = pd.read_csv("./Disease_Symptom_Dataset_For_All_Symptom_Subsets.csv") 
df_independent = pd.read_csv("./Disease_Symptom_Dataset_For_Respective_Symptoms.csv") 

X_combination = df_combination.iloc[:, 1:]
Y_combination = df_combination.iloc[:, 0:1]

X_independent = df_independent.iloc[:, 1:]
Y_independent = df_independent.iloc[:, 0:1]

    # List of all possible symptoms
all_symptoms = list(X_independent.columns)
all_diseases = list(set(Y_independent['Disease_Name']))
all_diseases.sort()
    # We obtain top 10 possible diseases

# Create Logistic Regression Classifier & fit the data to it
print("Processing with Logistic Regression...")

lr_classifier = LogisticRegression()
lr_classifier = lr_classifier.fit(X_combination, Y_combination)
filename = 'log_reg.sav'
joblib.dump(lr_classifier, filename)

# Create Random Forest Classifier & fit the data to it
print("Processing with Random Forest Classifier...")

rf_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
rf_classifier = rf_classifier.fit(X_combination, Y_combination)
filename = 'random_forest.sav'
joblib.dump(rf_classifier, filename)

# Create KNN Classifier & fit the data to it
print("Processing with KNN Classifier...")

knn_classifier = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=4)
knn_classifier = knn_classifier.fit(X_combination, Y_combination)
filename = 'knn.sav'
joblib.dump(knn_classifier, filename)

print("Processing with Multinomial Naive Bayes...")

mnb_classifier = MultinomialNB()
mnb_classifier = mnb_classifier.fit(X_combination, Y_combination)
filename = 'mnb.sav'
joblib.dump(mnb_classifier,filename)
