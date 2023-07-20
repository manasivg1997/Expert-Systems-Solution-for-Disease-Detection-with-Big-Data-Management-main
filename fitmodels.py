# Import all necessary packages
import os
import math
import warnings
import pandas as pd
from itertools import combinations
warnings.filterwarnings("ignore")

# Get file path
current_directory = os.getcwd()
data_path = current_directory + "/Datasets-CSV"

# Load datasets for all possible combinations & for individual disease's respective symptoms
df_combination = pd.read_csv(data_path + "/Disease_Symptom_Dataset_For_All_Symptom_Subsets.csv") 
df_independent = pd.read_csv(data_path + "/Disease_Symptom_Dataset_For_Respective_Symptoms.csv") 

X_combination = df_combination.iloc[:, 1:]
Y_combination = df_combination.iloc[:, 0:1]

X_independent = df_independent.iloc[:, 1:]
Y_independent = df_independent.iloc[:, 0:1]

# List of all possible symptoms
all_symptoms = list(X_independent.columns)
all_diseases = list(set(Y_independent['Disease_Name']))
all_diseases.sort()

# We obtain top 10 possible diseases
no_of_diseases = 10

# Function to generate top 10 diseases from the ML results
# Pass mean_score as another argument if you need probabilities too
def ProcessResultAndGenerateDiseases(top10_list, mean_score, cooccuring_symptoms, user_symptoms_len):
    
    global df_independent, all_symptoms, all_diseases
    top10_dict = {}

    # Checks for each disease, the matched symptoms & generates probability of having that disease
    for (idx, disease_id) in enumerate(top10_list):
        matched_symptoms = set()
        top10 = df_independent.loc[df_independent['Disease_Name'] == all_diseases[disease_id]].values.tolist()
        
        # Obtains the disease name which is at the top of the dataframe
        disease = top10[0].pop(0)

        # Each row contains 0s & 1s indicating whether a disease is associated with a particular symptom or not
        for (idx, value) in enumerate(top10[0]):
            if value != 0:
                matched_symptoms.add(all_symptoms[idx])
                
        #print("\n", matched_symptoms)
        probability = (len(matched_symptoms.intersection(set(cooccuring_symptoms))) + 1) / (user_symptoms_len + 1)
        top10_dict[disease] = round(probability * mean_score * 100, 2)
    
    top10_sorted_dict = dict(sorted(top10_dict.items(), key=lambda kv: kv[1], reverse=True))
    return top10_sorted_dict  


# Function to display the results from the dictionary
def PrintDictionary(top10_sorted_dict):
    for (key, value) in top10_sorted_dict.items():
        print(key, "\t", value, "%")

        
# Function to generate subsets
def GetPossibleSubsets(user_symptoms):
    
    global all_symptoms
    processed_symptoms = []
    user_symptoms_len = len(user_symptoms)
    minSubsetLength = math.floor(user_symptoms_len * 0.8)
    
    # Form possible subsets with minSubsetLength
    for combination in range(minSubsetLength, user_symptoms_len + 1):
        for subset in combinations(user_symptoms, combination):
            temp_processed_symptoms = [0 for x in range(0, len(all_symptoms))]
            for symptom in subset:
                temp_processed_symptoms[all_symptoms.index(symptom)] = 1
            processed_symptoms.append(temp_processed_symptoms)
    
    return processed_symptoms
   
# Function to get predictions for possible subsets from given symptoms
def GetTop10BySubsets(model, mean_score, user_symptoms, processed_symptoms):
    
    model_dict_res, res_dict = {}, {}
    user_symptoms_len = len(user_symptoms)
    subsets = 0
    
    for proc_sym in processed_symptoms:
        subsets += 1
        model_result = model.predict_proba([proc_sym])
        model_top10 = model_result[0].argsort()[-10:][::-1]
        model_dict = ProcessResultAndGenerateDiseases(model_top10, mean_score, user_symptoms, user_symptoms_len)

        for (key, value) in model_dict.items():
            if key not in model_dict_res.keys():
                model_dict_res[key] = [value, 1]
            else:
                model_dict_res[key] = [model_dict_res[key][0] + value, model_dict_res[key][1] + 1]
        #print(model_dict_res, "\n")
    
    print("Total no. of subsets considered: ", subsets)
    for (key, value) in model_dict_res.items():
        res_dict[key] = round(value[0] / value[1], 2)
        
    res_dict = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    return res_dict

# Function to find co-occuring symptoms with all the symptoms user chosen
# We use a threshold to check for a 90% match with the given symptoms
def FindCooccuringSymptomsWithThreshold(user_symptoms):
    
    global df_independent, all_symptoms
    threshold = math.floor(len(user_symptoms) * 0.90)

    # Get all unique possible diseases with the given symptoms
    unique_diseases = set()
    for symptom in user_symptoms:
        possible_diseases_for_symptom = list(df_independent[df_independent[symptom] == 1]['Disease_Name'])
        for disease in possible_diseases_for_symptom:
            unique_diseases.add(disease)
        
    # Get all unique diseases & sort them
    unique_diseases = sorted(list(unique_diseases))
    
    #print(unique_diseases)

    # Obtain co-occuring symptoms with 90% threshold
    # cooccuring_symptoms must have all given symptoms by default
    cooccuring_symptoms = set(user_symptoms)   
    for disease in unique_diseases:
        
        # First, obtain all symptoms associated with each disease in unique diseases obtained
        symptoms_of_disease = df_independent.loc[df_independent['Disease_Name'] == disease].values.tolist().pop(0)

        # Maintain a temporary set of symptoms of the disease & add them only when they meet threshold requirements
        temp_symptoms = set()
        count, add_symptoms = 0, False
        for idx in range(len(symptoms_of_disease)):
            
            # Symptoms of a disease will have 1 in their respective symptom columns
            if symptoms_of_disease[idx] == 1:
                temp_symptoms.add(all_symptoms[idx-1])
                count = count + 1

                # Our threshold is set to 90% of original symptoms
                if count > threshold:
                    add_symptoms = True

        # Adds temporary symptoms to cooccuring symptoms only if they meet threshold requirements
        if add_symptoms == True:
            for symp in temp_symptoms:
                cooccuring_symptoms.add(symp)

    cooccuring_symptoms = sorted(list(cooccuring_symptoms))
    return cooccuring_symptoms