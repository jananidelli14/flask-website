import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib 

# --- CONFIGURATION & CONSTANTS ---

# NOTE: Update this path to where your CSV file is located if it's not in the same directory.
data_file = r"C:\Users\janan\Downloads\augmented_ipf_patient_data_200.csv"
model_output_file = 'ade_risk_model.joblib'

# FEATURES: Must EXACTLY match the list in app.py
MODEL_FEATURES = [
    'Age', 
    'TotalMedications', 
    'AlphaDiversity', 
    'MicrobialMetabolite_A_Level',
    'FB_Ratio',
    'RiskDrugCount',
    'AntiFibroticUse'
]

# High-risk drug list: Must EXACTLY match the list in app.py
ADE_RISK_DRUGS = ["Pantoprazole", "Omeprazole", "Azithromycin", "Ciprofloxacin", "Prednisone", "Warfarin"]


# --- FEATURE ENGINEERING FUNCTIONS ---

def count_risk_drugs(med_json_str):
    """Counts how many high-risk drugs are in the patient's medication list."""
    try:
        # Load the JSON string from the CSV column
        med_list = json.loads(med_json_str) 
        return sum(1 for drug in med_list if drug.get('drugName') in ADE_RISK_DRUGS)
    except:
        return 0

def has_antifibrotic(med_json_str):
    """Checks if the patient is taking one of the core IPF anti-fibrotic drugs (Pirfenidone or Nintedanib)."""
    try:
        med_list = json.loads(med_json_str)
        return any(drug.get('drugName') in ["Pirfenidone", "Nintedanib"] for drug in med_list)
    except:
        return 0

def run_training():
    """Loads augmented data, engineers features, trains the model, and saves the pipeline."""
    
    # Check for the augmented data file
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} augmented records for training from '{data_file}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Training data file '{data_file}' not found. Please ensure it is in the correct location.")
        return

    # 1. FEATURE ENGINEERING: Calculate all derived features 
    # This must be identical to the logic in the engineer_features function in app.py.
    
    # 1.1 Calculate RiskDrugCount
    df['RiskDrugCount'] = df['MedicationList_JSON'].apply(count_risk_drugs)
    
    # 1.2 Calculate AntiFibroticUse
    df['AntiFibroticUse'] = df['MedicationList_JSON'].apply(has_antifibrotic).astype(int)
    
    # 1.3 Calculate the Firmicutes/Bacteroides ratio (FB_Ratio)
    # Replaces 0 in Bacteroides_Abundance with a tiny floor value (0.001) to prevent division by zero
    df['FB_Ratio'] = df['Firmicutes_Abundance'] / df['Bacteroides_Abundance'].replace(0, 0.001)
    
    # 2. MODEL PREPARATION
    
    # Select features (X) and target (Y)
    X = df[MODEL_FEATURES]
    # Target (Y): Adverse Drug Event Prediction (Binary: 0 or 1)
    Y = df['AdverseDrugEvent_Occurred']
    
    # Split data to measure performance
    # Setting random_state and stratify ensures reproducibility and balanced splits
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # 3. TRAINING AND PIPELINE CREATION
    # Use a Pipeline to combine scaling and the model, which is essential for consistent preprocessing 
    # when the model is loaded in app.py.
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    model_pipeline.fit(X_train, Y_train)
    
    # 4. REPORT AND SAVE MODEL
    train_accuracy = model_pipeline.score(X_train, Y_train)
    test_accuracy = model_pipeline.score(X_test, Y_test)
    print(f"\n--- Model Training Complete ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    joblib.dump(model_pipeline, model_output_file)
    print(f"✅ Trained model saved successfully as '{model_output_file}'")

if __name__ == '__main__':
    run_training()