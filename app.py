import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import joblib 
import pandas as pd
import os

# --- CONFIGURATION & MODEL LOADING ---
# 1. Gemini Configuration (REPLACE KEY)
# 🚨 WARNING: YOU MUST REPLACE THIS PLACEHOLDER WITH YOUR ACTUAL KEY.
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBn1RQK0uMb_Pb9i_t8qSeo4y8LDWWLoGw") # <--- REPLACE THIS LINE

try:
    genai.configure(api_key=API_KEY) 
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"❌ GEMINI CONFIG ERROR: Check your key and connection. Error: {e}")
    
# 2. ML Model Configuration
try:
    # Load the trained model pipeline saved by train.py
    ml_model = joblib.load('ade_risk_model.joblib')
    print("✅ ML Model loaded successfully.")
except FileNotFoundError:
    print("❌ FATAL ERROR: 'ade_risk_model.joblib' not found. Run train.py first.")
    ml_model = None

# --- CONSTANTS (Must match train.py) ---
MODEL_FEATURES = [
    'Age', 'TotalMedications', 'AlphaDiversity', 'MicrobialMetabolite_A_Level',
    'FB_Ratio', 'RiskDrugCount', 'AntiFibroticUse'
]
ADE_RISK_DRUGS = ["Pantoprazole", "Omeprazole", "Azithromycin", "Ciprofloxacin", "Prednisone", "Warfarin"]
ANTI_FIBROTIC_DRUGS = ["Pirfenidone", "Nintedanib"]


app = Flask(__name__)


# --- FEATURE EXTRACTION FUNCTION (Natural Language Parsing) ---
def extract_features_from_text(user_input: str) -> dict or None:
    """
    Uses Gemini to extract structured patient data from an unstructured text query.
    Returns a dictionary of patient data or None if extraction fails.
    """
    extraction_prompt = f"""
    **TASK:** Analyze the following patient data text. Extract the required features and output ONLY a single JSON object. If a medication is mentioned, list ALL mentioned drugs under the MedicationList_Text key.
    
    **REQUIRED OUTPUT JSON KEYS (Use 0 or 0.0 as default if a value is not found in the text):**
    {{"Age": <int>, "TotalMedications": <int>, "AlphaDiversity": <float>, 
    "MicrobialMetabolite_A_Level": <float>, "Bacteroides_Abundance": <float>, 
    "Firmicutes_Abundance": <float>, "MedicationList_Text": "<string of all meds>"}}
    
    **INPUT TEXT:** "{user_input}"
    
    **INSTRUCTIONS:** Provide ONLY the JSON object. Do not include any explanations or markdown formatting like ```json.
    """
    try:
        response = gemini_model.generate_content(extraction_prompt)
        json_text = response.text.strip()
        # Clean up common LLM markdown wrappers if present
        json_text = json_text.replace("```json", "").replace("```", "")
        extracted_data = json.loads(json_text)
        return extracted_data
    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

# --- FEATURE ENGINEERING FUNCTION (Used before ML prediction) ---
def engineer_features(patient_data: dict) -> pd.DataFrame:
    """Processes extracted data into the features required by the ML model."""
    
    age = patient_data.get('Age', 65)
    total_meds = patient_data.get('TotalMedications', 4)
    alpha_diversity = patient_data.get('AlphaDiversity', 4.0)
    metabolite_a = patient_data.get('MicrobialMetabolite_A_Level', 10.0)
    bacteroides = patient_data.get('Bacteroides_Abundance', 0.25)
    firmicutes = patient_data.get('Firmicutes_Abundance', 0.50)
    medication_list_text = patient_data.get("MedicationList_Text", "")
    
    # 1. Feature Calculation: FB_Ratio 
    fb_ratio = firmicutes / (bacteroides if bacteroides > 0 else 0.001)

    # 2. Feature Parsing: RiskDrugCount and AntiFibroticUse (using simple text search)
    risk_drug_count = sum(1 for drug in ADE_RISK_DRUGS if drug.lower() in medication_list_text.lower())
    anti_fibrotic_use = int(any(drug.lower() in medication_list_text.lower() for drug in ANTI_FIBROTIC_DRUGS))
    
    # 3. Create the final DataFrame
    data_row = {
        'Age': age, 
        'TotalMedications': total_meds, 
        'AlphaDiversity': alpha_diversity, 
        'MicrobialMetabolite_A_Level': metabolite_a,
        'FB_Ratio': fb_ratio,
        'RiskDrugCount': risk_drug_count,
        'AntiFibroticUse': anti_fibrotic_use
    }
    
    return pd.DataFrame([data_row])[MODEL_FEATURES]


# --- FRONTEND ROUTE ---
@app.route('/')
def index():
    # NOTE: Ensure index.html is in a folder named 'templates' 
    return render_template('index.html')


# --- CONSOLIDATED CHAT/ANALYSIS ENDPOINT ---
@app.route('/chat', methods=['POST'])
def handle_chat_query():
    raw_data = request.get_json()
    user_input = raw_data.get('query', '').strip()
    
    if ml_model is None:
        return jsonify({"error": "ML Model is not loaded. Run train.py first."}), 503
    if not user_input:
        return jsonify({"response": "Please enter a query."}), 400
        
    
    # 1. ATTEMPT STRUCTURED ANALYSIS (XAI MODE)
    patient_data = extract_features_from_text(user_input)

    # Check if a meaningful amount of data was extracted
    if patient_data and patient_data.get('Age', 0) > 0 and patient_data.get('TotalMedications', 0) > 0:
        
        try:
            # 1.1 PHASE 1: Feature Engineering and ML Prediction
            features_df = engineer_features(patient_data)
            ade_prob = ml_model.predict_proba(features_df)[0][1]
            
            # 1.2 PHASE 2: Dynamic Risk Summary Generation
            ade_percentage = f"{ade_prob * 100:.2f}%"
            
            risk_message = ""
            if features_df['RiskDrugCount'].iloc[0] > 0:
                # Use bold markdown here, but we will remove it in the final cleanup
                risk_message += f"**High Polypharmacy Risk:** {features_df['RiskDrugCount'].iloc[0]} known high-risk drugs detected. "
            if features_df['FB_Ratio'].iloc[0] > 2.5: 
                 risk_message += f"**Dysbiotic Gut Profile:** High Firmicutes/Bacteroides ratio ({features_df['FB_Ratio'].iloc[0]:.2f}) is associated with altered drug metabolism. "

            
            # 1.3 PHASE 3: Dynamic Prompting for Explainable AI (XAI)
            prompt = f"""
            **ROLE:** You are an Explainable AI (XAI) Assistant for the FibroGuide system. You MUST provide a concise, professional, and personalized report (max 200 words) to an IPF specialist doctor.
            
            **PATIENT PROFILE SUMMARY:** Age {patient_data.get('Age')}, Total Meds {patient_data.get('TotalMedications')}, F/B Ratio {features_df['FB_Ratio'].iloc[0]:.2f}.
            
            **MODEL PREDICTIONS (FROM TRAINED MODEL):**
            - Adverse Drug Event (ADE) Risk Probability: {ade_prob:.4f} (or {ade_percentage})
            
            **INSTRUCTIONS for Response Structure (Strictly adhere to this format):**
            
            1.  **SUMMARY & RATIONALE:** Start with "Recommendation Summary:". State the predicted ADE Risk Probability in both decimal and percentage formats.
            2.  **RATIONALE:** Use bullet points to clearly explain the primary risk factors driving the score. The rationale should be based on the following:
                * Advanced Age ({patient_data.get('Age')} years) and Polypharmacy.
                * Specific contributing factors: {risk_message}
            3.  **RECOMMENDATIONS:** Provide two clear, actionable medical recommendations derived from the rationale.
            4.  **Note:** Ensure the output uses only professional language.
            """
            
            response = gemini_model.generate_content(prompt)
            response_text = response.text
            
            # --- FINAL CLEANUP: Remove ALL unwanted markdown symbols ---
            # 1. Remove the entire unnecessary section generated in the last chat
            unwanted_section = """
* **GAP score (Gender, Age, Physiology)**: This score assesses lung function, age, and gender to help predict mortality in IPF.
* **DLCO (Diffusing Capacity of the Lung for Carbon Monoxide)**: A measure of how well the lungs transfer oxygen into the blood.
* **FVC (Forced Vital Capacity)**: A measure of the largest amount of air you can forcibly exhale after taking a deep breath."""
            response_text = response_text.replace(unwanted_section, "").strip()
            
            # 2. Add the main report header (as HTML class for styling)
            response_text = "🔬 <span class='report-header-text'>FibroGuide XAI Analysis Report</span>\n\n" + response_text
            
            # 3. Use regex to remove all remaining markdown bolding (** symbols)
            response_text = response_text.replace('**', '') 
            
            # 4. Use HTML span class for the summary header (for color styling in CSS)
            response_text = response_text.replace("Recommendation Summary:", "<span class='analysis-header'>Recommendation Summary:</span>")


            return jsonify({"success": True, "response": response_text})

        except Exception as e:
            print(f"ML/Prompting Error: {e}")
            # Fall through to general conversation if the analysis pipeline fails
            pass

    # 2. FALLBACK TO GENERAL CONVERSATIONAL QUERY (CHATBOT MODE)
    
    # If extraction failed or the input didn't look like patient data, run a general chat
    system_prompt = "You are a friendly, factual, and supportive medical assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question concisely and professionally."
    try:
        response = gemini_model.generate_content([system_prompt, user_input])
        response_text = response.text.replace('**', '') # Remove bolding from simple chat too
        return jsonify({"success": True, "response": response_text})
    except Exception as e:
        app.logger.error(f"Gemini Chat Error: {e}")
        return jsonify({"error": "LLM service unavailable or invalid API key."}), 500
        
    except Exception as e:
        app.logger.error(f"General Error: {e}")
        return jsonify({"error": f"An unexpected error occurred during analysis: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')