import os
import io
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, Response

# Explicitly importing these ensures Flask can unpack them safely from joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

app = Flask(__name__)

# 1. Configuration
FEATURES = ['C', 'W', 'S', 'G', 'SF', 'FA', 'LS', 'SP', 'TM']

# Placeholder historical errors (update these with your actual validation metrics if desired)
PV_ERROR = 1.5  
DYS_ERROR = 2.3

# 2. Load the complete Stacking Packages
MODEL_PV_PATH = os.path.join('models', 'stacking_ensemble_PV.pkl')
MODEL_DYS_PATH = os.path.join('models', 'stacking_ensemble_DYS.pkl')

try:
    pkg_pv = joblib.load(MODEL_PV_PATH)
    pkg_dys = joblib.load(MODEL_DYS_PATH)
    print("Ensemble packages loaded successfully!")
except Exception as e:
    print(f"Initialization Error: Could not load model packages. Details: {e}")


# 3. Helper function replicating your exact pipeline architecture
def predict_stacking(package, df_input):
    """
    Extracts components from the saved package dictionary, scales input features,
    generates meta-features, and runs the final meta-learner prediction.
    """
    scaler_X = package["scaler_X"]
    scaler_y = package["scaler_y"]
    base_models = package["base_models"]
    meta_learner = package["meta_learner"]
    
    # Step A: Standardize input features using training configuration
    X_scaled = scaler_X.transform(df_input.values)
    
    # Step B: Generate meta-features from the 4 underlying base models
    meta_features = []
    for model in base_models:
        pred_scaled = model.predict(X_scaled)
        # Reshape and inverse transform to get original scale metrics
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        meta_features.append(pred)
        
    X_meta = np.column_stack(meta_features)
    
    # Step C: Final execution through the trained Ridge Meta-Learner
    final_predictions = meta_learner.predict(X_meta)
    return final_predictions


# 4. Web Application Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect single-sample web inputs
        input_data = {}
        for feature in FEATURES:
            input_data[feature] = [float(request.form[feature])]
        
        # Structure incoming values into a formatted DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Execute customized stack processing predictions
        pred_pv = predict_stacking(pkg_pv, input_df)[0]
        pred_dys = predict_stacking(pkg_dys, input_df)[0]
        
        results = {
            'pv': round(float(pred_pv), 3), 'pv_err': PV_ERROR,
            'dys': round(float(pred_dys), 3), 'dys_err': DYS_ERROR,
            'inputs': {f: request.form[f] for f in FEATURES}
        }
        return render_template('index.html', single_results=results)
        
    except Exception as e:
        return render_template('index.html', error=f"Prediction System Error: {str(e)}")


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return render_template('index.html', error="File slot empty")
        
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No spreadsheet selected")
        
    try:
        df = pd.read_csv(file)
        
        # Validation validation check
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            return render_template('index.html', error=f"CSV missing mandatory headers: {missing_features}")
            
        # Target specific column subset
        X_df = df[FEATURES]
        
        # Process arrays simultaneously
        df['Predicted_PV'] = predict_stacking(pkg_pv, X_df).round(3)
        df['PV_Error_Margin'] = PV_ERROR
        df['Predicted_DYS'] = predict_stacking(pkg_dys, X_df).round(3)
        df['DYS_Error_Margin'] = DYS_ERROR
        
        # Stream modified file configuration data out back to user
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=rheology_predictions.csv"}
        )
        
    except Exception as e:
        return render_template('index.html', error=f"Spreadsheet Error: {str(e)}")


if __name__ == '__main__':
    # Hugging Face needs host='0.0.0.0' and port=7860 to route public web traffic
    app.run(host='0.0.0.0', port=7860, debug=False)