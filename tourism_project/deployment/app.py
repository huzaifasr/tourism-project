import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title='Tourism Purchase Predictor', layout='wide')

# Paths
MODEL_PATH = 'tourism_project/tourism_model_v1.joblib'
PREPARED_X_PATH = 'tourism_project/Xtrain.csv'
RAW_CSV_CANDIDATES = ['tourism_project/tourism.csv', 'tourism.csv']

# Load model if available
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.info(f'Loaded model from {MODEL_PATH}')
    except Exception as e:
        st.error(f'Failed to load model at {MODEL_PATH}: {e}')
else:
    st.warning(f'No trained model found at {MODEL_PATH}. Please run training first.')

# Load raw CSV (for populating select options)
raw_df = None
for p in RAW_CSV_CANDIDATES:
    if os.path.exists(p):
        raw_df = pd.read_csv(p)
        break

if raw_df is None:
    st.error('Could not find tourism.csv in repo. Place it at repo root or tourism_project/data.')
    st.stop()

# Quick cleanup matching prep.py behavior
# Drop unnamed index column if present
first_col = raw_df.columns[0]
if first_col == '' or str(first_col).lower().startswith('unnamed'):
    raw_df = raw_df.drop(columns=[first_col])

# Drop CustomerID if present
if 'CustomerID' in raw_df.columns:
    raw_df = raw_df.drop(columns=['CustomerID'])

# Ensure target exists
TARGET = 'ProdTaken'
if TARGET not in raw_df.columns:
    st.error(f"Expected target column '{TARGET}' not found in dataset.")
    st.stop()

# Candidate input fields (based on dataset)
numeric_inputs = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_inputs = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation', 'Passport', 'OwnCar'
]
# Make sure fields exist in raw_df
numeric_inputs = [c for c in numeric_inputs if c in raw_df.columns]
categorical_inputs = [c for c in categorical_inputs if c in raw_df.columns]

st.title('Tourism Package Purchase Predictor')
st.write('Enter customer and interaction details below and click Predict.')

with st.form('input_form'):
    cols = st.columns(3)

    inputs = {}
    # Numeric inputs
    for i, col in enumerate(numeric_inputs):
        c = cols[i % 3]
        mean = float(raw_df[col].dropna().mean()) if not raw_df[col].dropna().empty else 0.0
        inputs[col] = c.number_input(col, value=mean)

    # Categorical / binary inputs
    for col in categorical_inputs:
        if col in ['Passport', 'OwnCar']:
            # binary
            unique_vals = sorted(raw_df[col].dropna().unique().tolist())
            default = 1 if 1 in unique_vals else (0 if 0 in unique_vals else unique_vals[0])
            inputs[col] = st.selectbox(col, options=unique_vals, index=0)
        elif col == 'CityTier':
            # CityTier numeric but categorical-like
            options = sorted(raw_df[col].dropna().unique().tolist())
            inputs[col] = st.selectbox(col, options=options, index=0)
        else:
            options = sorted(raw_df[col].dropna().astype(str).unique().tolist())
            inputs[col] = st.selectbox(col, options=options, index=0)

    submitted = st.form_submit_button('Predict')

# Normalize Gender similar to prep.py
if 'Gender' in inputs:
    g = str(inputs['Gender']).strip().lower()
    if any(x in g for x in ['fe', 'fem']):
        inputs['Gender'] = 'Female'
    elif 'male' in g:
        inputs['Gender'] = 'Male'
    else:
        inputs['Gender'] = inputs['Gender']

# Build single-row DataFrame from inputs using the original raw columns order
input_df = pd.DataFrame([inputs])

# Apply same one-hot encoding used in prep.py (pandas.get_dummies with drop_first and prefix_sep='__')
cat_for_dummies = [c for c in categorical_inputs if c in input_df.columns and input_df[c].dtype == object or isinstance(input_df[c].iloc[0], str)]
if cat_for_dummies:
    input_dummies = pd.get_dummies(input_df, columns=cat_for_dummies, prefix_sep='__', drop_first=True)
else:
    input_dummies = input_df.copy()

# If prepared Xtrain exists, align columns to it (this ensures dummies match training features)
if os.path.exists(PREPARED_X_PATH):
    prepared_cols = pd.read_csv(PREPARED_X_PATH, nrows=0).columns.tolist()
    # Reindex to prepared columns, filling missing with 0
    input_prepared = input_dummies.reindex(columns=prepared_cols, fill_value=0)
else:
    # If no prepared Xtrain file, pass input_dummies as-is and hope pipeline handles raw columns
    input_prepared = input_dummies.copy()

st.subheader('Prepared input (what will be fed to the model)')
st.dataframe(input_prepared.transpose())

if submitted:
    if model is None:
        st.error('No model loaded; cannot predict. Train model and save to the expected path.')
    else:
        try:
            # Ensure columns order matches model training
            # If model is a pipeline expecting a full feature set, the input_prepared must match that
            X_in = input_prepared
            # Some sklearn pipelines expect numpy arrays - convert accordingly
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_in)[:, 1][0]
                pred = int(proba >= 0.5)
            else:
                pred = int(model.predict(X_in)[0])
                proba = None

            st.markdown('## Prediction')
            if proba is not None:
                st.metric('Probability of purchase (ProdTaken=1)', f'{proba:.3f}')
                st.write('Predicted class:', 'Will Purchase (1)' if pred == 1 else 'Will Not Purchase (0)')
            else:
                st.write('Predicted class:', 'Will Purchase (1)' if pred == 1 else 'Will Not Purchase (0)')

        except Exception as e:
            st.error(f'Prediction failed: {e}')

# Optionally show a few sample rows from dataset and model prediction
st.sidebar.markdown('## Dataset samples')
if st.sidebar.button('Show 5 random samples'):
    st.sidebar.dataframe(raw_df.sample(5))

st.sidebar.markdown('Model & data paths')
st.sidebar.text(MODEL_PATH)
st.sidebar.text(PREPARED_X_PATH)
