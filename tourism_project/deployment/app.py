import streamlit as st
import pandas as pd
import os
import joblib
from huggingface_hub import hf_hub_download


st.set_page_config(page_title='Tourism — Predictor UI', layout='centered')


# Known candidate paths for model and data (search a few likely locations)
MODEL_CANDIDATES = [
    'tourism_model_v1.joblib',
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# Prefer loading model from Hugging Face Hub; fall back to local candidates
model = None
try:
    model_path = hf_hub_download(repo_id="huzaifa-sr/tourism-project", filename="tourism_model_v1.joblib")
    model = joblib.load(model_path)
    st.info(f'Loaded model from Hugging Face: {model_path}')
except Exception as e:
    st.warning(f'HF download failed or unavailable: {e}. Falling back to local model candidates.')

    def load_model(path):
        try:
            return joblib.load(path)
        except Exception as e2:
            st.warning(f'Failed to load model at {path}: {e2}')
            return None

    MODEL_PATH = find_first_existing(MODEL_CANDIDATES)
    model = load_model(MODEL_PATH) if MODEL_PATH else None


st.title('Tourism Package Purchase — Simple Predictor')

st.write('This lightweight interface lets you try the trained model (if available) using manual inputs or an uploaded CSV.')

uploaded = st.file_uploader('Upload a tourism CSV (optional)', type=['csv'])

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success('Uploaded data loaded — you may use values from it externally to populate inputs if desired')
    except Exception as e:
        st.error(f'Failed to read uploaded CSV: {e}')


st.sidebar.header('Options')
st.sidebar.markdown('Provide manual inputs below; uploading a CSV is optional and will not be auto-loaded by the app.')

st.subheader('Manual inputs (simple subset)')
# Fixed defaults only — no dependence on local CSV
Age = st.number_input('Age', value=30)
TypeofContact = st.selectbox('TypeofContact', options=['Self Enquiry', 'Company Invited', 'Agent'], index=0)
CityTier = st.selectbox('CityTier', options=[1, 2, 3], index=0)
DurationOfPitch = st.number_input('DurationOfPitch', value=10.0)
Occupation = st.selectbox('Occupation', options=['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government', 'Other'], index=0)
Gender = st.selectbox('Gender', options=['Male', 'Female'], index=0)
NumberOfPersonVisiting = st.number_input('NumberOfPersonVisiting', value=2)
NumberOfFollowups = st.number_input('NumberOfFollowups', value=2)
ProductPitched = st.selectbox('ProductPitched', options=['Basic', 'Standard', 'Deluxe', 'King', 'Super Deluxe'], index=0)
PreferredPropertyStar = st.number_input('PreferredPropertyStar', min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced', 'Unmarried'], index=0)
NumberOfTrips = st.number_input('NumberOfTrips', value=1)
Passport = st.radio('Passport (has passport?)', options=['Yes', 'No'], index=1)
PitchSatisfactionScore = st.number_input('PitchSatisfactionScore', value=3)
OwnCar = st.radio('OwnCar (has car?)', options=['Yes', 'No'], index=1)
NumberOfChildrenVisiting = st.number_input('NumberOfChildrenVisiting', value=0)
Designation = st.text_input('Designation', value='Executive')
MonthlyIncome = st.number_input('MonthlyIncome', value=30000.0)

inputs = {
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == 'Yes' else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == 'Yes' else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
}
input_df = pd.DataFrame([inputs])


st.markdown('---')

st.subheader('Prepared input preview')
if input_df is not None:
    st.dataframe(input_df.transpose())
else:
    st.write('No input prepared yet')


if st.button('Predict'):
    if model is None:
        st.error('No trained model found. Place a joblib model at one of the known locations or train one first.')
        if MODEL_PATH:
            st.write('Tried model path:', MODEL_PATH)
    else:
        try:
            # Very small alignment: if model expects a numpy array, pass values; else pass DataFrame
            X = input_df
            # Some models require feature ordering / dummies; we keep this simple and let pipeline handle it if present
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1][0]
                pred = int(proba >= 0.5)
                st.success(f'Predicted probability of purchase: {proba:.3f} — Class: {pred}')
            else:
                pred = int(model.predict(X)[0])
                st.success(f'Predicted class: {pred}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')


st.sidebar.markdown('---')
st.sidebar.write('Model path:')
st.sidebar.text(MODEL_PATH or 'Not found')
