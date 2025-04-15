import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("final_linear_svc_model.joblib")
preprocessor = joblib.load("preprocessing_pipeline.joblib")

st.title("SVM Classifier - Bulk Upload")

uploaded_file = st.file_uploader("Upload a CSV file with original features", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", input_df.head())

        # Preprocess and predict
        processed = preprocessor.transform(input_df)
        predictions = model.predict(processed)

        # Map numerical predictions to labels
        prediction_labels = ['Unlikely Survived' if pred == 0 else 'Likely Survived' for pred in predictions]

        # Add to dataframe
        input_df['Prediction'] = prediction_labels
        st.success("✅ Predictions completed.")
        st.write(input_df)

        # Downloadable output
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
