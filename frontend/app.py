import streamlit as st
import requests
import pandas as pd
import io
import json

st.set_page_config(page_title="AutoML Project", layout="wide")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Training"]
)

st.title('End-to-End AutoML Project: Insurance Cross-Sell')

if menu == "Prediction":
    endpoint = 'http://192.168.2.83:8000/predict'
    test_csv = st.file_uploader('', type=['csv'], accept_multiple_files=False)

    if test_csv:
        test_df = pd.read_csv(test_csv)
        st.subheader('Sample of Uploaded Dataset')
        st.write(test_df.head())

        test_bytes_obj = io.BytesIO()
        test_df.to_csv(test_bytes_obj, index=False)
        test_bytes_obj.seek(0)

        files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

        if st.button('Start Prediction'):
            if len(test_df) == 0:
                st.write("Please upload a valid test dataset!")
            else:
                with st.spinner('Prediction in Progress. Please Wait...'):
                    output = requests.post(endpoint, files=files, timeout=8000)
                st.success('Success! Click Download button below to get prediction results (in JSON format)')
                st.download_button(
                    label='Download',
                    data=json.dumps(output.json()),
                    file_name='automl_prediction_results.json'
                )

elif menu == "Training":
    st.header("Model Training")
    st.write("Upload your training dataset and start model training here.")

    experiment_name = st.text_input("Experiment Name", "")
    target = st.text_input("Target Column", "")
    models = st.number_input("Number of Models (max_models)", min_value=1, max_value=50, value=10)

    train_csv = st.file_uploader('Upload Training CSV', type=['csv'], key='train')
    if train_csv:
        train_df = pd.read_csv(train_csv)
        st.subheader('Sample of Training Dataset')
        st.write(train_df.head())
        if st.button('Start Training'):
            if not experiment_name:
                st.warning("Please enter an experiment name.")
            elif not target:
                st.warning("Please enter the target column.")
            elif train_df.empty:
                st.warning("Please upload a valid training dataset.")
            else:
                with st.spinner(f"Training started for experiment '{experiment_name}' with {models} models."):
                    train_bytes_obj = io.BytesIO()
                    train_df.to_csv(train_bytes_obj, index=False)
                    train_bytes_obj.seek(0)
                    files = {"file": ('train_dataset.csv', train_bytes_obj, "multipart/form-data")}
                    data = {
                        "experiment_name": experiment_name,
                        "target": target,
                        "models": int(models)
                    }
                    try:
                        response = requests.post(
                            'http://192.168.2.83:8000/train',
                            files=files,
                            data=data,
                            timeout=8000
                        )
                        if response.status_code == 200:
                            st.success("Training completed successfully!")
                            st.json(response.json())
                        else:
                            st.error(f"Training failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error during training: {e}")