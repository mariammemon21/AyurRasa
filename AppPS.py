import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def run_app():
    # Load the trained machine learning models
    model_primary = pickle.load(open(r'D:\IOT\VADODARA\AllRasaPS\AllRasaPS\RasaP.pkl', 'rb'))
    model_secondary = pickle.load(open(r'D:\IOT\VADODARA\AllRasaPS\AllRasaPS\RasaS.pkl', 'rb'))
    df = pd.read_csv(r"D:\IOT\VADODARA\AllRasaPS\AllRasaPS\dataPS.csv")

    st.title('Ayurvedic Herbs Rasa Prediction')

    # Section for input features
    input_col1, input_col2 = st.columns(2)
    input_features = [
        "Glucose", "Sucrose", "Fructose", "Tannins",
        "Phenolic Acids", "Citric", "Malic",
        "Tartaric Acid", "Alkaloids", "Terpenes"
    ]

    user_input = {}
    for idx, feature in enumerate(input_features):
        # Splitting features into two columns
        if idx < 5:
            user_input[feature] = input_col1.number_input(f"Enter value for {feature}:", min_value=0.0, max_value=100.0)
        else:
            user_input[feature] = input_col2.number_input(f"Enter value for {feature}:", min_value=0.0, max_value=100.0)

    input_df = pd.DataFrame([user_input])

    if st.button('Predict'):
        prediction_primary = model_primary.predict(input_df)[0]
        prediction_secondary = model_secondary.predict(input_df)[0]

        labels_rasa = {
            1: 'Madhura(Sweet)', 2: 'Katu(Pungent)',
            3: 'Kashaya(Astringent)', 4: 'Amla(Sour)',
            5: 'Tikta(Bitter)'
        }

        predicted_label_primary = labels_rasa.get(prediction_primary, 'Unknown')
        predicted_label_secondary = labels_rasa.get(prediction_secondary, 'Unknown')

        st.header('Prediction Results')
        st.write(f"Predicted Primary Rasa: {predicted_label_primary}")
        st.write(f"Predicted Secondary Rasa: {predicted_label_secondary}")

        st.header('Accuracy Graph')

        # Calculate or provide actual accuracy scores
        accuracy_primary_rasa = 0.85  # Replace with actual accuracy
        accuracy_secondary_rasa = 0.92  # Replace with actual accuracy

        labels = ['Primary Rasa', 'Secondary Rasa']
        accuracies = [accuracy_primary_rasa * 100, accuracy_secondary_rasa * 100]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, accuracies, color=['skyblue', 'lightgreen'])
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of Rasa Prediction')

        st.pyplot(fig)
