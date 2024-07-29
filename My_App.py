import streamlit as st
import pandas as pd
import numpy as np

# Import the run_app function from AppPS.py
from AppPS import run_app

# Import the run_herb_analysis function from herb_analysis.py
from herb_analysis import run_herb_analysis

st.header('Welcome to Our AyurRasa Web App')
st.subheader('''
             The page is divided into three categories:
             1. Report Visualization
             2. Prediction of Ayurvedic Herbs
             3. Herb Analysis
             ''')

options = st.selectbox('Please select', ['Drill Down','Report Visualization', 'Ayurvedic Herbs Prediction', 'Herb Analysis'])

if options == 'Report Visualization':
    st.markdown('''
    <iframe title="Rasas and its constituents" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiY2MwNzRkZmItYzZmZi00OGU1LWI1ZjctZWU1ZmZjYTI2Nzk5IiwidCI6IjI1ZDNhZDg5LWJlYjUtNDJmOS05ODVkLTExNTUxYzVlZDZhNyIsImMiOjN9" frameborder="0" allowFullScreen="true"></iframe>
    ''', unsafe_allow_html=True)

elif options == 'Ayurvedic Herbs Prediction':
    run_app()

elif options == 'Herb Analysis':
    run_herb_analysis()
