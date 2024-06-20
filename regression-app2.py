import base64
import pickle
from streamlit_option_menu import option_menu
import numpy as np
import os
import pandas as pd
import streamlit as st
# import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import subprocess

# The App

logo = Image.open("bch project logo placeholder.png")
st.image(logo)

st.title("Alzheimer's Drug Prediction")
st.divider()




# loading the saved models
bioactivity_first_model = pickle.load(open('rfmodel.pkl', 'rb'))

# Define the tabs
tab1,tab2 = st.tabs(['Main', 'About'])

with tab1:
    st.title('Application Description')
    st.success(
        " This module of [**MAO-B-Pred**](https://github.com/RatulChemoinformatics/MAO-B has been built to predict bioactivity and identify potent inhibitors against MAO-B using robust machine learning algorithms."
    )

# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            'AChE Inhibitor Prediction model using pubchemfingerprints'
        ],
    )

# MAO-B prediction model using pubchemfingerprints
if selected == 'AChE Inhibitor Prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against MAO-B using pubchemfingerprints')

    # Molecular descriptor calculator
      
    def desc_calc():
        # Performs the descriptor calculation
        # bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        subprocess.run("padel.sh", shell = True)
        os.remove('molecule.smi')

    
      # File download
    def filedownload(df):
       csv = df.to_csv(index=False)  # Convert DataFrame to CSV format
       b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV string in base64
       href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'  # Create download link
       return href

# Example usage:
# Assuming df is your DataFrame
# download_link = filedownload(df)
# print(download_link)
    

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('SMILES Input'):
        userinput = st.sidebar.text_input(" ", 'ccccc')
        with open('user_input.txt', 'w') as f:
            f.write(f"{userinput} SMILES1")

    if st.sidebar.button('Predict'):
        if userinput is not None:
            load_data = pd.read_table('user_input.txt', sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t',
                             header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('features.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.') 
            
