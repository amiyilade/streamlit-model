import base64
import pickle
import os
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import subprocess
from dataclasses import dataclass
from time import sleep

# The App

logo = Image.open("bch project logo placeholder.png")
st.image(logo)

st.title("Alzheimer's Drug Prediction")
st.divider()

st.markdown('''By Onomeyimi Onesi
          
Dataset obtained from [ChEMBL database](https://www.ebi.ac.uk/chembl/g/#search_results/targets/query=ACHE) ''')

# loading the saved models
ache_model = pickle.load(open('rfmodel.pkl', 'rb'))
# Update this to BChE when available
bche_model = pickle.load(open('rf_bchemodel.pkl', 'rb'))

# Define the tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Main', 'What is AChE?', 'What is BChE?', 'Dataset', 'Model performance', 'Python libraries', 'Citing me'])

with tab1:
    st.title('Application Description')
    st.success("""The goal of this project is to create a linear regression model that utilizes ChEMBL bioactivity data 
        to generate inhibitor bioactivity predictions with respect to a specified target of interest. 
        The test case shown here uses Acetylcholinesterase (AChE) and Butyrylcholinesterase (BChE) as targets. 
        These proteins were selected as targets of interest due to their applications in Alzheimer's drug development.
        """)
    
with tab2:
    st.write("""Acetylcholinesterase (AChE) is a cholinergic enzyme primarily found at postsynaptic neuromuscular 
             junctions, especially in muscles and nerves. It immediately breaks down or hydrolyzes acetylcholine (ACh), 
             a naturally occurring neurotransmitter, into acetic acid and choline.
        """)
   
with tab3:
    st.write("""Butyrylcholinesterase (BChE) is a nonspecific cholinesterase enzyme that hydrolyzes many different 
             choline-based esters.
        """)
    
with tab4:
    st.write("""I used the bioactivity data of Acetylcholinesterase from the ChEMBL database. I extracted 7806 
             records which I then filtered down to 6368 records, after removing redundant or erroneous data. Each 
             record had 881 features. I selected the highest-performing features, removing those that exceeded my 
             variance threshold of 0.16. This left me with 145 features.
        """)
    
with tab5:
    st.write("""The models used for the targets were Random Forest Regressors. The model for AChE had an R-Squared 
             of 0.2834 with an accuracy of 82.96%. The model for BChE had an R-Squared of 0.3724 with an accuracy 
             of 84.63%. There is much room for improvement and further research.
        """) 

with tab6:
    st.write("""Kindly refer to the tutorial document on the Github repo of this project to find all required Python 
             libraries.
        """)  

with tab7:
    st.write("""Coming soon . . .""")   
    
# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            'AChE Inhibitor Prediction model using pubchemfingerprints',
            'BChE Inhibitor Prediction model using pubchemfingerprints'
        ],
    )

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    subprocess.run("padel.sh", shell=True)
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)  # Convert DataFrame to CSV format
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV string in base64
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'  # Create download link
    return href

# Model building
def build_model(modelv, input_data, molecule_names):
    # Apply model to make predictions
    prediction = modelv.predict(input_data.values)
    prediction_output = pd.Series(prediction, name='pIC50')
    df = pd.concat([molecule_names, prediction_output], axis=1)
    
    # Convert pIC50 to IC50 to determine the bioactivity class
    def IC50(input):
        return 10**(-input) * (10**9)

    df['IC50'] = df['pIC50'].apply(IC50)
    df['Bioactivity'] = df['IC50'].apply(lambda x: 'inactive' if x >= 10000 else 'active' if x <= 1000 else 'intermediate')

    st.header('**Prediction output**')
    st.write(df)

    st.markdown(filedownload(df), unsafe_allow_html=True)

# Prediction logic for selected model
if selected == 'AChE Inhibitor Prediction model using pubchemfingerprints':
    st.title('Predict bioactivity of molecules against AChE using pubchemfingerprints')

if selected == 'BChE Inhibitor Prediction model using pubchemfingerprints':
    st.title('Predict bioactivity of molecules against BChE using pubchemfingerprints')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    load_data = pd.read_csv(uploaded_file)
    load_data.iloc[:, 0] = load_data.iloc[:, 0].str.replace(' ', '')  # Remove whitespace from SMILES strings
    st.header('**Input SMILES**')
    st.write(load_data)
    
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

    if st.sidebar.button('Predict'):
        @dataclass
        class Program:
            progress: int = 0

            def increment(self):
                self.progress += 1
                sleep(0.01)

        my_bar = st.progress(0, text="Prediction in progress. Please wait.")
        p = Program()

        desc_calc()

        while p.progress < 100:
            p.increment()
            my_bar.progress(p.progress, text=f"Prediction in progress: {p.progress}%")
        
        my_bar.empty()

        # Read in calculated descriptors and display the dataframe
        st.header('**Compound\'s Computed Descriptors**')
        desc = pd.read_csv('descriptors_output.csv')
        st.write(desc)

        st.header('**Selection of Compound\'s Computed Descriptors**')
        
        if selected == 'AChE Inhibitor Prediction model using pubchemfingerprints':
            Xlist = list(pd.read_csv('features.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            # Apply trained model to make prediction on query compounds
            build_model(ache_model, desc_subset, load_data.iloc[:, 0])
            
        elif selected == 'BChE Inhibitor Prediction model using pubchemfingerprints':
            Xlist = list(pd.read_csv('bche_features.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            # Apply trained model to make prediction on query compounds
            build_model(bche_model, desc_subset, load_data.iloc[:, 0])

else:
    st.warning('Please upload a CSV file.')
