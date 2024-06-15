import numpy as np
import os
import pandas as pd
import streamlit as st
# import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import subprocess

logo = Image.open("bch project logo placeholder.png")
st.image(logo)

st.title("Alzheimer's Drug Prediction")
st.divider()

st.markdown('''By Onomeyimi Onesi
          
Dataset obtained from [ChEMBL database](https://www.ebi.ac.uk/chembl/g/#search_results/targets/query=ACHE) ''')



tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['About', 'What is AChE?', 'Dataset', 'Model performance', 'Python libraries', 'Citing me'])

with tab1:
    st.write("""The goal of this project is to create a linear regression model that utilizes ChEMBL bioactivity data 
to generate inhibitor bioactivity predictions with respect to a specified target of interest. 
The test case shown here uses Acetylcholinesterase (AchE) as a target. 
This protein was selected as a target of interest due to its applications in alzheimers drug development.

""")

with tab2:
    st.write("""Acetylcholinesterase (AChE) is a cholinergic enzyme primarily found at postsynaptic neuromuscular 
             junctions, especially in muscles and nerves. It immediately breaks down or hydrolyzes acetylcholine (ACh), 
             a naturally occurring neurotransmitter, into acetic acid and choline.""")
    
with tab3:
    st.write("""I used the bioactivity data of Acetylcholinesterase from the ChEMBL database. I extracted 7806 
             records which I then filtered down to 6368 records, after removing redundant or erroneous data. Each 
             record had 881 features. I selected the highest-performing features, removing those that exceeded my 
             variance threshold of 0.16. This left me with 145 features.""")
    
with tab4:
    st.write("""The model used was a Random Forest Regressor with an R-Squared of 0.2834 with ana ccuracy of 82.96%. There is much room for improvement.""") 

with tab5:
    st.write("Kindly refer to the tutorial document on the Github repo of this project to find all required Python libraries.")  

with tab6:
    st.write("""Coming soon . . .""")    

st.sidebar.header(("User Input"))

SMILES_input = "CCCCC\nCCCC\nCN"
chem_id = "A1"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = SMILES.strip()
SMILES = SMILES.replace(" ", "")
SMILES = SMILES.replace("\n", "")

st.header("Input SMILES")
SMILES

drug = [[chem_id, SMILES]]
df = pd.DataFrame(drug)
df.columns = ["chem_id", "canonical_smiles"]

# calculating molecular descriptors
def lipinski(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    i = 0
    baseData = np.array([])  # Initialize an empty array 

    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
        
        row = np.array([desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]).reshape(1, -1)  # Reshape row to have 1 row and 4 columns

        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        
        i += 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data = baseData, columns = columnNames)

    os.remove('molecule.smi')

    return descriptors

df_lipinski = lipinski(df.canonical_smiles)

# combine the dataframes of the molecular descriptors and the original data
df2 = pd.concat([df, df_lipinski], axis = 1)

selection = ["canonical_smiles", "chem_id"]
df2_selection = df2[selection]
df2_selection.to_csv("molecule.smi", sep="\t", index = False, header = False)

subprocess.run("padel.sh", shell = True)

df3_X = pd.read_csv('descriptors_output.csv')
df3_X = df3_X.drop(columns=['Name'])

st.header("Compound's Computed Descriptors")
df3_X

features = pd.read_csv("features.csv")

df4_X = df3_X[features.columns]

# convert pIC50 to IC50 to determine the bioactivity class
def IC50(input):
    i = 10**(-input) * (10**9)

    return i

# Load the model
load_model = pd.read_pickle("rfmodel.pkl")

# st.balloons()
# st.snow()

st.header("Predicted Value")

if df4_X.shape == (0, 145):
    st.write("The predicted value for this compound could not be determined successfully :(. Try again.")
else:
    prediction = load_model.predict(df4_X)

    IC50pred = IC50(prediction[0])

    # assigning the bioactivity class
    bioactivity_threshold = []
        
    if float(IC50pred) >= 10000:
        bioactivity_threshold.append("inactive")
    elif float (IC50pred) <= 1000:
        bioactivity_threshold.append("active")
    else:
        bioactivity_threshold.append("intermediate")

    st.write("The compound {} is an {} inhibitor with a pIC50 value of {}".format(SMILES, bioactivity_threshold[0], prediction[0]))