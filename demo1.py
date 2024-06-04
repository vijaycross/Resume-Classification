# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:04:45 2024

@author: dell
"""

# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import spacy
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import sklearn
import PyPDF2
import nltk
import pickle as pk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer

# Specify NLTK data path
nltk_data_path = os.path.join(os.environ['APPDATA'], 'nltk_data')

# Set NLTK data path
nltk.data.path.append(nltk_data_path)

# Now import and download the necessary NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
nltk.download('words', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

en_stopwords = set(stopwords.words('english'))

#----------------------------------------------------------------------------------------------------

st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)

st.subheader('Hey, Welcome')

# FUNCTIONS
def getText(filename):
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
        for paragraph in page_content:
            fullText =  fullText + paragraph
    return fullText


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in en_stopwords]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type = pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile'])
filename = []
predicted = []

#-------------------------------------------------------------------------------------------------
# MAIN CODE
model = pk.load(open("modelRF.pkl", 'rb'))
Vectorizer = pk.load(open("vector.pkl", 'rb'))

MAX_FILE_SIZE_MB = 2  # Maximum file size allowed in MB

upload_file = st.file_uploader('Upload Your Resumes',
                                type=['docx', 'pdf'], accept_multiple_files=True)

for doc_file in upload_file:
    if doc_file is not None:
        # Check file size
        if len(doc_file.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File '{doc_file.name}' exceeds the maximum size limit of {MAX_FILE_SIZE_MB} MB.")
            continue
        
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        
if len(predicted) > 0:
    # Define a mapping dictionary to map label encoded values to original categories
    label_mapping = {
        0: 'PeopleSoft',
        1: 'React JS Developer',
        2: 'SQL Developer',
        3: 'Workday'
        # Add more mappings as needed
    }

    # Predict and map the labels to their original categories
    predicted_categories = [label_mapping[label] for label in predicted]

    # Update the DataFrame with the original category predictions
    file_type['Uploaded File'] = filename
    file_type['Predicted Profile'] = predicted_categories

    # Display the updated DataFrame
    st.table(file_type.style.format())

