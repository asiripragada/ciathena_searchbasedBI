import streamlit as st

PAGE_CONFIG = {"page_title":"Athena","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
 
from PIL import Image
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import os
import re
#import nltk
#import numpy as np
#from nltk.corpus import stopwords

#import spacy
#from spacy import displacy
#from collections import Counter
#import en_core_web_sm
#from collections import OrderedDict
from pprint import pprint
#import itertools



st.title('Athena')
st.markdown('<style>h1{color: orange; text-align: center}</style>', unsafe_allow_html=True)
st.markdown('<style>p{color: black;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: blue;}</style>', unsafe_allow_html=True)
image = Image.open('CI ORPHEUS LOGO+TEXT.png')
st.sidebar.image('CI ORPHEUS LOGO+TEXT.png',width=200)
st.sidebar.subheader('**Athena**')
st.sidebar.write('eMail configurator helps you to classify given eMail to specific category. Each category is assigned to concern department in the organisation. The application provides automated solution  to route eMails to the concern department with minimal human intervention')
st.set_option('deprecation.showPyplotGlobalUse', False)
col1, col2, col3 = st.sidebar.columns([1,1,1])
with col1:
  st.image("CIAI LOGO_Original-01.png")
  st.sidebar.markdown('[Read more about CIAI](https://www.customerinsights.ai/)')
with col2:
  st.write("")
with col3:
  st.write("")

#nlp_en_core_web_sm = spacy.load('en_core_web_sm')

#def extract_named_ents(text):
    #"""Extract named entities, and beginning, middle and end idx using spaCy's out-of-the-box model. 
    
    #Keyword arguments:
    #text -- the actual text source from which to extract entities
    
    #"""
    #return [(ent.text) for ent in nlp_en_core_web_sm(text).ents]

# def entity_analyzer(my_text):
#     nlp = spacy.load("en_core_web_sm")
#     docx = nlp(my_text)
#     s1 = spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
#     return s1

def main():
  data=st.file_uploader("Upload file",type=['csv', 'excel'])
  #st.success("success")
  
  if data is not None:
      df=pd.read_csv(data)
      st.dataframe(df.head())
      if st.checkbox('Select Multiple Columns to plot'):
          selected_columns=st.multiselect("Select Preferred columns",df.columns)
          df1=df[selected_columns]
          st.dataframe(df1)
      if st.checkbox("Display Hist"):
          fig, ax = plt.subplots()
          df1.hist(bins=8, grid=False, figsize=(8, 8), color="#86bf91", zorder=2, rwidth=0.9, ax=ax,)
          st.pyplot()
      # if st.checkbox("Display Linechart"):
      #     st.bar_chart(df1)		
      #     st.pyplot()
      if st.checkbox("Display Heatmap"):
          st.write(sns.heatmap(df1.corr(), vmax=1, square=True, annot=True,cmap='viridis'))		
          st.pyplot()
      if st.checkbox("Display Pairplot"):
          st.write(sns.pairplot(df1,diag_kind='kde'))
          st.pyplot()
      if st.checkbox("Pie Chart"):
          all_columns=df.columns.to_list()
          pie_columns=st.selectbox("Select a column, NB: Select Target column",all_columns)
          pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
          st.write(pieChart)
          st.pyplot()
          
      #text=st.text_area("Enter the text","Type Here ..")
      #st.markdown("**Named Entity Recognition:**")
      #if st.checkbox("Named Entities of input text"):
#         st.subheader("Analyze Your Text")
#         message = st.text_area("Enter Text for NER","Type Here ..")
#         if st.button("Extract"):
          #entity_result = extract_named_ents(text)
          #st.markdown(entity_result)

if __name__ == '__main__':
	main()  

