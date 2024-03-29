#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from getpass import getpass
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import PIL.Image
import time
import plotly.express as px

st.title("Futurism Technologies Pvt. Ltd.")
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Exploratory Data Analysis using LLM (Google Gemini)</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file")
    
if st.button("Run the analysis"):
    df = pd.read_csv(uploaded_file)
    pd.set_option('display.max_columns', None)
    st.dataframe(df) # to show dataframe
    st.write('Shape of the Dataframe: ',df.shape)
    st.write('Number of columns in dataset: ',df.shape[1])
    st.write('Number of datapoints in dataset: ',df.shape[0])
    def analyze_and_replace_datatypes(column):
        """if dataframe column contains values of multiple datatypes. and to find the datatype which is appearing maximum 
        in the column. If max appearing dtype is string then replace values of other dtypes with the value that is 
        appearing most in string dtype. If max appearing dtype is numeric then replace values of other dtypes with the 
        mean of the values of numeric dtype. And in the end values in that column can be converted to float, then convert 
        it into float
        """
        # Count occurrences of each data type in the column
        dtype_counts = column.apply(type).value_counts()
        # Check if there are multiple data types in the column
        has_multiple_datatypes = len(dtype_counts) > 1
        if has_multiple_datatypes:
            # Find the data type with the maximum count
            most_common_dtype = dtype_counts.idxmax()
            if most_common_dtype == str:
                # If the most common dtype is string, replace values with the most common string value
                most_common_string = column.replace('', pd.NA).mode().iloc[0]
                column = column.apply(lambda x: most_common_string if type(x) != str or x == '' else x)
            elif pd.api.types.is_numeric_dtype(most_common_dtype):
                # If the most common dtype is numeric, replace values with the mean of numeric values
                column = column.apply(lambda x: column.mean() if not pd.api.types.is_numeric_dtype(type(x)) else x)
                
        else:
            #print(f"Column has a single data type: {column.dtype}")
            pass
        # Convert values to float at the end
        try:
            column.replace('', pd.NA).mode().iloc[0]
            column = column.apply(lambda x: most_common_string if type(x) != str or x == '' else x)
        except:
            pass
        try:
            if column[0].replace('.', '', 1).isdigit() :
                column = pd.to_numeric(column, errors='coerce')
                print('Changed to float')
        except:
            pass
        return column
    for column in df.columns:
        df[column] = analyze_and_replace_datatypes(df[column])

    if "api_key" in st.secrets:
        genai.configure(api_key=st.secrets["api_key"])
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns.values)  # to numeric feature names from the dataset excluding target variable 
    st.write('Numeric Columns: ',numeric_columns)
    categorical_columns= list(df.select_dtypes('object'))  # to categorical feature names from the dataset excluding target variable
    st.write("Categorical Columns: ",categorical_columns)
    df[categorical_columns] = df[categorical_columns].apply(lambda x: pd.factorize(x)[0])
    
    #correlation plot
    df_corr= df.corr()
    df_corr= df_corr.round(2)
    fig= px.imshow(df_corr, text_auto=True,color_continuous_scale = 'RdYlBu',width= 1000, height=1000)
    fig.write_image("chart.png")
    st.image("chart.png")
    image = PIL.Image.open('chart.png')
    vision_model = genai.GenerativeModel('gemini-pro-vision')
    response = vision_model.generate_content(["What are the observations and analysis from this graph can be made?",image])
    st.write(response.text)
        
    # Univariate pdf plots for numeric variables
    for i, column in enumerate(numeric_columns):
        fig = plt.figure(figsize=(8,8))
        sns.histplot(df[column], kde=True)
        fig.savefig('univariate.png')
        st.write("Univariate plot for {}".format(column))
        st.image('univariate.png')
        image = PIL.Image.open('univariate.png')
        response = vision_model.generate_content(["What are the observations and analysis from this graph can be made?",image])
        st.write(response.text)
        
    # Countplot for categorical variables
    for i, column in enumerate(categorical_columns):
        if 'ID'not in column:
            fig = plt.figure(figsize=(8,8))
            sns.countplot(x=column, data=df)
            fig.savefig('countplot.png')
            st.write("countplot for {}".format(column))
            st.image('countplot.png')
            image = PIL.Image.open('countplot.png')
            response = vision_model.generate_content(["What are the observations and analysis from this graph can be made?",image])
            st.write(response.text)
    #pair plots
    #fig = plt.figure(figsize=(8,8))
    #sns.pairplot(df.loc[:,numeric_columns], size=3, palette='muted',corner=True)
    #fig.savefig('pairplot.png')
    #st.pyplot(fig)
    #st.write("pairplot for {}".format(column))
    #st.image('pairplot.png')
    #image = PIL.Image.open('pairplot.png')
    #response = vision_model.generate_content(["What are the observations and analysis from this graph can be made?",image])
    #st.write(response.text)
    st.success('Analysis completed successfully!', icon="✅")
   


# In[ ]:




