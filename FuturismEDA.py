#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os
#from chart_studio import plotly
#import plotly.offline as offline
#import plotly.graph_objs as go
#offline.init_notebook_mode()
#import plotly.graph_objects as go
import seaborn as sns
from pandas import Series
from matplotlib import pyplot
from tqdm import tqdm
import streamlit as st
from PIL import Image


def main():
    st.title("Futurism Technologies Pvt. Ltd.")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Exploratory Data Analysis</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    #taking input of data
    uploaded_file = st.file_uploader("Choose a CSV file")
    
    #taking inputs of main model and nested model
    target = st.text_input("Name of Target KPI","Type here")
    
    if st.button("Run the analysis"):
        df = pd.read_csv(uploaded_file)
        pd.set_option('display.max_columns', None)
        st.dataframe(df) # to show dataframe
        
        st.write('Shape of the Dataframe: ',df.shape)
        st.write('Number of columns in dataset: ',df.shape[1])
        st.write('Number of datapoints in dataset: ',df.shape[0])
        st.write('Target column: ',target)   # name of the target column
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
                #print(f"Column has multiple data types. Replaced values based on the most common data type.")
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
            #print(f"Analyzing and replacing values in '{column}' column:")
            df[column] = analyze_and_replace_datatypes(df[column])
            #print()
        if df[target].dtype=='O':
            df[target]= pd.factorize(df[target])[0]

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = list(df.select_dtypes(include=numerics).columns).remove(target)
        #numeric_columns= list(df.select_dtypes('number')).remove(target)  # to numeric feature names from the dataset excluding target variable
        st.write(numeric_columns)
        categorical_columns= list(df.select_dtypes('object'))  # to categorical feature names from the dataset excluding target variable
        st.write(categorical_columns)
        #st.write(df.info())  
        #st.write(df.describe())
        
        #correlation plot
        df_corr= df.corr(numeric_only=True)
        fig = plt.figure(figsize=(8,8))
        sns.heatmap(df_corr,cmap='PuBuGn',annot=True)
        st.pyplot(fig)
        
        #univariate pdf plots for numeric variables
        for column in numeric_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f'Univariate Distribution of {column}')
            st.pyplot(fig)
        #countplot for categorical variables
        for column in categorical_columns:    
            plt.figure(figsize=(8, 6))
            sns.countplot(dataframe[column])
            plt.title(f'Countplot of {column}')
            st.pyplot(fig)
        #bivariate plot for numeric variable 
        for column in numeric_columns:    
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=column, y=target, data=dataframe)
            plt.title(f'Bivariate Scatter Plot: {column} vs. target')
            st.pyplot(fig)
        for column in categorical_columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=column, y=target, data=dataframe)
            plt.title(f'Bivariate Boxplot: {column} vs. target')
            st.pyplot(fig)
        
        #st.write("The residual plot and waterfall for incrementality are generated and saved in 'C:/Users/akshay.bhasme/Documents/MMM_streamlit/'")

if __name__=='__main__':
    main()



# In[ ]:




