#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def loadcsvfile():
    csvfile = 'coe-course-prefs.csv'
    df = pd.read_csv(csvfile, dtype='str', header=0, sep = ",", encoding='latin') 
    return df

def createPlots(df, columnName):
    st.write('Distribution by ' + columnName)
    scounts=df[columnName].value_counts()
    labels = list(scounts.index)
    sizes = list(scounts.values)
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels = labels, textprops={'fontsize': 10}, startangle=140, \
            autopct='%1.0f%%', colors=sns.color_palette('Set2'))
    plt.subplot(1, 2, 2)
    p = sns.barplot(x = scounts.index, y = scounts.values, palette= 'viridis')
    plt.setp(p.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    # get value counts and percentages of unique values in column 
    value_counts = df[columnName].value_counts(normalize=True)
    value_counts = value_counts.mul(100).round(2).astype(str) + '%'
    value_counts.name = 'Percentage'

    # combine counts and percentages into a dataframe
    result = pd.concat([df[columnName].value_counts(), value_counts], axis=1)
    result.columns = ['Counts', 'Percentage']
    st.write(pd.DataFrame(result))
    
    return

def horizontal_barplot(df, column):
    # Create a horizontal bar plot
    scounts=df[column].value_counts()
    labels = list(scounts.index)
    sizes = list(scounts.values)

    fig = plt.figure(figsize=(8,len(labels)*.5))
    sns.set(style="darkgrid")
    sns.barplot(x=sizes, y=labels, color="b")

    # Set plot title and axis labels
    plt.title("Horizontal Bar Plot")
    plt.xlabel("No. of Respondents")
    plt.ylabel(column)
    #plt.xlim(0, 1)

    # Show the plot
    st.pyplot(fig)

    # get value counts and percentages of unique values in column 
    value_counts = df[column].value_counts(normalize=True)
    value_counts = value_counts.mul(100).round(2).astype(str) + '%'
    value_counts.name = 'Percentage'

    # combine counts and percentages into a dataframe
    result = pd.concat([df[column].value_counts(), value_counts], axis=1)
    result.columns = ['Counts', 'Percentage']
    st.write(pd.DataFrame(result))    

def createTable(df, columnName):  
    st.write('Graduate Distribution by ' + columnName)
    # get value counts and percentages of unique values in column 
    value_counts = df[columnName].value_counts(normalize=True)
    value_counts = value_counts.mul(100).round(2).astype(str) + '%'
    value_counts.name = 'Percentage'

    # combine counts and percentages into a dataframe
    result = pd.concat([df[columnName].value_counts(), value_counts], axis=1)
    result.columns = ['Counts', 'Percentage']
    st.write(pd.DataFrame(result))
    
    return

def twowayPlot(df, var1, var2):
    fig = plt.figure(figsize =(10, 3))
    p = sns.countplot(x=var1, data = df, hue=var2, palette='bright')
    _ = plt.setp(p.get_xticklabels(), rotation=90) 
    st.pyplot(fig)

# Define the Streamlit app
def app():
    st.title("2023 WVSU College of Education Course Preference Survey")      
                 
    st.write("The qualifiers for the interview among the incoming first year students of the College \
    of Education were asked to participate in this course-preference survey. There were 207 respondents \
    who filled up the survey form.")
                 
    st.write("A course preference study is a research study conducted to examine the preferences \
    and interests of individuals regarding their choice of courses or subjects to pursue in an \
    educational setting. It aims to gather data on the factors that influence students' \
    decision-making process when selecting courses and to gain insights into their motivations \
    and preferences..")

    df = loadcsvfile()
    st.write('Distribution by Sex')
    createPlots(df, 'Sex')
    
    st.write('Distribution by Civil Status')
    createPlots(df, 'Civil Status')

    st.write('Distribution By First Priority Course')
    horizontal_barplot(df, 'First Priority')
    

#run the app
if __name__ == "__main__":
    app()
