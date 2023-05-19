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

def twoway_plot(df, var1, var2):
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
    
    st.write('Distribution By Province')
    horizontal_barplot(df, 'Province')
    
    st.write('Distribution of First Priority and Sex')
    twoway_plot(df, 'First Priority', 'Sex')
    
    st.write('Distribution of First Priority and Province')
    twoway_plot(df, 'First Priority', 'Province')
    
    st.subheader('Natural Language Processing (NLP)')
    
    st.write('Natural Language Processing (NLP) methods were applied in the analysis of course preference data to extract insights from textual information provided by participants. Here are the NLP methods used in analyzing course preference data:')

    st.write('Text Preprocessing: Before applying NLP techniques, it\'s essential to preprocess the textual data. This typically involves tasks such as removing punctuation, converting text to lowercase, removing stopwords (common words like "the," "and," etc.), and performing stemming or lemmatization to reduce words to their root forms. This preprocessing step helps to standardize the text and improve the accuracy of subsequent NLP analyses.')

    st.write('Sentiment Analysis: Sentiment analysis, also known as opinion mining, is used to determine the sentiment expressed in the text. In the context of course preference analysis, sentiment analysis can be used to gauge participants\' attitudes or emotions towards different courses. This analysis can be performed by using machine learning algorithms or predefined sentiment lexicons to classify text as positive, negative, or neutral.')

    st.write('POS tagging and chunking were methods used in the course preference analysis to extract relevant information from textual data related to course preferences. Here\'s an overview of how these techniques were applied:')

    st.write('POS Tagging:\n \
POS tagging assigns part-of-speech tags to each word in a sentence or text. In the context of course preference analysis, POS tagging can help identify the grammatical roles and categories of words, allowing for a better understanding of the text. By applying POS tagging, you can extract relevant information such as course names, subjects, or academic disciplines mentioned in the text.')

    st.write('For example, if a sentence mentions "I am interested in teaching english teaching courses," POS tagging would label "I" as a pronoun, "am" as a verb, "interested" as an adjective, "in" as a preposition, "teaching english" as a noun phrase, and "courses" as a noun. This tagging can help identify the subject of interest (teaching english) and the type of courses mentioned.')

    st.write('Chunking:\n \
Chunking, also known as shallow parsing, involves grouping words into meaningful phrases or chunks based on their grammatical structure. It helps extract higher-level syntactic units from the text. In the context of course preference analysis, chunking can be used to identify and extract phrases or noun phrases that provide insights into participants\' preferences.')
    
    st.write('For example, consider the sentence: "I prefer advanced mathematics and physics courses." Chunking would identify and extract the noun phrases "advanced mathematics" and "physics courses." These chunks can provide information about the specific subjects or academic areas that the participant prefers.')

    st.write('By combining POS tagging and chunking, we can extract relevant phrases or noun phrases related to course preferences. These extracted chunks can be further analyzed to identify popular subjects, specific course preferences, or academic disciplines that are frequently mentioned by participants.')
    
#run the app
if __name__ == "__main__":
    app()
