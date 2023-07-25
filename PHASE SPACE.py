#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nltk
from gensim import corpora, models
nltk.download('brown')
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
import networkx as nx
from nltk.metrics import edit_distance
from nltk.metrics import jaccard_distance
nltk.download('omw-1.4')
nltk.download('punkt')
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import stem
stemmer = stem.PorterStemmer()
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
import string
nltk.download('vader_lexicon')
punct = list(string.punctuation)
get_ipython().system('pip install PRAW')
from mpl_toolkits.mplot3d import Axes3D
import praw
import datetime
import nltk.sentiment.vader as vd
import re
import pandas as pd
import seaborn as sns
sns.set()
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import IFrame
import random
from collections import Counter
import random
from random import sample
from scipy.stats import entropy
import requests
import os
import glob
from tqdm import tqdm
get_ipython().system('pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_lexicon = SentimentIntensityAnalyzer()
from wordcloud import WordCloud
get_ipython().system('pip install wordcloud')
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
import spacy
from sklearn.cluster import KMeans
get_ipython().system('pip install gensim')
import gensim
from gensim import corpora
from gensim import corpora, models


# In[8]:


data = {
    'Text': [
        'Find peace everyday with Mindfulness App',
        'Your All-In-One App Mindfulness Wellbeing',
        'Transform your sleep wellbeing life with personalised all-in-one wellness app',
        'Alter your state mind choose how you feel perform',
        'Corporate, Clinical Education Clients',
        'Find Your Calm',
        'most relaxing app world',
        'Our goal help you improve your health happiness',
        'Find more joy',
        'Meditation sleep made simple',
        'Make every day happier',
        'Improve your stress, sleep more with world’s first personalised meditation program, now free  your first year',
        'Meditation adapts you',
        'personalisation matters',
        'AI-Therapy works everyone',
        'Personalised Interactive Efficient',
        'worlds first intelligent journal',
        'Your personal mental health companion',
        'Everyday Mental Maintenance',
        "We're all bit messy",
        'Support your people your business',
        'Mental Health Starts You',
        'Support across your mental health Journey',
        'Self-care your mental Well-Being',
        'Transform you Transform world',
        'meditate breathe stretch explore',
        'Open Letters',
        'Therapy Demand',
        'Works around your life',
        'Transport your own meditative island',
        'meet your spirit companion explore mindfulness portals designed experts',
        'Mindfulness imperative self-reflection growth',
        'two things we could all more consistent meditation practice helps teach us mindfulness,',
        'unlocking key better mental health wellbeing',
        'Let powers Maloka spark your path toward life filled more joy more love more shine',
        'personalization matters',
        'Get guidance experts',
        'Complete your hand-picked daily recommendations steadily grow your meditation practice,',
        'explore Singles enhance moments your everyday life',
        'Explore innovative activities',
        'Relax deeply suite innovative experiences vibration-based Immersive Meditations interactive',
        'Wind Down Single',
        'Balance breaks down meditation into concrete trainable skills, tracks time you spend',
        'practicing each',
        'Whether you want boost your mental health, record your mood real-time therapist reinforce',
        'strategies you’ve learned your',
        'stay top mental health feeling emotionally distressed been diagnosed mental illness.',
        'Gain Insight on Your Mental Health',
        'Detect Your Patterns',
        'Identify Your Thoughts and Feelings',
        'Take action',
        'Make yourself more mindfully aware  present moment attention-stimulating',
        'gameplay mechanics',
        'Train your nervous system deeply relax greater present moment awareness',
        'unwind end busy day',
        'Designed help you explore deeper connection your life experiences',
        'Take vacation far off land experience sights sounds nature',
        'Building more kind equal world tomorrow',
        'Driven make change Created care',
        'Giving everyone world class therapy experience',
        'Put your mind bed wake up refreshed make good days your new normal.',
        'Catch your breath relax your mind feel less stressed just days'
    ]
}

df = pd.DataFrame(data)
print(df)


# In[9]:





# Function to tokenize and lemmatize the text into a list of words
def tokenize_and_lemmatize(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words

# Create the DataFrame with 'Text' column
df = pd.DataFrame(data)

# Tokenize and lemmatize all sentences in the 'Text' column and store them as 'all_words' in the DataFrame
df['all_words'] = df['Text'].apply(tokenize_and_lemmatize)

# Load the VAD scores from the Warriner rescale
vad = pd.read_csv('/Users/felixhawkings/Desktop/Warriner_rescale.csv', index_col=0)
vad = vad[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
vad.columns = ['Valence', 'Arousal', 'Dominance']

# Create a new DataFrame with individual words and their VAD scores
individual_words = [word for words in df['all_words'] for word in words]
vad_scores = [vad.loc[word.lower()].values if word.lower() in vad.index else [np.nan, np.nan, np.nan] for word in individual_words]

new_df = pd.DataFrame({
    'Word': individual_words,
    'Valence': [score[0] for score in vad_scores],
    'Arousal': [score[1] for score in vad_scores],
    'Dominance': [score[2] for score in vad_scores]
})

print(new_df)


# In[10]:


import plotly.graph_objs as go

# Assuming you have defined valence_scores, arousal_scores, and dominance_scores
valence_scores = new_df['Valence']
arousal_scores = new_df['Arousal']
dominance_scores = new_df['Dominance']

# Create a list of custom labels for each data point (tooltip text) including the words from the original dataframe
hover_labels = [f"Word: {word}<br>Valence: {valence:.2f}<br>Arousal: {arousal:.2f}<br>Dominance: {dominance:.2f}" for word, valence, arousal, dominance in zip(new_df['Word'], valence_scores, arousal_scores, dominance_scores)]

# Create a trace for the scatter plot
trace = go.Scatter3d(
    x=valence_scores,
    y=arousal_scores,
    z=dominance_scores,
    mode='markers',
    text=hover_labels,    # Set the custom labels for each data point (tooltip text)
    hoverinfo='text',     # Set hoverinfo to 'text' to show only the custom labels on hover
    marker=dict(
        size=5,
        color=valence_scores,  # Set color to valence scores
        colorscale='Viridis',  # Set the color scale
        opacity=0.8
    )
)

# Create the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Valence'),
        yaxis=dict(title='Arousal'),
        zaxis=dict(title='Dominance')
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[trace], layout=layout)

fig.show()

#plot_file = "vad_plot37.html"
#fig.write_html(plot_file)


# In[11]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you already have the DataFrame 'df' with the "Text" column

# Join all the text in the "Text" column into a single string
all_text = ' '.join(df['Text'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

