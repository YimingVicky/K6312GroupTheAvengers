
import pandas as pd 
import numpy as np 

df1=pd.read_csv('movies_matched.csv') # see hybrid model data cleaning



# In[2]:


# Parse the stringified features into their corresponding python objects
features = ['actors', 'director', 'writer', 'genre']


# In[3]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, str):
        split_list = x.split(',')
        if len(split_list) > 1:
            names = [i for i in split_list]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
        elif len(split_list) == 1:
            names = [x]
    else:
        names = []
    return names


# In[4]:


features = ['actors', 'director', 'writer', 'genre']
# features = ['actors']
for feature in features:
    df1[feature] = df1[feature].apply(get_list)



# In[6]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[7]:


# Apply clean_data function to your features.
features = ['actors', 'director', 'writer', 'genre']

for feature in features:
    df1[feature] = df1[feature].apply(clean_data)



# In[9]:


def create_soup(x):
    return ' '.join(x['writer']) + ' ' + ' '.join(x['actors']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genre'])
df1['soup'] = df1.apply(create_soup, axis=1)



# In[11]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['soup'])


# In[12]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[13]:


# Reset index of our main DataFrame and construct reverse mapping as before
df1 = df1.reset_index()
indices = pd.Series(df1.index, index=df1['original_title'])


# In[16]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[original_title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df1['original_title'].iloc[movie_indices]



