#!/usr/bin/env python
# coding: utf-8

# # British Airways Case Study with Python
# 
# ### Zhanara Zeinesheva
# ### 09/05/2024

# # Task 1
# 
# ---
# 
# ## Web scraping and analysis
# 
# This Jupyter notebook includes codes to analyze data with web scraping. I use a package called `BeautifulSoup` to collect the data from the web. The collected data is saved it into a local `.csv` file.
# 
# ### Scraping data from Skytrax
# 
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, I am only interested in reviews related to British Airways and the Airline itself.
# 
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, I can use `Python` and `BeautifulSoup` to collect all the links to the reviews and then to collect the text data on each of the individual review links.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


# In[3]:


df = pd.DataFrame()
df["reviews"] = reviews
df.head()


# In[4]:


df.to_csv("BA_reviews.csv")


# The loops above collected 1000 reviews by iterating through the paginated pages on the website.
# 
# The next thing is data cleaning to remove any unnecessary text from each of the rows. 

# ### Data Cleaning

# Removing "✅ Trip Verified" and "Not Verified" from each row, as it is unnecessary for the analysis.

# In[5]:


# Replace "✅ Trip Verified" and "Not Verified" with an empty string
df['reviews'] = df['reviews'].str.replace('✅ Trip Verified', '', regex=False)
df['reviews'] = df['reviews'].str.replace('Not Verified', '', regex=False)

# Show the updated DataFrame
df.head()


# Filter only British Airways reviews.

# In[6]:


# Filter the DataFrame to only include rows where the review contains 'British Airways'
df_british_airways = df[df['reviews'].str.contains('British Airways', case=False, na=False)]

# Show the filtered DataFrame
df_british_airways.head()


# Remove unnecessary characters.

# In[7]:


# Remove any unwanted characters (e.g., â€” or non-ASCII characters)
df['reviews'] = df['reviews'].str.encode('ascii', 'ignore').str.decode('ascii')

# Display cleaned data
df.head()


# I use a sentiment analysis to analyze the tone of reviews.

# In[9]:


get_ipython().system('pip install textblob')


# In[10]:


import nltk
nltk.download('punkt')


# In[11]:


from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the function to the reviews
df['sentiment'] = df['reviews'].apply(get_sentiment)

# Display the DataFrame with sentiment scores
df.head()


# Extract keywords or topics to identify the most common themes in the reviews.

# In[12]:


from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the reviews
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['reviews'])

# Get the frequency of each word
word_freq = X.sum(axis=0)
words_freq_df = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'frequency': word_freq.A1})

# Sort by frequency and display top words
words_freq_df = words_freq_df.sort_values(by='frequency', ascending=False)
words_freq_df.head(10)


# Use Latent Dirichlet Allocation (LDA) to discover underlying topics or themes in the reviews.

# In[13]:


from sklearn.decomposition import LatentDirichletAllocation

# Use CountVectorizer to get word frequency matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['reviews'])

# Perform LDA
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)


# Split the reviews into positive, neutral, and negative categories.

# In[14]:


# Classify reviews based on sentiment
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Analyze each category
positive_reviews = df[df['sentiment_category'] == 'Positive']
negative_reviews = df[df['sentiment_category'] == 'Negative']

# Display results
print("Positive Reviews:", positive_reviews['reviews'].head())
print("Negative Reviews:", negative_reviews['reviews'].head())


# ### Data Analysis and Visualization

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot sentiment distribution
sns.countplot(x='sentiment_category', data=df)
plt.title('Sentiment Distribution of Reviews')
plt.show()


# In[31]:


# Filter the DataFrame for Negative and Neutral reviews
filtered_df = df[df['sentiment_category'].isin(['Negative', 'Neutral'])]


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

# Join all reviews into a single string
text = ' '.join(filtered_df['reviews'])


# In[43]:


# Use CountVectorizer to extract 4- and 5-word phrases and remove stopwords
vectorizer = CountVectorizer(ngram_range=(4, 5), stop_words='english')
ngram_counts = vectorizer.fit_transform([text])


# In[44]:


# Convert n-gram counts to a DataFrame
ngram_count_df = pd.DataFrame(ngram_counts.toarray(), columns=vectorizer.get_feature_names_out()).T
ngram_count_df.columns = ['count']


# In[45]:


# Sort by the most frequent 4- and 5-word phrases
ngram_count_df = ngram_count_df.sort_values(by='count', ascending=False).head(20)


# In[46]:


# Print the top 20 most frequent 4- and 5-word phrases
print(ngram_count_df)


# In[49]:


# Plot the top 20 trigrams
ngram_count_df.plot(kind='bar', figsize=(10, 5))
plt.title('Top 20 Ngrams in Negative and Neutral Reviews')
plt.xlabel('Ngrams')
plt.ylabel('Count')
plt.show()


# ## Recommendations

# #### Improve Customer Service:
# Phrases like "customer service non existent" and "british airways customer service" are highly frequent, indicating dissatisfaction with customer service.
# * Recommendation: Invest in training customer service staff to be more responsive and empathetic. Consider implementing a dedicated, more accessible customer support line or chat service, particularly for resolving issues quickly.
# 
# #### Enhance Business Class Experience:
# The terms "worst business class experience" and "british airways business class" highlight a need to improve the business class experience.
# * Recommendation: Re-evaluate the business class offerings such as seating comfort, food quality, in-flight comfort, and customer service for premium customers. Ensure passengers are receiving the value they expect when purchasing business class tickets.
# 
# #### Streamline Check-in and Booking Processes:
# Phrases like "prior 24 hour check," "minute charged select," "check online wouldn’t accept," and "tried manage booking online" show that check-in and online booking systems are causing frustration.
# * Recommendation: Invest in improving the check-in services, both online and at the airport. Ensure the online booking system is user-friendly, intuitive, and responsive. Provide clear communication for issues during online booking or check-in and resolve technical problems proactively.
# 
# #### Improve Handling of Delays and Cancellations:
# Phrases such as "cancelled flight home," "ba cancelled return flight," "delayed missed connecting flight" indicate recurring issues with delays and flight cancellations.
# * Recommendation: Improve the communication around delays and cancellations. Proactively inform passengers about the status of their flights, and provide compensation or alternative travel options when significant delays or cancellations occur. Have backup plans for handling missed connecting flights.
# 
# #### Enhance In-Flight Services:
# The phrase "cost airline customer service" suggests passengers may feel they are not receiving adequate value for the cost, especially in in-flight services.
# * Recommendation: Review and upgrade in-flight services like meal options, seating comfort, and entertainment. Ensure that the service provided is proportionate to the price passengers are paying.
# 
# #### Address Regional Concerns:
# Certain geographic locations like "madrid london british airways" and "buenos aires london heathrow" come up frequently, which may indicate recurring issues on specific routes or regions.
# * Recommendation: Investigate if certain regions or routes have more operational issues than others. If specific routes are repeatedly facing problems, assign more resources and oversight to ensure smoother operations.
