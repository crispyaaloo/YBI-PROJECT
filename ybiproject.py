import requests
import pandas as pd
import re
from collections import Counter
from bs4 import BeautifulSoup

#API_KEY = ""
#API_KEY_SECRET = ""
#BEARER_TOKEN = ""
#insert your own twitter api keys in here

handle = "CommBank"
url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{handle}&tweet.fields=created_at&expansions=author_id&user.fields=created_at"
headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}



import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_tweets(handle, num_tweets=100):
    # Initialize an empty list to store tweets
    tweet_list = []
    url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{handle}&tweet.fields=created_at&expansions=author_id&user.fields=created_at"
    headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # The response is JSON, not HTML. Use response.json() instead of BeautifulSoup
        data = response.json()

        # Check if 'data' key exists in the response
        if 'data' in data:
            for tweet in data['data']:
                # Extract the tweet text from the 'text' field
                tweet_text = tweet['text'].strip()
                tweet_list.append(tweet_text)

                if len(tweet_list) >= num_tweets:
                    break
        else:
            print("No tweets found in the response.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return [] # Return empty list if an error happens

    return tweet_list # Return the list of tweets


handle = "CommBank"
tweets = get_tweets(handle, 100)

if tweets:
    df = pd.DataFrame({'Tweet Text': tweets})
    print(df)

else:
    print("Failed to retrieve tweets or no tweets found")

df.to_csv('CommBank.csv', index=False)

lt = pd.read_csv('CommBank.csv')
lt.head()


import nltk
import pandas as pd
import re
import emoji
nltk.download()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def clean_text(text):
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special symbols and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Assuming 'df' is your DataFrame and 'Tweet Text' is the column name
lt = pd.read_csv('CommBank.csv')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

lt['cleaned_tweet'] = lt['Tweet Text'].apply(clean_text)

lt['tokenized_tweet'] = lt['cleaned_tweet'].apply(word_tokenize)

lt['pos_tags'] = lt['tokenized_tweet'].apply(pos_tag)

lt['stemmed_tweet'] = lt['tokenized_tweet'].apply(lambda tokens: [stemmer.stem(w) for w in tokens])

lt['lemmatized_tweet'] = lt['tokenized_tweet'].apply(lambda tokens: [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens])

print(lt.head())

# prompt: remove the first word of each tweet

import pandas as pd

lt = pd.read_csv('CommBank.csv')

def remove_first_word(tweet):
    words = tweet.split()
    if len(words) > 1:
        return " ".join(words[1:])
    else:
        return ""

lt['Tweet Text'] = lt['Tweet Text'].apply(remove_first_word)
print(lt.head())

# prompt: convert this clean data into a csv

import pandas as pd
lt.to_csv('cleaned_tweets.csv', index=False)
ct = pd.read_csv('cleaned_tweets.csv')
ct.head()

from textblob import TextBlob

ct = pd.read_csv('cleaned_tweets.csv')

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Get polarity score


ct['sentiment_score'] = ct['Tweet Text'].apply(get_sentiment)

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

ct['sentiment_category'] = ct['sentiment_score'].apply(categorize_sentiment)

# Print or analyze sentiment distribution
print(ct.head())
print(ct['sentiment_category'].value_counts())

