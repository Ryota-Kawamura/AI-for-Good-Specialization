import pandas as pd
import nltk
import string
import sklearn
import ipywidgets as widgets
from ipywidgets import interact
from termcolor import colored
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from collections import Counter
from typing import Callable, Tuple, List

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Defining some useful globals
STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(["n't", "'s", "'m", "..", "'d", "'c", "u", "'re", "ca", "'ve", "'ll", 'n', "oh", "ha", "u"])
punctuation = string.punctuation + '``'+ "''" 

columns_for_filter = [
    'medical_help', 'medical_products', 'search_and_rescue', 'security',
    'military', 'child_alone', 'water', 'food', 'shelter', 'clothing',
    'money', 'missing_people', 'refugees', 'death', 'other_aid',
    'infrastructure_related', 'transport', 'buildings', 'electricity',
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
    'other_weather'
]

# Clean and load data as in previous notebooks
def process_text(
    text: str,
    tokenizer: Callable,
    pos_tagger: Callable,
    lemmatizer: WordNetLemmatizer,
    stopwords: List[str],
    punctuation: str
) -> List[str]:
    '''Processes the text with the following steps: tokenize, lowercase, remove punctuation,
    remove stopwords, lemmatize.
    
    Args:
        text (str): A string of text
        tokenizer (Callable): a function for tokenizing the text
        pos_tagger (Callable): a function for creating the POS tags
        lemmatizer (WordNetLemmatizer): a function to lemmatize the words
        stopwords (List[str]): A list of stopwords
        punctuation (str): A punctuation string
    
    Returns:
        tokens (List[str]): list of processed tokens
    '''

    # Step 1: Tokenize
    tokens = pos_tagger(tokenizer(text))

    # Step 2: Standardize Lettercase
    tokens = [(w[0].lower(), w[1]) for w in tokens]

    # Step 3: Remove Punctuation
    tokens = [w for w in tokens if w[0] not in punctuation]

    # Step 4: Remove stop words
    tokens = [w for w in tokens if w[0] not in stopwords]

    # Step 5: Lemmatize each word 
    tokens = [lemmatizer.lemmatize(w[0], pos_tag_convert(w[1])) for w in tokens]

    return tokens


def load_and_clean_haiti_data() -> pd.core.frame.DataFrame:
    '''Loads and cleans the dataframes with messages.
    
    Returns:
        haiti_df (pd.core.frame.DataFrame): dataframe with messages
    '''
    # Load in the three dataset splits
    train_df = pd.read_csv("data/disaster_response_training.csv", low_memory=False)
    val_df = pd.read_csv("data/disaster_response_validation.csv", low_memory=False)
    test_df = pd.read_csv("data/disaster_response_test.csv", low_memory=False)

    # Concat the ones that are about the haiti earthquake
    haiti_df = pd.concat([
        train_df[train_df.event=="haiti_earthquake"],
        val_df[val_df.event=="haiti_earthquake"],
        test_df[test_df.event=="haiti_earthquake"],
    ])

    # Fix types
    haiti_df.actionable_haiti = haiti_df.actionable_haiti.astype(int)
    haiti_df.date_haiti = pd.to_datetime(haiti_df.date_haiti)

    # Remove unrelated messages
    haiti_df = haiti_df.query('related == 1')
    
    # Just subset the columns you want
    haiti_df = haiti_df[["id", "date_haiti", "message"]]

    # Rename date column
    haiti_df.rename(columns={"date_haiti": "date"}, inplace=True)

    # Process messages as tokens
    haiti_df["message_tokens"] = haiti_df.message.apply(clean_tokenize_process_text)
    # Save it as a string
    haiti_df["message_proc"] = haiti_df["message_tokens"].apply(lambda x: ' '.join(x))

    # Count number of words and save as a column
    haiti_df["num_words"] = haiti_df.message.apply(lambda x: len(x.split(" ")))

    # Count number of tokens and save as a column
    haiti_df["num_tokens"] = haiti_df.message_tokens.apply(lambda x: len(x))

    # You can drop the rows with only one token
    haiti_df = haiti_df[haiti_df.num_tokens>1]
    
    return haiti_df


def get_random_message(df: pd.core.frame.DataFrame) -> str:
    '''Chooses a random message from the df, prints out original and english message
    and returns the english message.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    
    Returns:
        english_message (string): selected random message from the dataframe
    '''
    sample = df.sample()
    original_message = sample.iloc[0].original
    english_message = sample.iloc[0].message
    print(colored('Original_message:', 'green'))
    print(original_message)
    print(colored('English_message:', 'blue'))
    print(english_message)
    print()
    return english_message


def interact_with_filters(func: Callable, df: pd.core.frame.DataFrame, **kwargs):
    '''Interactive function for creating a dropdown menu to select filters for the dataset.
    It filters the dataset based on the selection and then runs the given function on it.

    Args:
        func (Callable): A function to be wrapped.
        df (pd.core.frame.DataFrame): The dataset with messages
        **kwargs: additional parameters for the function
    '''
        
    def _select_filter(filters):        
        if filters != 'All':
            df_filtered = df[df[filters]==1]
        else:
            df_filtered = df
        
        if len(df_filtered) == 0:
            print(colored('No samples in the dataset with selected parameters!', 'red'))
        else:
            func(df_filtered, **kwargs)
    
    # Widget for picking the city
    filters_selection = widgets.Dropdown(
        options=['All'] + columns_for_filter,
        description='Data to show',
        choice='All'
    )

    # Putting it all together
    interact(
        _select_filter,
        filters=filters_selection,  
    )


def print_messages(df: pd.core.frame.DataFrame, number_of_messages: int) -> Tuple[List[str], int]:
    '''Chooses a few random messages from the df, prints out the english message and its tokenized version.
    Returns a list of all tokens from the randomly selected messages as well as the number of messages used to create the list.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    
    Returns:
        all_tokens (lisz): list of all tokens from the randomly selected messages
        number_of_messages (int): number of messages used to create the listv
    '''
    if len(df) < number_of_messages:
        number_of_messages = len(df)
    messages = df.sample(number_of_messages)
    all_tokens = []
    for i in range(number_of_messages):
        message = messages.iloc[i].message
        print(colored(f'Message {i+1}:\n', 'green'), message)

        tokens = messages.iloc[i].message_tokens
        all_tokens += tokens
        print(colored(f'Tokens {i+1}:\n', 'green'), tokens)
    
        print()
    return all_tokens, number_of_messages
    
    
def mini_corpus(df: pd.core.frame.DataFrame, corpus_size: int=3):
    ''' Creates a mini corpus from a small number of sentences and prints it
    out for demonstration.
    
    Args:
        df (pd.core.frame.DataFrame): Dataframe with messages
        corpus_size (int): Number of sentences to use for creating the mini corpus
    '''
    all_tokens, corpus_size = print_messages(df, corpus_size)
    
    # Creating dictionary mapping vocabulary words to numerical ids
    mini_corpus = [all_tokens]
    mini_corpus_dictionary = gensim.corpora.Dictionary(mini_corpus)

    # To get only the mapping from token to id, use token2id
    mini_corpus_token2id = mini_corpus_dictionary.token2id

    print(colored(f'Mini dictionary (made of all {corpus_size} messages):\n', 'blue'), mini_corpus_token2id)

    mini_corpus_bow = [mini_corpus_dictionary.doc2bow(doc) for doc in mini_corpus]

    print(colored(f'Bag of Words:\n', 'blue'), mini_corpus_bow)

    mini_corpus_bow_id2word = [[(mini_corpus_dictionary[i], count) for i, count in line] for line in mini_corpus_bow]

    print(colored(f"Human readable BOW:\n", "blue"), mini_corpus_bow_id2word[0], "\n")
    
    
def relative_words_visualization(df: pd.core.frame.DataFrame, n: int, show_other: bool=False):
    ''' Plots a visualization of the relative amount of top words in the corpus with respect to all words.
    
    Args:
        df (pd.core.frame.DataFrame): Dataframe with messages
        n (int): Number of top words to display
        show_other (bool): show also the relative amount of all other words in addition to the top words
    '''
    top_words, total_number_of_words = explore_top_tokens(df, print_top=0)
    words = [[w, f] for w, f in top_words.most_common(n)]
    if show_other:
        all_other_words = total_number_of_words - sum([i[1] for i in words])
        words = [['all_other_words', all_other_words]] + words
    tmp_df = pd.DataFrame(data=words, columns=['words', 'count'])
    
    fig = px.treemap(tmp_df[0:100], path=[px.Constant("Haiti project"), 'words'],
                     values='count',
                     color='count',
                     color_continuous_scale='viridis',
                     color_continuous_midpoint=np.average(tmp_df['count'])
                    )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()


def explore_top_tokens(df: pd.core.frame.DataFrame, print_top: int=20) -> Tuple[List[str], int]:
    ''' Prints out the top most common tokens in the dataframe.
    
    Args:
        df (pd.core.frame.DataFrame): Dataframe with messages
        print_top (int): Number of top words to display
    '''
    all_tokens = [a for sublist in df.message_tokens for a in sublist]
    total_number_of_words = len(all_tokens)
    top_words = Counter(all_tokens)
    if print_top:
        print(colored('Total number of tokens across all messages: ', 'blue'), total_number_of_words)
        for i in top_words.most_common(print_top):
            print(i)
    return top_words, total_number_of_words
  
    
def wordcloud_from_top_words(df: pd.core.frame.DataFrame, n: int):
    ''' Shows a wordcloud of the top most common tokens in the dataframe.
    
    Args:
        df (pd.core.frame.DataFrame): Dataframe with messages
        n (int): Number of top words to display
    '''
    top_words, _ = explore_top_tokens(df, print_top=0)
    wordcloud = WordCloud(width=2000, height=1200, background_color="white")
    wordcloud.generate_from_frequencies(frequencies={w:f for w,f in top_words.most_common(n)})
    plt.figure(figsize=(13,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def histogram_number_of_words(df: pd.core.frame.DataFrame):
    ''' Plots a histogram of the number of messages with given number of words or tokens.
    
    Args:
        df (pd.core.frame.DataFrame): Dataframe with messages
    '''
    num_words = df.num_words
    num_tokens = df.num_tokens

    bins = np.linspace(0, 100, 100)
    
    plt.figure(figsize=(12,6))
    plt.hist(num_words, bins, alpha=0.5, label='Number of words')
    plt.hist(num_tokens, bins, alpha=0.5, label='Number of tokens')
    plt.title(f'Number of messages with given number of words/tokens', fontsize=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Number of words/tokens in a message', fontsize=16)
    plt.ylabel('Count of messages', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()



def pos_tag_convert(nltk_tag: str) -> str:
    '''Converts nltk tags to tags that are understandable by the lemmatizer.
    
    Args:
        nltk_tag (str): nltk tag
        
    Returns:
        _ (str): converted tag
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN
