import warnings
warnings.filterwarnings('ignore')
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import ipywidgets as widgets
from ipywidgets import interact
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from termcolor import colored
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from typing import Callable, Tuple, List

FONT_SIZE_TITLE = 22
FONT_SIZE_AXES = 18
FONT_SIZE_TICKS = 14

columns_for_filter = [
    'medical_help', 'medical_products', 'search_and_rescue', 'security',
    'military', 'child_alone', 'water', 'food', 'shelter', 'clothing',
    'money', 'missing_people', 'refugees', 'death', 'other_aid',
    'infrastructure_related', 'transport', 'buildings', 'electricity',
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
    'other_weather'
]

# Defining some useful globals
STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(["n't", "'s", "'m", "..", "'d", "'c", "u", "'re", "ca", "'ve", "'ll", 'n', "oh", "ha", "u"])
punctuation = string.punctuation + '``'+ "''" 

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


def exclude_words_from_dictionary(dictionary, words_to_exclude):
    '''Exclude a set of words from a given dictionary

    Args: 
        dictionary (dict): An input dictionary
        words_to_exclude (list): A list of words to exclude
    Returns:
        (dict): A new dictionary 
    '''
    tokens_to_exclude = []
    for word in words_to_exclude:
        try:
            tokens_to_exclude.append(dictionary.token2id[word])
        except:
            print(f'"{word}" was already excluded')
    dictionary.filter_tokens(bad_ids=tokens_to_exclude)
    
    return dictionary


def compute_coherence_score(
    topic_words, dictionary, texts, corpus
):
    """Computes coherence score for a list of words.

    Args:
        topic_words (list[str]): List of words in a topic.
        dictionary (gensim.corpora.Dictionary, optional): The gensim dictionary. Defaults to corpus_dictionary.
        texts (numpy.ndarray, optional): Text in the corpus encoded as lists in a numpy array. Defaults to corpus.values.
        corpus (list[int], optional): Bag of words representation of corpus. Defaults to corpus_bow.

    Returns:
        float: The u_mass coherence score metric.
    """

    # Let the CoherenceModel do the necessary processing to compute the score
    coherence_model = gensim.models.CoherenceModel(
        topics=topic_words,
        dictionary=dictionary,
        texts=texts,
        corpus=corpus,
        coherence="u_mass",
    )

    # Save the coherence score
    coherence_score = coherence_model.get_coherence()

    return coherence_score


def get_top_words_lda(n_comp, lda_model, num_words=20):
    # Get words and importance for each topic
    lda_topics = lda_model.show_topics(num_topics=n_comp, num_words=num_words, formatted=False)
    
    all_top_words = [[word for word, importance in topic[1]] for topic in lda_topics]
    
    return all_top_words


def plot_top_words_lda(n_comp, lda_model, num_words=20):

    # Get words and importance for each topic
    lda_topics = lda_model.show_topics(num_topics=n_comp, num_words=num_words, formatted=False)
    rows = ((n_comp - 1) // 3) + 1
    cols = min(n_comp, 3) 
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    axes = axes.flatten()
    for num_topic, topic in enumerate(lda_topics):

        top_words = [word for word, importance in topic[1]]
        importance = [importance for word, importance in topic[1]]

        ax = axes[num_topic]
        ax.barh(top_words, importance)
        ax.invert_yaxis()
        ax.tick_params(labelsize=12)
        ax.set_title(f"Topic {num_topic + 1}", fontsize=18)
        ax.set_xlabel("Word importance", fontsize=16)
    for ax in axes.flat[n_comp:]:
        ax.remove()
    plt.subplots_adjust(top=0.9, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.suptitle("LDA", size=20, y=1.03)
    plt.show()


def plot_coherences_lda(n_topics_range, coherence_scores_lda):

    plt.figure(figsize=(8, 5))

    plt.plot(
        n_topics_range,
        coherence_scores_lda,
        color="green",
        marker="o",
        linestyle="dashed",
        linewidth=2,
        markersize=12,
    )
    plt.xticks(n_topics_range)
    # Set the y-axis limits
    ymax = max(coherence_scores_lda) + 0.1
    ymin = min(coherence_scores_lda) - 0.1
    plt.ylim((ymin, ymax))

    plt.title("Coherence Score per Number of Topics", size=18)
    plt.xlabel("Number of Topics", size=16)
    plt.ylabel("Coherence Score", size=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(["Coherence Score"])

    plt.show()

    
@dataclass
class LDAModel:
    num_topics: int
    model: gensim.models.ldamodel.LdaModel
    top_words: List[List[str]]
    coherence_score: float
        
        
def train_lda_model(num_topics, corpus_bow, corpus, corpus_dictionary):
    lda_model = gensim.models.LdaModel(
        corpus=corpus_bow,
        id2word=corpus_dictionary,
        num_topics=num_topics,
        passes=3,
        random_state=123
    )
    
    shown_topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)

    top_words = [[word[0] for word in topic[1]] for topic in shown_topics]
    
    coherence_score = compute_coherence_score(top_words, corpus_dictionary, corpus.values, corpus_bow)
    
    return LDAModel(num_topics, lda_model, top_words, coherence_score)


def top_words_plot(models, max_number_of_topics):

    def _plot(n_topic):
        m = models[n_topic]
        print(f"Coherence score for {n_topic} topics: {m.coherence_score:.2f}")
        plot_top_words_lda(n_topic, m.model)


    n_topic_selection = widgets.Dropdown(
        options=[*range(2, max_number_of_topics + 1)],
        value=2,
        description="# of topics"
    )

    interact(_plot, n_topic=n_topic_selection)
    

def plot_topic_importance(corpus_bow, data, lda_3_topics, num_topics, n_days=90):
    topics_on_messages = lda_3_topics.model.get_document_topics(corpus_bow, minimum_probability=0.0)

    result = []
    start_day =  pd.to_datetime("2010-01-17")
    for delta in range(n_days):
        sum_by_topic = [0] * num_topics
        today = start_day + pd.Timedelta(days=delta)
        tomorrow = start_day + pd.Timedelta(days=delta+1)
        index_range = data.loc[(data['date_haiti'] >= today) & (data['date_haiti'] < tomorrow)].index
        for k in index_range:
            for i in range(num_topics):
                sum_by_topic[i] += topics_on_messages[k][i][1]
        sum_by_topic = np.array(sum_by_topic)
        sum_by_topic = sum_by_topic / np.sum(sum_by_topic)

        result.append(sum_by_topic)
    result = np.array(result)

    result3 = []
    days_per_tick = 7

    for i in range(n_days):
        for topic in range(num_topics):
            rows = result[i, topic]
            result3.append([int(i / days_per_tick), np.sum(rows), topic])

    result3 = np.array(result3)

    plt.figure(figsize=(15, 5))

    style = [f"Topic {int(x) + 1}" for x in result3[:,2]]
    ax = sns.lineplot(x=result3[:,0]*days_per_tick, y=result3[:,1], hue=style, style=style)
    ax.set(xlabel='Days', ylabel='Topic importance')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.set_title(f"Topic importance over time", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Days", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Topic importance", fontsize=FONT_SIZE_AXES)
    ax.legend(fontsize=FONT_SIZE_TICKS)

    plt.show()
    
    
def plot_message_classification(topics_on_message):
    height = list(map(lambda x: x[1], topics_on_message[0]))
    topics = [f"Topic {i + 1}" for i, _ in enumerate(height)]
    plt.figure(figsize=(10,6))
    plt.bar(topics, height)
    plt.title('Topic importance for a single message', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Topics', fontsize=FONT_SIZE_AXES)
    plt.ylabel('Importance', fontsize=FONT_SIZE_AXES)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)

    
def plot_random_message_classification(df, lda_topics, corpus_dictionary):
    new_message = df.sample(1).message.values[0]
    new_message_tokens = [df.sample(1).message_tokens.values[0]]
    message_bow = [corpus_dictionary.doc2bow(doc) for doc in new_message_tokens]
    topics_on_message = lda_topics.model.get_document_topics(message_bow, minimum_probability=0.0)
    print(new_message)

    plot_message_classification(topics_on_message)  
    
    
def interact_with_filters(function, df, **kwargs):
        
    def _select_filter(filters):        
        if filters != 'All':
            df_filtered = df[df[filters]==1]
        else:
            df_filtered = df
        
        if len(df_filtered) == 0:
            print(colored('No samples in the dataset with selected parameters!', 'red'))
        else:
            function(df_filtered, **kwargs)
    
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
