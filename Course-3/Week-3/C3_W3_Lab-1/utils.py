import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from termcolor import colored
import pandas as pd
from typing import Callable


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

kreyol_stopwords = [
    "nou", "tankou", "la", "soti", "koupe", "kijan", "desann", "sou", "non",
    "tou", "ni", "kòm", "deyò", "pou", "sa", "gen", "si", "oswa", "moute",
    "men", "dwe", "m", "kote", "ta", "ka", "konsa", "trè", "anvan", "yon",
    "chak", "yo", "a,", "atravè", "pase", "ba", "sèlman", "de", "nan", "ap",
    "ankò", "anwo", "ant", "pi", "kounye", "t", "ou", "apre", "y", "a", "kèk",
    "don", "isit", "li", "pral", "menm", "jis", "lè", "moun", "mwen", "plis",
    "l", "anba", "se", "kapab", "fè", "nenpòt", "te", "ke", "poukisa", "ak",
    "kont", "paske", "lòt", "pwòp", "jouk", "ki", "tout", "pandan", "pa"
]

def messages_distribution(df: pd.core.frame.DataFrame):
    '''Plots the distribution of messages over the "related" parameter.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.countplot(x ='related', data = df, ax = ax, palette=['#129412',"#E80D05",'#FFD343'])

    ax.set_xticklabels(['0 = no', '1 = yes', '2 = unsure'], rotation=0)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.set_xlabel('Related', fontsize=FONT_SIZE_AXES)
    ax.set_ylabel('Count', fontsize=FONT_SIZE_AXES)

    

def offer_request(df: pd.core.frame.DataFrame):
    '''Plots a histogram of messages whether they are offers, requests or none.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    '''
    # Count the number of offers and requests
    data = df[['offer', 'request']].sum()
    # Find the ones which are neither offers nor requests
    none = df[(df['offer'] == 0) & (df['request'] == 0)]
    data['none'] = len(none)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    ax = data.plot.bar()
    ax.set_title("Messages requesting or offering help", fontsize=FONT_SIZE_TITLE)
    ax.set_xticklabels(['offer', 'request', 'none'], rotation=0)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.set_ylabel('Count', fontsize=FONT_SIZE_AXES)


def daily_plot(df: pd.core.frame.DataFrame):
    '''Plots the distribution of messages per day.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    '''
    plt.figure(figsize=(10, 6))
    df.groupby('date_haiti')['id'].count().plot()
    plt.title('Number of Messages per Day', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Date', fontsize=FONT_SIZE_AXES)
    plt.ylabel('Message count', fontsize=FONT_SIZE_AXES)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    

def monthly_histogram(df: pd.core.frame.DataFrame):
    '''Plots the distribution of messages per month.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    '''
    months = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    df['month'] = [a.month for a in df['date_haiti']]
    grouped = df.groupby('month')['id'].count()
    months_for_messages = [months[i] for i in grouped.index]
    plt.figure(figsize=(10, 6))
    plt.bar(months_for_messages, grouped)
    plt.title('Number of Messages per Month', fontsize=FONT_SIZE_TITLE)
    plt.ylabel('Number of messages', fontsize=FONT_SIZE_AXES)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.show()


def plot_wordclouds(df: pd.core.frame.DataFrame, remove_stopwords: bool=False):
    '''Plots the wordclouds of words that appear in the messages.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
        use_stopwords (bool): A flag wether to use stopwords or not
    '''
    
    def _create_wordcloud(text, stopwords_list):
        # Create the wordcloud
        wordcloud = WordCloud(
            width=2000, height=1200,
            background_color="white",
            stopwords=stopwords_list
        ).generate(text)
        
        # Plot the WordCloud
        plt.figure(figsize=(13,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    
    # Empty list of stopwords to pass to the wordcloud function
    stopwords_list = []
    
    print("Wordcloud of messages translated to English:")
    # Creating the text variable for English wordcloud
    text = " ".join(title for title in df.message) 
    # None gives the default list of english stopwords for the wordcloud
    if remove_stopwords:
        stopwords_list = None
    # Create and plot the wordcloud
    _create_wordcloud(text, stopwords_list)
    
    print("Wordcloud of original messages:")
    # Creating the text variable for Kreyol wordcloud
    text = " ".join(title for title in df.original) 
    # For Kreyol, we need a custom list of stopwords
    if remove_stopwords:
        stopwords_list = kreyol_stopwords
    # Create and plot the wordcloud
    _create_wordcloud(text, stopwords_list)

    
def show_messages(df: pd.core.frame.DataFrame):
    '''Prints 5 messages given by the filters chosen by user using a dropdown menu.

    Args:
        df (pd.core.frame.DataFrame): The dataset with messages
    '''
    related_dict = {'No': 0, 'Yes': 1, 'Unsure': 2}
    
    def _select_filter(filters, related):

        if filters != 'All':
            df_filtered = df[df[filters]==1]
        else:
            df_filtered = df
        
        df_filtered = df_filtered[df_filtered.related == related_dict[related]]
        df_filtered = df_filtered[['message', 'original']]
        if df_filtered.empty:
            print(colored('No samples in the dataset with selected parameters!', 'red'))
        else:
            sample = df_filtered.sample(5)
            for index, row in enumerate(sample.iterrows()):
                print(colored(f'Message {index + 1}:', 'blue'))
                print(f'Original:\n    {row[1][1]}')
                print(f'English:\n    {row[1][0]}')
                print()
    
    # Widget for selecting a filter from the dataset
    filters_selection = widgets.Dropdown(
        options=['All'] + columns_for_filter,
        description='Data to show',
        choice='All'
    )
    
    # Widget for selecting whether the messages are related to disasters
    related_selection = widgets.Dropdown(
        options=['Yes', 'No', 'Unsure'],
        description='Related',
        choice='Yes'
    )

    # Putting it all together
    interact(
        _select_filter,
        filters=filters_selection,
        related=related_selection,
    )


def interact_with_filters(function: Callable, df: pd.core.frame.DataFrame, **kwargs):
    '''Interactive function for creating a dropdown menu to select filters for the dataset.

    Args:
        function (Callable): A function to be wrapped.
        df (pd.core.frame.DataFrame): The dataset with messages
        **kwargs: additional parameters for the function
    '''
    
    def _select_filter(filters: str):        
        if filters != 'All':
            df_filtered = df[df[filters]==1]
        else:
            df_filtered = df
        
        if len(df_filtered) == 0:
            print(colored('No samples in the dataset with selected parameters!', 'red'))
        else:
            function(df_filtered, **kwargs)
    
    # Widget for selecting a filter from the dataset
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
