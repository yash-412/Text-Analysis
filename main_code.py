import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from collections import defaultdict
import string
from nltk.tokenize import RegexpTokenizer


def scrape_and_analyze(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the article title
    title_element = soup.find("h1")
    title = title_element.text.strip() if title_element else ""

    # Extract the article text
    article_text = ""
    article_elements = soup.find_all("p")
    for element in article_elements:
        if "entry-title" in element.get("class", []):
            continue
        if "tdm-descr" in element.get("class", []):
            continue
        article_text += element.text.strip() + " "

    return title, article_text


def remove_stop_words(text, stop_words_df):
    # Add a space between both sides of punctuations and strings because stopwords were not considered due to punctuations attached with string without space
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    words = text.split()
    # Remove stop words from the DataFrame
    words = [word for word in words if word.lower() not in stop_words_df["word"].values]
    # Join the remaining words back into a string
    processed_text = " ".join(words)
    return processed_text


def remove_punctuation(text):
    # Add a space between both sides of punctuations and strings because stopwords were not considered due to punctuations attached with string without space
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    """text_no_punctuations = re.sub(r"\'s\b", '', text_no_punctuations)"""
    text_no_punctuations = re.sub(r'[^\w\s\"]', ' ', text)
    text_no_punctuations = re.sub(r'(?<=\w)\s+(?=\w)', ' ', text_no_punctuations)
    return text_no_punctuations


def count_syllables(word, d):
    if word.lower() in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    else:
        return [max(1, len(re.findall(r'[aeiouy]+', word.lower())))]

        
def calculate_average_sentence_length(text):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    tokens = tokenizer.tokenize(text)
    total_words = len(tokens)
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    average_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    return average_sentence_length


def calculate_percentage_complex_words(text, d):
    words = nltk.word_tokenize(text)
    complex_words = [word for word in words if max(count_syllables(word, d)) > 2]
    percentage_complex_words = len(complex_words) / len(words) * 100 if len(words) > 0 else 0
    return percentage_complex_words


def calculate_average_word_length(text):
    words = text.split()
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    average_length = total_characters / total_words if total_words > 0 else 0
    return average_length


def calculate_average_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    words = nltk.word_tokenize(text)
    total_words = len(words)
    average_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    return average_words_per_sentence


def calculate_fog_index(text, d):
    average_sentence_length = calculate_average_words_per_sentence(text)
    percentage_complex_words = calculate_percentage_complex_words(text, d)
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
    return fog_index


def count_personal_pronouns(text):
    pronouns = ["I", "we", "my", "ours", "us"]
    pronouns_pattern = r"\b(?:{})\b".format("|".join(pronouns))

    # Find all occurrences of personal pronouns in the text
    a_pronouns_count = len(re.findall(pronouns_pattern, text, flags=re.IGNORECASE))
    return a_pronouns_count


def count_words(text):
    word_count = len(text)
    return word_count


def calculate_syllables_per_word(paragraph, d):
    words = re.findall(r'\b\w+\b', paragraph)  # Tokenize the paragraph into words
    syllable_counts = [count_syllables(word, d) for word in words]  # Calculate syllable count for each word
    flattened_counts = [count for sublist in syllable_counts for count in sublist]  # Flatten the list of lists
    syllable_count = sum(flattened_counts)  # Sum up the syllable counts
    return syllable_count

"""positive_words = [word for sublist in positive_words for word in sublist]
negative_words = [word for sublist in negative_words for word in sublist]"""

def calculate_sentiment_scores(text, positive_words, negative_words):
    # Create an instance of RegexpTokenizer with a pattern to match alphabetical words
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    # Tokenize the text and extract alphabetical words as tokens
    tokens = tokenizer.tokenize(text)
    # Calculate Positive Score
    positive_score = sum(1 for word in tokens if word.lower() in positive_words)
    # Calculate Negative Score
    negative_score = sum(1 for word in tokens if word.lower() in negative_words)
    # Calculate Polarity Score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    # Calculate Subjectivity Score
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return polarity_score, subjectivity_score

def calculate_complex_word_count(text, d):
    complex_word_count = 0
    syllable_counts = []
    lowercase_filtered_words = [word.lower() for word in text]

    for word in lowercase_filtered_words:
        syllables = count_syllables(word, d)
        syllable_counts.append(syllables)

    complex_word_count = sum(1 for syllables in syllable_counts if max(syllables) > 2)

    return complex_word_count 

def calculate_average_syllables_per_word(text, d):
    syllable_counts = []
    lowercase_filtered_words = [word.lower() for word in text]

    for word in lowercase_filtered_words:
        syllables = count_syllables(word, d)
        syllable_counts.append(syllables)

    total_words = len(lowercase_filtered_words)
    total_syllables = sum(sum(syllables) for syllables in syllable_counts)

    average_syllables_per_word = total_syllables / total_words if total_words > 0 else 0

    return average_syllables_per_word




# Specify the path for the input and output Excel files
input_file = 'D:\\VSCodium\\Text Analysis\\Text-Analysis\\Input.xlsx' #URLs should be given in a single coulumn (Preferable Column B, SI No in Column A)
output_file = 'D:\\VSCodium\\Text Analysis\\Text-Analysis\\Output Data.xlsx'

# Read the input data from Excel into a DataFrame
data = pd.read_excel(input_file)

# Combine nltk stopwords with given stopwords
directory = "D:\\VSCodium\\Text Analysis\\Text-Analysis\\StopWords"
combined_stopwords_file = "combined_stopwords.txt"
combined_stopwords_path = os.path.join(directory, combined_stopwords_file)

with open(combined_stopwords_path, "r", encoding='utf-8') as file:
    combined_stopwords = file.read().splitlines()

nltk_stopwords = set(stopwords.words("english"))

combined_stopwords_set = set(combined_stopwords) | nltk_stopwords

combined_stopwords_df = pd.DataFrame(list(combined_stopwords_set), columns=["word"])

combined_stopwords_csv = "combined_stopwords.csv"
combined_stopwords_df.to_csv(combined_stopwords_csv, index=False)

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

# Create positive and negative dictionary
# Read the positive words file
positive_words_file = "D:\\VSCodium\\Text Analysis\\Text-Analysis\\MasterDictionary/positive-words.txt"
positive_words_df = pd.read_csv(positive_words_file, comment=";", header=None, encoding='utf-8', names=["word"])
positive_words = set(positive_words_df["word"])
# Read the negative words file
negative_words_file = "D:\\VSCodium\\Text Analysis\\Text-Analysis\\MasterDictionary/negative-words.txt"
negative_words_df = pd.read_csv(negative_words_file, comment=";", header=None, encoding='cp1252', names=["word"])
negative_words = set(negative_words_df["word"])


# Iterate over each URL in column B (starting from row 2)
for index, url in data['URL'].items():
    # Define functions for web scraping and analysis
    def scrape_and_analyze(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the article title
        title_element = soup.find("h1")
        title = title_element.text.strip() if title_element else ""

        # Extract the article text
        article_text = ""
        article_elements = soup.find_all("p")
        for element in article_elements:
            if "entry-title" in element.get("class", []):
                continue
            if "tdm-descr" in element.get("class", []):
                continue
            article_text += element.text.strip() + " "

        return title, article_text

    def remove_punctuation(text):
        # Add a space between both sides of punctuations and strings because stopwords were not considered due to punctuations attached with string without space
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        """text_no_punctuations = re.sub(r"\'s\b", '', text_no_punctuations)"""
        text_no_punctuations = re.sub(r'[^\w\s\"]', ' ', text)
        text_no_punctuations = re.sub(r'(?<=\w)\s+(?=\w)', ' ', text_no_punctuations)
        return text_no_punctuations

    def remove_stop_words(text, stop_words_df):
        # Add a space between both sides of punctuations and strings because stopwords were not considered due to punctuations attached with string without space
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        words = text.split()
        # Remove stop words from the DataFrame
        words = [word for word in words if word.lower() not in stop_words_df["word"].values]
        # Join the remaining words back into a string
        processed_text = " ".join(words)
        return processed_text

    # Scrape and analyze the article
    title, article_text = scrape_and_analyze(url)

    # Remove stopwords and punctuation
    processed_text = remove_stop_words(article_text, combined_stopwords_df)
    processed_text_no_punctuations = remove_punctuation(processed_text)

    article_text_no_punct = remove_punctuation(article_text)
    # Create an instance of RegexpTokenizer with a pattern to match alphabetical words

    tokenizer = RegexpTokenizer(r'[A-Za-z]+')

    # Tokenize the text and extract alphabetical words as tokens
    tokens = tokenizer.tokenize(processed_text_no_punctuations)

    # Calculate Positive Score
    positive_score = sum(1 for word in tokens if word.lower() in positive_words)

    # Calculate Negative Score
    negative_score = sum(-1 for word in tokens if word.lower() in negative_words)

    # Calculate the metrics
    average_sentence_length = calculate_average_sentence_length(article_text)
    percentage_complex_words = calculate_percentage_complex_words(processed_text_no_punctuations, d)
    average_word_length = calculate_average_word_length(processed_text_no_punctuations)
    average_words_per_sentence = calculate_average_words_per_sentence(article_text)
    fog_index = calculate_fog_index(article_text, d)
    pronouns_count = count_personal_pronouns(article_text_no_punct)
    word_count = count_words(processed_text_no_punctuations)
    syllable_count = calculate_average_syllables_per_word(article_text_no_punct, d)
    polarity_score, subjectivity_score = calculate_sentiment_scores(processed_text_no_punctuations, positive_words, negative_words)
    complex_word_count = calculate_complex_word_count(processed_text_no_punctuations, d)

    # Store the calculated metrics in the DataFrame using .loc
    data.loc[index, 'POSITIVE SCORE'] = positive_score
    data.loc[index, 'NEGATIVE SCORE'] = negative_score
    data.loc[index, 'POLARITY SCORE'] = polarity_score
    data.loc[index, 'SUBJECTIVITY SCORE'] = subjectivity_score
    data.loc[index, 'AVG SENTENCE LENGTH'] = average_sentence_length
    data.loc[index, 'PERCENTAGE OF COMPLEX WORDS'] = percentage_complex_words
    data.loc[index, 'FOG INDEX'] = fog_index
    data.loc[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = average_words_per_sentence
    data.loc[index, 'COMPLEX WORD COUNT'] = complex_word_count
    data.loc[index, 'WORD COUNT'] = word_count
    data.loc[index, 'SYLLABLE PER WORD'] = syllable_count
    data.loc[index, 'PERSONAL PRONOUNS'] = pronouns_count
    data.loc[index, 'AVG WORD LENGTH'] = average_word_length

    # Save the updated data to a new Excel file
    data.to_excel(output_file, index=False)