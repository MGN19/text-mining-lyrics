import pandas as pd 
import numpy as np 
import re 
import unicodedata
import contractions
import langid
import unidecode


from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords 
from textblob import TextBlob

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Doc2Vec


lemmatizer = WordNetLemmatizer()
sent_tokenizer = PunktSentenceTokenizer()
stop_words = set(stopwords.words('english'))
stop_words.discard('only')


def extract_and_remove_square_brackets(data, column_name, show): 
    
    # Extract content between square brackets 
    extracted_content = data[column_name].str.extract(r'\[(.*?)\]')
    
    # Combine all extracted content into a single list
    content_list = extracted_content.stack().tolist()

    # Find unique strings in the list
    unique_strings = set(content_list)
    
    if show:
        print("Unique strings in the extracted content:")
        print(unique_strings)

    # Remove square brackets and text within
    data[column_name] = data[column_name].map(lambda s: re.sub(r'\[.*?\]', '', s))
    
def remove_url(data, column_name):
    data[column_name] = data[column_name].map(lambda s: re.sub(r'\bhttp[^\s]+\b|\bwww\.[^\s]+\b', '', s))
    
def apostrophe_substitution(text):
    # Substitute various apostrophe forms with a standard single quote
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('‛', "'")
    text = text.replace('‚', "'")
    text = text.replace('´', "'")
    return text

def custom_expand_contractions(text):    
    # Handle "I ain't" as "I am not"
    text = re.sub(r"i ain't", "i am not", text, flags=re.IGNORECASE)
    text = re.sub(r"you ain't", "you are not", text, flags=re.IGNORECASE)
    text = re.sub(r"he ain't", "he is not", text, flags=re.IGNORECASE)
    text = re.sub(r"she ain't", "she is not", text, flags=re.IGNORECASE)
    text = re.sub(r"it ain't", "it is not", text, flags=re.IGNORECASE)
    text = re.sub(r"we ain't", "we are not", text, flags=re.IGNORECASE)
    text = re.sub(r"they ain't", "they are not", text, flags=re.IGNORECASE)
    text = re.sub(r"nothing's", "nothing is", text, flags=re.IGNORECASE)
    
    # Use contractions library for other contractions
    text = contractions.fix(text)
    
    return text

def stopword_remover(tokenized_comment): 
    clean_comment = [] 
    
    for token in tokenized_comment: 
        if token not in stop_words: clean_comment.append(token) 
    return clean_comment


# def translate_sentences(sentences, target_language='en'):
#     translator = Translator()
#     translated_sentences = []

#     # Check if sentences is a list or a string
#     if isinstance(sentences, list):
#         for sentence in sentences:
#             translated = translator.translate(sentence, dest=target_language)
#             translated_sentences.append(translated.text)
#     elif isinstance(sentences, str):
#         translated = translator.translate(sentences, dest=target_language)
#         translated_sentences.append(translated.text)

#     # Join the translated sentences into a single string without spaces between them
#     translated_text = ''.join(translated_sentences)

#     return translated_text


def preprocessor(raw_text, 
                 lowercase=True, 
                 leave_punctuation = False, 
                 remove_stopwords = True,
                 correct_spelling = False, 
                 lemmatization=False, 
                 tokenized_output=False, 
                 sentence_output=False
                 ):
    
    # Check if raw_text is a string
    if not isinstance(raw_text, str):
        # Skip preprocessing for non-string values
        return raw_text
    
    # Normalize Unicode characters - italic
    raw_text = unicodedata.normalize('NFKD', raw_text).encode('ascii', 'ignore').decode('utf-8')

    # Translate
    # raw_text = translate_sentences(sent_tokenizer.tokenize(raw_text), target_language='en')

    if lowercase == True:
        #convert to lowercase
        clean_text = raw_text.lower()
    else:
        clean_text = raw_text   
    
    #remove newline characters
    clean_text = re.sub(r'(\n|\r|\t|</ul>)', ' ', clean_text)

    # Expand contractions
    clean_text = custom_expand_contractions(clean_text)

    if leave_punctuation == False:
        #remove punctuation:
        clean_text = re.sub(r'(\W)', ' ', clean_text)

    #remove isolated consonants:
    clean_text = re.sub(r'\b([^aeiou])\b',' ',clean_text, flags=re.IGNORECASE)

    #correct spelling
    if correct_spelling == True:
        
        incorrect_text = TextBlob(clean_text)
        clean_text = incorrect_text.correct()

    #tokenize
    clean_text = word_tokenize(str(clean_text))

    #remove stopwords
    if remove_stopwords == True:
        clean_text = stopword_remover(clean_text)
        # Remove unecessary words
        additional_stopwords = ["intro", "verse", 'pre-chorus', "chorus", "hook", "refrain", "x"]
        clean_text = [token for token in clean_text if not any(pattern in token.lower() for pattern in additional_stopwords)]


    #lemmatize
    if lemmatization == True:
        for pos_tag in ["v","n","a"]:
            clean_text = [lemmatizer.lemmatize(token, pos=pos_tag) for token in clean_text]

    if remove_stopwords == True:
        # Remove most common words
        # common_words = ['got', 'life', 'love', 'oh', 'la', 'see', 'yeah', 'girl',
        #                 'know', 'say', 'one', 'baby', 'go', 'think',
        #                 'u', 'come', 'need', 'make']
        common_words = ['one', 'see', 'go', 'know']
        clean_text = [token for token in clean_text if token.lower() not in common_words]

    if tokenized_output == False:
        
        #re-join
        clean_text = " ".join(clean_text)
        
        #Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)

    if sentence_output == True:
        #split into sentences:
        clean_text = sent_tokenizer.tokenize(str(clean_text))

    return clean_text

def fold_score_calculator(y_pred, y_test):
    
    # Compute classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    return (acc, prec, recall, f1)

# Function to infer vectors for documents
def infer_vectors(model, docs):
    vectors = [model.infer_vector(doc.split()) for doc in docs]
    return vectors