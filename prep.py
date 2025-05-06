from flask import Flask, request, jsonify
import pandas as pd
import re
import csv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

factory = StopWordRemoverFactory()
default_stopwords = factory.get_stop_words()

custom_stopwords = ['sih', 'nya', 'yg']
all_stopwords = default_stopwords + custom_stopwords
stopword_dictionary = ArrayDictionary(all_stopwords)
stopword_remover = StopWordRemover(stopword_dictionary)

def CaseFoldingText(text):
    if isinstance(text, str):
        text = text.casefold()
    return text

def cleaning_text(text):
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  
    text = re.sub(r'#[A-Za-z0-9]+', ' ', text)  
    text = re.sub(r"http\S+", ' ', text)  
    text = re.sub(r'[0-9]+', ' ', text) 
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'(.)\1{2,}', r'\1', text)  
    text = text.replace('\n', ' ') 
    text = text.strip()  
    return text

def load_slangwords(file_path):
    slangwords = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                slang = row[0].strip()  
                correct = row[1].strip()  
                slangwords[slang] = correct
    return slangwords

def fix_slangwords(text, slangwords):
    words = text.split()
    fixed_words = []

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text

def tokenizingText(text):
    tokens = text.split() 
    return tokens

def replace_text(text):
    text = [word.replace('awat', 'rawat') for word in text]
    return text
    
def remove_stopwords1(text):
    text = str(text)
    if isinstance(text, str):
        clean_text = stopword.remove(text)
        return clean_text
    else:
        return text

def remove_stopwords(text):
    text = str(text)
    if isinstance(text, str):
        return stopword_remover.remove(text)
    return text

avoid_stemming_words = ['rawat']

def lemmatize_text(text):
    text = str(text)
    if isinstance(text, str):
        words = text.split()
        
        stemmed_words = []
        for word in words:
            if 'rawat' in word:
                stemmed_words.append('rawat') 
            else:
                stemmed_words.append(stemmer.stem(word))

        return ' '.join(stemmed_words)
    return text