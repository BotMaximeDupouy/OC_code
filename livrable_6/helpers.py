import string
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

class CleaningText:
    def __init__(self, row):
        self.row = row

    def remove_upper_case(row:str)->str:
        '''take a sentence, put char in lower and return sentence'''
        return row.lower()

    def remove_punctuation(row:str)->str:
        '''take a sentence, remove punctuation and return sentence'''
        for punctuation in string.punctuation:
            row= row.replace(punctuation, ' ').strip(' ')
        return row

    def remove_number(row:str)->str:
        '''take a sentencde and return sentence without digit'''
        return ''.join([char for char in row if char.isdigit()==False])

    def remove_stop_words(row:str)->list:
        '''take a sentence, remove stop words and return a tokenise sentence'''
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(row)
        row = [w for w in word_tokens if not w in stop_words]
        return row

    def stemming(row:list)->list:
        '''take eeach word, reduce it to its word stem that affixes to suffixes
        and prefixes, return list tokens'''
        stemmer = nltk.stem.porter.PorterStemmer()
        row_stem = [stemmer.stem(word) for word in row]
        return row_stem

    def lemmatizing(row:list) -> list:
        '''take each word, switch it to its base root mode
        and return list tokens'''
        lemma = nltk.wordnet.WordNetLemmatizer()
        row_lemmatized = [lemma.lemmatize(word.strip()) for word in row]
        return row_lemmatized

    def remove_encode_char(row:str) -> str:
        '''take a string and return string without \n\t\r'''
        row = row.splitlines()
        row = [word.strip() for word in row]
        row = ' '.join(row)
        row = re.sub(r'\t', '', row)
        return row
