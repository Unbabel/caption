import pandas as pd
import re
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
import os


# Create a reference variable for Class WordPunctTokenizer
tk = WordPunctTokenizer()

def get_word_punctuation_pairs_of_sentence(input_sentence: str) -> list:
    if type(input_sentence) == str:
        if len(re.sub(r'[^\w\s]', '', input_sentence)) > 0:
            sentence = tk.tokenize(input_sentence) # Split sentence in words and punctuation
            words = []
            punctuation = []
            for word_index, word in enumerate(sentence):
                if not word.isalnum(): # If it's not a word, we don't want to tag it
                    continue
                if word_index + 1 < len(sentence):
                    words.append(word)
                    if sentence[word_index + 1].isalnum(): # If the next word is an actual word, then the tag for the current word is 0
                        punctuation.append(0)
                    else:
                        punctuation.append(sentence[word_index + 1]) # If the next word is punctuation, then the tag for the current word is the punctuation
                else:
                    if word.isalnum(): #If the sentence doesn't end in punctuation the last segment will be a word
                        words.append(word)
                        punctuation.append(0)

            return (words, punctuation)
        else:
            return (None, None)
    else:
        return (None, None)

def get_word_casing_pairs_of_sentence(input_sentence: str) -> list:
    if type(input_sentence) == str:
        if len(re.sub(r'[^\w\s]', '', input_sentence)) > 0:
            sentence = tk.tokenize(input_sentence) # Split sentence in words and punctuation
            capitalisation = []
            for word_index, word in enumerate(sentence):
                if word.isalnum(): # If it's a word
                    if word.istitle():
                        capitalisation.append('T')
                    elif word.isupper():
                        capitalisation.append('U')
                    else:
                        capitalisation.append('L')

            return capitalisation
        else:
            return None
    else:
        return None


def print_word_punctuation_and_capitalisation_pair_to_file(punctuation_pairs: list, capitalisation_pairs:list, mqm_100_sentences: int, folder: str):
    f = open(f"processed_data/{folder}/{mqm_100_sentences}.tsv", "w")
    for pair_index, punctuation_pair in enumerate(punctuation_pairs):
        f.write(f"{punctuation_pair[0].lower()}\t{capitalisation_pairs[pair_index][1]}\t{punctuation_pair[1]}\n") 
    f.close()

df = pd.read_csv('data/train.csv')
pd.set_option('display.max_columns', None)
df.fillna('', inplace=True)
df = df.sample(frac = 1) # Random shuffle of lines


mqm_100_source_sentences = 0
mqm_100_target_sentences = 0

number_of_perfect_rows = df[df['score'] == 1].shape[0]

source_data_train = []
source_data_dev = []
mt_data_train = {}
mt_data_dev = {}

for row_index, row in tqdm(
        df[df['score'] == 1].iterrows(),
        total=number_of_perfect_rows
    ):
    # Process and save source sentences
    words, punctuation = get_word_punctuation_pairs_of_sentence(row['src'])
    capitalisation = get_word_casing_pairs_of_sentence(row['src'])
    if (words is not None) and (punctuation is not None) and (capitalisation is not None):
        source_data_train.append([words, capitalisation, punctuation]) if (mqm_100_source_sentences < number_of_perfect_rows * 0.9) else source_data_dev.append([words, capitalisation, punctuation])
        mqm_100_source_sentences += 1
    
    # Process and save target sentences
    words, punctuation = get_word_punctuation_pairs_of_sentence(row['mt'])
    capitalisation = get_word_casing_pairs_of_sentence(row['mt'])
    if (words is not None) and (punctuation is not None) and (capitalisation is not None):
        if (mqm_100_target_sentences < number_of_perfect_rows * 0.9):
            if row['lp'].split('-')[1] in mt_data_train:
                mt_data_train[row['lp'].split('-')[1]].append([words, capitalisation, punctuation])  
            else:
                mt_data_train[row['lp'].split('-')[1]] = [[words, capitalisation, punctuation]]
        else:
            if row['lp'].split('-')[1] in mt_data_dev:
                mt_data_dev[row['lp'].split('-')[1]].append([words, capitalisation, punctuation])  
            else:
                mt_data_dev[row['lp'].split('-')[1]] = [[words, capitalisation, punctuation]]
        mqm_100_target_sentences += 1

source_df = pd.DataFrame(source_data_train, columns=['words', 'capitalisation', 'punctuation'])   
source_df.to_csv('single_files/en/train/source_en_train.csv')   

source_df = pd.DataFrame(source_data_dev, columns=['words', 'capitalisation', 'punctuation'])   
source_df.to_csv('single_files/en/dev/source_en_dev.csv') 

for language in mt_data_train.keys():
    if not os.path.exists("single_files/{}".format(language)):
        os.makedirs("single_files/{}".format(language))
        os.makedirs("single_files/{}/train".format(language))
        os.makedirs("single_files/{}/test".format(language))
        os.makedirs("single_files/{}/dev".format(language))
    mt_df = pd.DataFrame(mt_data_train[language], columns=['words', 'capitalisation', 'punctuation'])   
    mt_df.to_csv("single_files/{}/train/mt_{}_train.csv".format(language, language))


for language in mt_data_dev.keys():
    if language in mt_data_train:
        if not os.path.exists("single_files/{}".format(language)):
            os.makedirs("single_files/{}".format(language))
            os.makedirs("single_files/{}/train".format(language))
            os.makedirs("single_files/{}/test".format(language))
            os.makedirs("single_files/{}/dev".format(language))
        mt_df = pd.DataFrame(mt_data_dev[language], columns=['words', 'capitalisation', 'punctuation'])   
        mt_df.to_csv("single_files/{}/dev/mt_{}_dev.csv".format(language, language))  


# Let's process the test data
df = pd.read_csv('data/test.csv')
df = df.sample(frac = 1)
number_of_perfect_rows = df[df['score'] == 1].shape[0]

source_data = []
mt_data = {}

for row_index, row in tqdm(
        df[df['score'] == 1].iterrows(),
        total=number_of_perfect_rows
    ):
    # Process and save source sentences
    words, punctuation = get_word_punctuation_pairs_of_sentence(row['src'])
    capitalisation = get_word_casing_pairs_of_sentence(row['src'])
    if (words is not None) and (punctuation is not None) and (capitalisation is not None):
        source_data.append([words, capitalisation, punctuation])
    
    # Process and save target sentences
    words, punctuation = get_word_punctuation_pairs_of_sentence(row['mt'])
    capitalisation = get_word_casing_pairs_of_sentence(row['mt'])
    if (words is not None) and (punctuation is not None) and (capitalisation is not None):
        if row['lp'].split('-')[1] in mt_data:
            mt_data[row['lp'].split('-')[1]].append([words, capitalisation, punctuation])  
        else:
            mt_data[row['lp'].split('-')[1]] = [[words, capitalisation, punctuation]]


source_df = pd.DataFrame(source_data, columns=['words', 'capitalisation', 'punctuation'])   
source_df.to_csv('single_files/en/test/source_en_test.csv') 

for language in mt_data.keys():
    if not os.path.exists("single_files/{}".format(language)):
        os.makedirs("single_files/{}".format(language))
        os.makedirs("single_files/{}/train".format(language))
        os.makedirs("single_files/{}/test".format(language))
        os.makedirs("single_files/{}/dev".format(language))
    mt_df = pd.DataFrame(mt_data[language], columns=['words', 'capitalisation', 'punctuation'])   
    mt_df.to_csv("single_files/{}/test/mt_{}_test.csv".format(language, language))
