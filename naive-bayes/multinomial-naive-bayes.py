# Here, we explore implementation of Multinomial Naive Bayes Classification Algo
# We use text classification for implementation

# import required libraries
import numpy as np
import pandas as pd
import re

# load dataset
dataset = pd.read_csv('dataset-for-multinomial-nb.csv')

# print(dataset)
# print(dataset.shape)

# functions

# convert text to lower case
def convert_lowercase(text):
    text = text.lower()
    return text

# remove punctuations
def rem_punctuations(text):
    text = re.sub(r'[^\w\s]', '', text) 
    #removes all characters except alphanumeric, underscore and whitespaces
    return text


# preprocess the data

# conversion to lower case
dataset['lowercase_text'] = dataset['text'].apply(convert_lowercase)

# remove unwanted characters
dataset['plain_text'] = dataset['lowercase_text'].apply(rem_punctuations)

# remove unwanted columns from dataset
dataset = dataset[['plain_text','label']]

# tokenize the text
dataset['tokens'] = dataset['plain_text'].apply(lambda x: x.split())

# build vocabulary
vocab = np.array([])

for tokens in dataset['tokens']:
    # print(tokens)
    for word in tokens:
        vocab = np.append(vocab,[word])
        # print(word)
    # print()

# print(vocab)
vocab = np.sort(vocab)
# print(vocab)

# vectorize tokens
def tokens_to_vectors(token, vocab):
    vector = [0] * len(vocab)
    for word in token:
        if word in vocab:
            index = np.where(vocab==word)[0][0]
            # print(index)
            vector[index] += 1
    # print(vector)
    return vector
# tokens_to_vectors(dataset['tokens'][1],vocab)
dataset['bag_of_words_vector'] = dataset['tokens'].apply(lambda x:
                                                         tokens_to_vectors(x,vocab))
# print(dataset)

# grouping based on labels
spam_class = dataset[dataset['label']=='Spam']
ham_class = dataset[dataset['label']=='Ham']
# print(ham_class)

# calculating priors
spam_prior = len(spam_class)/len(dataset)
ham_prior = len(ham_class)/len(dataset)
# print(spam_prior,ham_prior)

# aggregate word count for each class
word_count_vector_spam = [0] * len(vocab)
word_count_vector_ham = [0] * len(vocab)

for vector in spam_class['bag_of_words_vector']:
    word_count_vector_spam = word_count_vector_spam + np.array(vector)
for vector in ham_class['bag_of_words_vector']:
    word_count_vector_ham = word_count_vector_ham + np.array(vector)

# print(len(vocab))
# print(len(word_count_vector_spam))
# print(word_count_vector_spam)

# calculate likelihoods for each word in each class
likelihoods_spam = np.array([])
likelihoods_ham = np.array([])
for x in word_count_vector_spam:
    likelihoods_spam = np.append(likelihoods_spam,
                                 ((x+1) / (len(vocab) +
                                  np.sum(word_count_vector_spam))).round(3))

for x in word_count_vector_ham:
    likelihoods_ham = np.append(likelihoods_ham,
                                ((x+1) / (len(vocab) + 
                                 np.sum(word_count_vector_ham))).round(3))

# print(likelihoods_spam)

# now build the prediction function
def scoreText(text):
    
    # text preprocessing
    convert_lowercase(text)
    rem_punctuations(text)
    text_token = text.split()

    # vectorize
    bow_vector = tokens_to_vectors(text_token,vocab)
    # print(bow_vector)

    # compute the score for each class

    # fetching indices
    indices_of_words = np.array([])
    for i in range(len(bow_vector)):
        if bow_vector[i]>0:
            indices_of_words = np.append(indices_of_words,[int(i)])
    # print(indices_of_words)

    # now we need to fetch likelihoods from these indices from each class
    # and add the log likelihoods
    spam_score_inter = ham_score_inter = 0
    for i in indices_of_words:
        # print(int(i))
        spam_score_inter = spam_score_inter + np.log(likelihoods_spam[int(i)])
        ham_score_inter = ham_score_inter + np.log(likelihoods_ham[int(i)])
    spam_score_final = np.log(spam_prior) + spam_score_inter
    ham_score_final = np.log(ham_prior) + ham_score_inter
    # print(spam_score_final,ham_score_final)
    return [spam_score_final.round(3),ham_score_final.round(3)]

message = "Urgent! Limited time deal." # replace string here to test for different messages
scores = scoreText(message)
# print(scores)
flag = ''
if scores[0]>scores[1]:
    flag='Spam'
else:
    flag='Ham'
print(f'The message "{message}" is a {flag}')