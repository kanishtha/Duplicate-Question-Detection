
# coding: utf-8

# In[ ]:

'''Important Links - 
*** Understanding embedding layer in the model -
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

*** Understanding how to make lstm model in keras - 
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

*** I couldnt understand AdaDelta - 
https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM

*** Why we need to clip the gradient descent
http://nmarkou.blogspot.in/2017/07/deep-learning-why-you-should-use.html
'''

# In[1]:

from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import model_from_json

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.layers as lyr
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
import tensorflow as tf


# In[2]:

TRAIN_CSV = 'questions.csv'
#TRAIN_CSV = 'sample.csv'
EMBEDDING_FILE = '/home/hemant/Desktop/GoogleNews-vectors-negative300.bin'


# In[15]:

train_df = pd.read_csv(TRAIN_CSV)
stops = set(stopwords.words('english'))


# In[16]:

def stringToWordList(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text


# In[17]:

questions_cols = ['question1', 'question2']
vocab = dict()
maxSeqLength = 0
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

for index, row in train_df.iterrows():
    for question in questions_cols:
        q2num = []
        for word in stringToWordList(row[question]):
            if word in stops and word not in word2vec.vocab:
                continue
            if word not in vocab:
                vocab[word] = len(vocab)+1
            q2num.append(vocab[word])
        train_df.set_value(index,question,q2num)
        if len(q2num) > maxSeqLength:
            maxSeqLength = len(q2num)

embeddingDim = 300
embeddings = 1 * np.random.randn(len(vocab) + 1, embeddingDim)
embeddings[0] = 0

# Build the embedding matrix
for word, index in vocab.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec
        


# In[20]:

# Split the data into Training set and Validation set
test_size = 20000
validation_size = 40000
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = train_df['is_duplicate']

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_val, Y_train_val, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_test = {'left': X_test.question1, 'right': X_test.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_test = Y_test.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=maxSeqLength)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# In[21]:

# Building the Siamese LSTM Model

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 5

# The visible layer
left_input = Input(shape=(maxSeqLength,), dtype='int32')
right_input = Input(shape=(maxSeqLength,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embeddingDim, weights=[embeddings], input_length=maxSeqLength, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = Bidirectional(LSTM(n_hidden))

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Concatenate the embeddings with their product and squared difference.
p = lyr.multiply([left_output, right_output])
negative_right_output = lyr.Lambda(lambda x: -x)(right_output)
d = lyr.add([left_output, negative_right_output])
q = lyr.multiply([d, d])
v = [left_output, right_output, p, q]
lstm_output = lyr.concatenate(v)

merged = lyr.BatchNormalization()(lstm_output)
merged = lyr.Dense(64, activation='relu')(merged)
merged = lyr.Dropout(0.2)(merged)
merged = lyr.BatchNormalization()(merged)
preds = lyr.Dense(1, activation='sigmoid')(merged)
model = Model(input=[left_input,right_input], output=preds)
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy'])

# Start training
training_start_time = time()

lstm_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size,epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

# evaluate the model
scores = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# In[ ]:




