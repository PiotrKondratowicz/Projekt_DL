import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics import classification_report

# glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)

word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

df = pd.read_csv("depression_dataset_reddit_cleaned.csv", sep=',')

x = df["clean_text"]
y = df["is_depression"]

print(len(x))

count_y_equals_1 = len(df[df["is_depression"] == 1])
print(count_y_equals_1)
print(len(x) - count_y_equals_1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_sequences_padded, Y_train, epochs=10, batch_size=32, validation_split=0.2)

predictions = model.predict(test_sequences_padded)

predictions = (predictions > 0.5).astype(int).flatten()
print(classification_report(Y_test, predictions))
