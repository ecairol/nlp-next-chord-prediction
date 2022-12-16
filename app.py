print("starting");
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import pickle
import numpy as np
import os

## Load the Dictionary
dictionary = 'dictionaries/chords-small.txt'
#dictionary = 'dictionaries/alice-in-wonderland.txt'
#dictionary = 'dictionaries/chords-small.txt'
file = open(dictionary, "r", encoding = "utf8")

# store file in list
lines = []
for i in file:
    lines.append(i)

# Convert list to string
data = ""
for i in lines:
  data = ' '. join(lines) 

# --------------------------
# Data clean up
# replace unnecessary stuff with space
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  #new line, carriage return, unicode character --> replace by space

#remove unnecessary spaces 
data = data.split()
data = ' '.join(data)


print("Sample data: ", data[:20])
print ( 'Total length of data: ', len(data) )

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
# print(sequence_data[:15])

vocab_size = len(tokenizer.word_index) + 1
print("vocab_size: ", vocab_size)

sequences = []

for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
print("Sequences Sample:", sequences[:3])

X = []
y = []

for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
    
X = np.array(X)
y = np.array(y)

print("Data Sample: ", X[:2])
print("Response Sample: ", y[:2])

y = to_categorical(y, num_classes=vocab_size)
print("to_categorical sample:", y[:3])

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

print(" --- Model Summary --- ")
print( model.summary() )


# ------------------------
# Train the model
print(" Traning the model... ")
epochs=70
checkpoint = ModelCheckpoint("next_chords.h5", monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, y, epochs=epochs, batch_size=64, callbacks=[checkpoint])

# ------------------------
# Prediction
model = load_model('next_chords.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

last_predicted = ''

def Predict_Next_Words(model, tokenizer, text):

  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""
  
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
  return predicted_word

while(True):
  text = input("Enter a series of chords: ")
  
  if text == "0":
      print("Ending.")
      break
  
  else:
      try:
          text = text.split(" ")
          text_last_3 = text[-3:]
          
          last_predicted = Predict_Next_Words(model, tokenizer, text_last_3)

          print("Chords: ", " ".join(text), " > ", last_predicted)
          
      except Exception as e:
        print("Error occurred: ",e)
        continue

exit();