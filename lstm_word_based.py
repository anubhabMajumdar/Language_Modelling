# The entire code is referenced from https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
# I have made few changes to suit my use cases - Added capability of building multi-layer network

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import config
import h5py as h5py

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

fileName = config.FILENAME
f = open(fileName, 'r')
lines = f.readlines()
lines = [x for x in lines if len(x)>1]

data = ""
for i in lines[len(lines)/3:]:
	data+=i
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
history = config.HISTORY
units = config.UNITS
epochs = config.EPOCHS
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()
for i in range(history, len(encoded)):
	sequence = encoded[i-history:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
# sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
# X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
# define model
# model = Sequential()
# model.add(Embedding(vocab_size, 10, input_length=max_length-1))
# model.add(LSTM(units))
# model.add(Dense(vocab_size, activation='softmax'))

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(config.UNITS, return_sequences=config.LAYERS>1, input_shape=(10, config.UNITS)))
# Add layers
for i in range(1,config.LAYERS):
	if i==config.LAYERS-1:
		model.add(LSTM(config.UNITS))
	else:
		model.add(LSTM(config.UNITS, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=epochs, verbose=2)
# save model
fn = fileName.split(".")
model_name = 'word_models/' + config.FILENAME.split('.')[0] + '_History_' + str(config.HISTORY) + '_Units_' + str(config.UNITS) + '_Layers_' + str(config.LAYERS) + '_EPOCHS_' + str(config.EPOCHS) + '.h5'
model.save(model_name)
# model = load_model(model_name)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'A', 50))
