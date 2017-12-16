# The entire code is referenced from https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
# I have made few changes to suit my use cases

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
from os import listdir
from os.path import isfile, join

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


path = "word_models/"
files = [f for f in listdir(path) if isfile(join(path, f))]

testFile = open('test.txt', 'r')
testLines = testFile.readlines()

testOutputFile = open('test_output.txt', 'r')
testOutputLines = testOutputFile.readlines()

for i in files:
	for j, k in zip(testLines, testOutputLines):
	# model_name = 'word_models/SH_History_10_Units_300_Layers_3_EPOCHS_200.h5'
		if i.startswith('SH'):
			model_name = path + i
			model = load_model(model_name)
			# evaluate model
			print model_name
			print j
			print(generate_seq(model, tokenizer, max_length-1, j, 10))
			print j + " " + k
			print "-------------------------------------------------------"








