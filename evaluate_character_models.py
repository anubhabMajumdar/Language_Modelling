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
from pickle import load

# generate a sequence from a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text


modelPath = "character_models/"
modelFiles = [f for f in listdir(modelPath) if isfile(join(modelPath, f))]

mapPath = "character_mappings/"
mapFiles = [f for f in listdir(mapPath) if isfile(join(mapPath, f))]

testFile = open('test.txt', 'r')
testLines = testFile.readlines()
# print testLines

testOutputFile = open('test_output.txt', 'r')
testOutputLines = testOutputFile.readlines()

for f,m in zip(modelFiles, mapFiles):
	for j, k in zip(testLines, testOutputLines):
	# model_name = 'word_models/SH_History_10_Units_300_Layers_3_EPOCHS_200.h5'
		# j = j[:-1]
		if f.startswith('SH'):
			model_name = modelPath + f
			model = load_model(model_name)

			mapping_name = mapPath + m
			mapping = load(open(mapping_name, 'rb'))

			# evaluate model
			print "model --> ", model_name
			print "Seed line --> ", j
			print "Output --> ", (generate_seq(model, mapping, config.HISTORY, j.lower(), 50))
			print "Expected Output --> ", j + " " + k
			print "-------------------------------------------------------"
		







