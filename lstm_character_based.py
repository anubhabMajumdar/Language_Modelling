from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import config

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'preprocessed/' + config.FILENAME.split('.')[0] + '_History_' + str(config.HISTORY) + '.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(config.UNITS, return_sequences=config.LAYERS>1, input_shape=(X.shape[1], X.shape[2])))
# Add layers
for i in range(1,config.LAYERS):
	if i==config.LAYERS-1:
		model.add(LSTM(config.UNITS))
	else:
		model.add(LSTM(config.UNITS, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=config.EPOCHS, verbose=2)

# save the model to file
model_name = 'character_models/' + config.FILENAME.split('.')[0] + '_History_' + str(config.HISTORY) + '_Units_' + str(config.UNITS) + '_Layers_' + str(config.LAYERS) + '_EPOCHS_' + str(config.EPOCHS) + '.h5'
model.save(model_name)
# save the mapping
mapping_name = 'character_mappings/' + config.FILENAME.split('.')[0] + '_History_' + str(config.HISTORY) + '_Units_' + str(config.UNITS) + '_Layers_' + str(config.LAYERS) + '_EPOCHS_' + str(config.EPOCHS) + '.h5'
dump(mapping, open(mapping_name, 'wb'))
