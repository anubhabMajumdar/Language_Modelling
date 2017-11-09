from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import config

# generate a sequence of characters with a language model
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

# load the model
model_name = 'character_models/' + config.FILENAME.split()[0] + '_History_' + str(config.HISTORY) + '_Units_' + str(config.UNITS) + '_EPOCHS_' + str(config.EPOCHS) + '.h5'
model = load_model(model_name)
# load the mapping
mapping_name = 'character_mappings/' + config.FILENAME.split()[0] + '_History_' + str(config.HISTORY) + '_Units_' + str(config.UNITS) + '_EPOCHS_' + str(config.EPOCHS) + '.h5'
mapping = load(open(mapping_name, 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, config.HISTORY, 'Caesar', config.OUTPUT_LENGTH))
# test mid-line
print(generate_seq(model, mapping, config.HISTORY, 'Brutus', config.OUTPUT_LENGTH))
# test not in original
print(generate_seq(model, mapping, config.HISTORY, 'Sing a son', config.OUTPUT_LENGTH))