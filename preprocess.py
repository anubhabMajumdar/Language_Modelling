# The entire code is referenced from https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
# I have made few changes to suit my use cases

import config

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text.lower()

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

fileName = config.FILENAME
# load text
raw_text = load_doc(fileName)
# print(raw_text)

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

# organize into sequences of characters
length = config.HISTORY
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	if all(i.isalpha() or i==' ' or i=='.' or i==',' for i in seq):
		# store
		sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'preprocessed/' + fileName.split('.')[0] + '_History_' + str(length) + '.txt'
save_doc(sequences, out_filename)