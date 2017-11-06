from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

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

# source text
# data = """ Jack and Jill went up the hill\n
# 		To fetch a pail of water\n
# 		Jack fell down and broke his crown\n
# 		And Jill came tumbling after\n """
# data = """ FLAVIUS. Hence, home, you idle creatures, get you home.\n
#     Is this a holiday? What, know you not,\n
#     Being mechanical, you ought not walk\n
#     Upon a laboring day without the sign\n
#     Of your profession? Speak, what trade art thou?\n
#   FIRST COMMONER. Why, sir, a carpenter.\n
#   MARULLUS. Where is thy leather apron and thy rule?\n
#     What dost thou with thy best apparel on?\n
#     You, sir, what trade are you?\n
#   SECOND COMMONER. Truly, sir, in respect of a fine workman, I am\n
#     but, as you would say, a cobbler.\n
#   MARULLUS. But what trade art thou? Answer me directly.\n
#   SECOND COMMONER. A trade, sir, that, I hope, I may use with a safe\n """

fileName = "JuliusCaesar.txt"
f = open(fileName, 'r')
lines = f.readlines()
lines = [x for x in lines if len(x)>1]

data = ""
for i in lines:
	data+=i
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
history = 10
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
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=10, verbose=2)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'Well Brutus thou art noble', 20))
# print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
# print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))
# print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))