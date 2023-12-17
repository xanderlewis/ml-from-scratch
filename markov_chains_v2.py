# Here we'll train a very simple Markov chain 'model' on text, treating each line separately.
# Not sure if 'n-gram' is the right terminology here, but I'll use it anyway.

import pickle
import string
import random
from termcolor import cprint

def learn_from_lines(text, model={}, n=2):
	m = model
	for i, line in enumerate(text.split('\n')):
		m = learn_from_line(line, m, n)
		if i % 50 == 0:
			cprint(f'learnt from line {i}.', 'green')
	return m

def extract_words(line):
	"""Gets a list of words from the given line."""
	trans = str.maketrans(dict.fromkeys(string.punctuation))

	# Remove numerals
	words = ''.join([char for char in line if not char.isdigit()])

	# Convert to lowercase; remove punctuation; remove whitespace and split into words.
	words = line.lower().translate(trans).split()
	
	return words

def learn_from_line(line, model={}, n=2):
	"""Returns a 'model' that has learnt all consecutive n-grams in the given string."""
	# The parameter n refers to the maximal length of chains (n-grams) we want to learn.

	# Clean up the line and get a list of words.
	words = extract_words(line)

	# For each word in the line...
	for i, word in enumerate(words):
		# Learn all the n-grams starting at this word.
		for j in range(n - 1):
			try:
				ngram, succ = tuple(words[i : i + j + 1]), words[i + j + 1]
			except IndexError:
				# Go to the next word
				#cprint('broke', 'red')
				break

			#cprint(f'ngram: {ngram}, succ: {succ}', 'blue')
			# If this successor is already recorded, increment it.
			if ngram in model.keys() and succ in model[ngram].keys():
				model[ngram][succ] += 1
			elif ngram in model.keys():
				# If the ngram is already there, just add the new successor.
				model[ngram][succ] = 1
			else:
				# If both are new, create them.
				model[ngram] = {succ: 1}
				

	return model

def generate(length, model, initial_ngram=(), temperature=0, n=2):
	"""Generates a sentence of the given length using the given model."""
	# If there isn't an initial word, generate one
	if initial_ngram == ():
		output = random.choice([key for key in model.keys() if len(key) == 1])
	else:
		output = initial_ngram

	for i in range(length - len(output)):
		# Try to match an n-gram starting from n-behind the last word
		for j in range(n):
			try:
				# Choose the next word from the top {temperature + 1} words
				succs = model[output[-n + j:]]
				ranked_succs = sorted(succs, key=succs.get)
				next_word = random.choice(ranked_succs[:temperature + 1])
			except KeyError:
				# Try adjusting length of n-gram
				pass
			else:
				# Move to the next word
				output += (next_word,)
				break

	return output

def gs(length, model, start=(), temperature=0, n=2):
	"""Returns a sentence string of the desired length given a starting phrase."""
	if start != ():
		start = tuple(start.split(' '))
	return ' '.join(generate(length, model, initial_ngram=start, temperature=temperature, n=n))

def save_model(model, name):
	with open(name + '.pickle', 'wb') as f:
		pickle.dump(model, f)

def load_model(name):
	with open(name + '.pickle', 'rb') as f:
		return pickle.load(f)