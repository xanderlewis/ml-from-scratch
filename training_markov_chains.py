# Experimenting with Markov chain-type things to generate (possibly) plausible sentences.
# We just have a dict where the keys are words and the corresponding values are lists of possible successor words.
# We can 'learn' such mappings by cranking through example text and simply appending successors whenever they appear,
# allowing duplication in order to reflect relative probabilities.
# Although it's just a plain old Python dict, I call it a model because that's what it is -- albeit a very naïve one.
# The sentences are at least believable at the level of pairs (because each such generated pair was a genuinely naturally
# occurring pair). This is n = 2.

# For n = 3, we learn consecutive triples of words. This requires a lot more data, but produces much better sentences.

import pickle
import string
import random

def extract_words(text):
	"""Gets a list of words from an arbitrary string."""
	trans = str.maketrans(dict.fromkeys(string.punctuation))

	# Remove numerals
	words = ''.join([char for char in text if not char.isdigit()])

	# Convert to lowercase; remove punctuation; remove whitespace and split into words.
	words = text.lower().translate(trans).split()
	
	return words

def learn_pairs(words, model={}):
	"""Returns a 'model' by looking at all consecutive pairs in the given list of words."""
	for i, word in enumerate(words):
		if i != len(words) - 1:
			try:
				model[word].append(words[i + 1])
			except KeyError:
				model[word] = [words[i + 1]]

	return model

def learn_from_text(text, model={}):
	"""Given some text, returns a model trained on it."""
	return learn_pairs(extract_words(text), model)

def generate_random(length, model, initial_word=''):
	"""Generates a sentence (of the desired randomness) of the given length using the given model."""
	sentence = initial_word
	if sentence == '':
		# Choose a random first word
		sentence = random.choice(list(model.keys()))
		current_word = sentence
	else:
		current_word = sentence
	for i in range(length - 1):
		# Sample from the given model until no longer possible
		try:
			current_word = random.choice(model[current_word])
		except KeyError:
			break
		sentence += f' {current_word}'

	return sentence + '.'

def generate(length, model, initial_word='', temperature=0):
	"""Generates a sentence of the given length using the given model."""
	# The higher the 'temperature', the lower the theoretical probability of each word transition.
	# If the temperature is zero, we'll choose the most common next word at each stage and the
	# sentences generated should be entirely deterministic.
	sentence = initial_word
	if sentence == '':
		# Choose a random first word
		sentence = random.choice(list(model.keys()))
		current_word = sentence
	else:
		current_word = sentence
	for i in range(length - 1):
		# Sample from the given model until no longer possible
		try:
			#print(f'choosing word {i}...')
			# Sort the possible successors by frequency
			sorted_succ = succ_ranks(current_word, model)

			#print('sorted')

			# Choose the next word from the top {temperature} words
			current_word = random.choice(sorted_succ[:temperature + 1])

			#print(f'next word is {current_word}')
		except KeyError:
			break
		sentence += f' {current_word}'

	return sentence + '.'


def extend(text, model, length):
	"""Given an intitial string, uses the model to try to extend it."""
	words = extract_words(text)
	return generate(model, length, initial_word=words[-1])

def succ_ranks(text, model):
	"""Given an initial string (for now, a word), returns an ordered list of the most common successors."""
	words = extract_words(text)
	succ = m[words[-1]]

	# The set of successors contains duplicates, so eliminate them
	uniques = list(set(succ))

	# Sort the unique successors according to how often they appear in the original list
	uniques.sort(key=(lambda x: succ.count(x)))
	uniques.reverse()
	return uniques

def save_model(model, name):
	with open(name + '.pickle', 'wb') as f:
		pickle.dump(model, f)

def load_model(name):
	with open(name, 'rb') as f:
		return pickle.load(f)