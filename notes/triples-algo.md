As **input**, we'll have a string where each line is a single 'sentence' to be trained on. Use \n to separate these.

As **output**, we want a dictionary where the keys are words and the corresponding values are also dicts where the keys are probabilities (proportions of occurences, basically) and the correponding values are words that occur with these probabilities.

We want to learn an association between pairs of words and the next word after that. So basically, word triples.

So we want a dict that looks like

{['I, want']: {0.5 : 'to', 0.5: 'a'}}.

We can use this dict to look up the likely next word given a pair, or...

----

Actually..... for simplicity, when we train on triples let's just consider pairs of words to be words themselves. (after all, this is supposed to be a Markov process, so we ONLY care about the current state. we should't think of pairs of words as being anything other than another type of word.)

So maybe let's talk about 'tokens' instead of words.

Training on the example 'I want to go\nI want a cake' with (let's call it) n = 2 yields

```

{
	'I': {1: 'want'},
	'want': {0.5: 'to', 0.5: 'a'},
	'to': {1: 'go'},
	'a': {1: 'cake'},
	'I want': {0.5: 'to', 0.5: 'a'},
	'want to': {1: 'go'},
	'want a': {1: 'cake'}
}.
```

We could do the same process but with *n = 3* and learn associations between *triples* of words and their successors as well.
Could be interesting...

So ultimately the model we produce is just a {input string} |--> {[likely outputs strings]} type thing.
At generation time, we just choose what to search for (be it the successor of a single word, or a pair of words, etc.) by varying that input string until we get something we want and choosing possibly with some stochasticity.

# The actual 'training' algorithm

The main (hyper)parameter for training is n, which defines the maximum length of those 'input strings'.

Instead of talking about *probabilities*, let's just record the number of occurences. This makes it easier to update. We can recover probabilities later if we really want by dividing by a sum anyway.

(0) clean up the data (split each line up into a list of words, remove punctuation, etc.)

(I) of the current line, look at the first word and record it and its successor in the dict, setting the key to 1.
(II) now look at this word and its successor and record the successor of that to the dict, again setting the key to 1.
(III) (keep doing this for n.)
(IV) move onto the next word in the line. record its successor as before, except if it's already there we just increment the integer (number of occurrences) key.
(V) ...


I wonder if increasing n in this algorithm will give us something that generates quite plausible (if meandering and meaningless) sentences... Also, since it's a general sequence predictor, we could train it on any sort of sequential data: maybe melodic data, or pure displacement information from a wave file.