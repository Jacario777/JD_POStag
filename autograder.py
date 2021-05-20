# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 8, Problem 2

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from provided.py, but they have
# to be copied over here.

import math
import random
import numpy
from collections import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags



#####################  STUDENT CODE BELOW THIS LINE  #####################


test1 = [('A', 'T'), ('dog', 'N'), ('jumped', 'V'), ('over', 'P'), ('the', 'T'), ('fence', 'N'), ('.', '.')]
words1 = ['A', 'dog', 'jumped', 'over', 'the', 'fence', '.']
tags1 = ['T', 'N', 'V', 'P', '.']
test2 = [('Buffalo', 'PN'), ('buffalo', 'N'), ('Buffalo', 'PN'), ('buffalo', 'N'), ('buffalo', 'V'), ('buffalo', 'V'), ('Buffalo', 'PN'), ('buffalo', 'N'), ('.', '.')]
words2 = ['Buffalo', 'buffalo', '.'] 
tags2 = ['PN', 'N', 'V', '.']
test3 = [('The','T'), ('quick','A'), ('brown','A'), ('fox','N'), ('jumps','V'), ('over','P'), ('the','T'), ('lazy','A'), ('dog','N'), ('.', '.')]
words3 = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
tags3 = ['T', 'A', 'N', 'V', 'P', '.']
test4 = [('cow','N'), ('sheep','N'), ('llama','N'), ('.', '.')]
words4 = ['cow', 'sheep', 'llama', '.']
tags4 = ['N', '.']

untag1 = ['The', 'lazy', 'dog', 'jumps', 'over', 'the', 'quick', 'brown', 'fox', '.']
untag2 = ['The', 'lazy', 'dog', 'loves', 'the', 'quick', 'brown', 'fox', '.']

def compute_counts(training_data: list, order: int) -> tuple:
	"""
	Input:
	- training_data: a list of (word, POS-tag) pairs
	- order: an integer of either 2 or 3
	Output:
	- If order is 2, then return:
		- A tuple containing the number of tokens (or length) in training_data
		- a dictionary that contains C(ti, wi), which counts the number of times wi is tagged with ti
		- a dictionary that contains C(ti), which counts the number of times ti appears
		- a dictionary that contains C(t(i-1), ti) 
	- If order is 3, then return:
		- as the fifth element, a dictionary that contains C(t(i-2), t(i-1), ti), along
		with the other 4 elements
	"""
	#Inititate all the elements for the tuples
	word_tag = defaultdict(lambda: defaultdict(int))
	num_tag = defaultdict(int)
	cont_tag = defaultdict(lambda: defaultdict(int))
	cont2_tag = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	pair = 0
	data_len = len(training_data)

	#running through the training_data to record down tag and word
	for pair in range(data_len):
		word = training_data[pair][0]
		tag = training_data[pair][1]
		num_tag[tag] += 1
		word_tag[tag][word] += 1
		#For looking one element ahead
		if pair < data_len - 1:
			cont_tag[tag][training_data[pair + 1][1]] += 1
		#For looking two elements ahead
		if pair < data_len - 2:
			cont2_tag[tag][training_data[pair + 1][1]][training_data[pair + 2][1]] += 1

	if order == 2:
		return (data_len, word_tag, num_tag, cont_tag)

	elif order == 3:
		return (data_len, word_tag, num_tag, cont_tag, cont2_tag)

"""
test cases:

print(compute_counts(test1, 2))
print(compute_counts(test1, 3))

print(compute_counts(test2, 2))
print(compute_counts(test2, 3))

print(compute_counts(test3, 2))
print(compute_counts(test3, 3))

compute1 = compute_counts(test1, 3)
compute2 = compute_counts(test2, 3)
compute3 = compute_counts(test3, 3)

compute1s = compute_counts(test1, 2)
compute2s = compute_counts(test2, 2)
compute3s = compute_counts(test3, 2)
"""
def compute_initial_distribution(training_data: list, order: int) -> dict:
	"""
	Input: 
	- training_data: a list of (word, POS-tag) pairs
	Output:
	- If order is 2:
		- return pi1
	- If order is 3:
		- return pi2
	"""
	if order == 2:
		pi_1 = defaultdict(int)
		pi_1[training_data[0][1]] += 1
		return pi_1
	elif order == 3:
		pi_2 = defaultdict(lambda: defaultdict(int))
		pi_2[training_data[0][1]][training_data[1][1]] += 1
		return pi_2

"""
test cases:
print(compute_initial_distribution(test1, 2))
print(compute_initial_distribution(test1, 3))
"""
#print(compute_initial_distribution(test2, 2))
#print(compute_initial_distribution(test2, 3))

#print(compute_initial_distribution(test3, 2))
#print(compute_initial_distribution(test3, 3))

def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
	"""
	Input:
	- unique_words: from read_pos_file
	- unique_tags: from read_pos_file
	- W: the dictionary from compute_counts for C(ti, wi)
	- C: the dictionary from compute_counts for C(ti)
	Output:
	The emission matrix represented by a nested dictionary
	"""
	emis_matrix = defaultdict(lambda: defaultdict(int))
	for tags in unique_tags:
		for words in unique_words:
			emis_prob = W[tags][words] / C[tags]
			emis_matrix[tags][words] = emis_prob
	return emis_matrix

"""
test cases:

print(compute_emission_probabilities(words1, tags1, compute1[1], compute1[2]))
print(compute_emission_probabilities(words2, tags2, compute2[1], compute2[2]))
print(compute_emission_probabilities(words3, tags3, compute3[1], compute3[2]))
"""

def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
	"""
	Input: 
	- unique_tags
	- num_tokens: the length of training_data
	- C1, C2, C3 (ti, ti-1, ti-2)
	- order (2 or 3)
	Output:
	- a list containing pi0, pi1, and pi2
	"""
	
	lin_inter = [0,0,0]
	tag_len = len(unique_tags)
	#To form i-2, i-1, and i of tag in order = 3
	if order == 3:
		for tag in range(0, tag_len-2):
			tag1 = unique_tags[tag]
			tag_keys = C3[tag1].keys()
			for tag2 in tag_keys:
				tag2_keys = C3[tag1][tag2].keys()
				for tag3 in tag2_keys:
					alpha_0 = (C1[tag2] - 1)/num_tokens
					#Check if any of the denominators are zero
					if C1[tag2] - 1 == 0:
						alpha_1 = 0
					if C2[tag1][tag2] - 1 == 0:
						alpha_2 = 0
					if C1[tag2] - 1 != 0 and C2[tag1][tag2] - 1 != 0:
						alpha_1 = (C2[tag2][tag3] - 1) / (C1[tag2] - 1)
						alpha_2 = (C3[tag1][tag2][tag3] - 1) / (C2[tag1][tag2] - 1)
					#get argmax
					alpha_list = [alpha_0, alpha_1, alpha_2]
					max_val = max(alpha_list)
					i = alpha_list.index(max_val)
					lin_inter[i] += C3[tag1][tag2][tag3]
	elif order == 2:
		#to form i-1 and i of tag in order = 2
		for tag in range(0, tag_len-1):
			tag1 = unique_tags[tag]
			tag_keys = C2[tag1].keys()
			for tag2 in tag_keys:
				alpha_0 = (C1[tag2] - 1)/num_tokens
				#Check if any of the denominators are zero
				if C1[tag1] - 1 == 0:
					alpha_1 = 0
				else:
					alpha_1 = (C2[tag1][tag2] - 1) / (C1[tag1] - 1)
				#get argmax
				alpha_list = [alpha_0, alpha_1]
				max_val = max(alpha_list)
				i = alpha_list.index(max_val)
				lin_inter[i] += C2[tag1][tag2]

	list_sum = lin_inter[0] + lin_inter[1] + lin_inter[2]
	lin_inter[:] = [x / list_sum for x in lin_inter]
	return lin_inter

"""
test cases:
compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int)

order = 3:
print(compute_lambdas(tags1, compute1[0], compute1[2], compute1[3], compute1[4], 3))
print(compute_lambdas(tags2, compute2[0], compute2[2], compute2[3], compute2[4], 3))
print(compute_lambdas(tags3, compute3[0], compute3[2], compute3[3], compute3[4], 3))
order = 2:
print(compute_lambdas(tags1, compute1s[0], compute1s[2], compute1s[3], {}, 2))
print(compute_lambdas(tags2, compute2s[0], compute2s[2], compute2s[3], {}, 2))
print(compute_lambdas(tags3, compute3s[0], compute3s[2], compute3s[3], {}, 2))
""" 
def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
	"""
	Input:
	- training_data
	- unique_tags
	- unique_words
	- order (2 or 3)
	- use_smoothing (boolean)
	Output:
	- hmm: a fully trained hmm class 
	"""
	counts_tuple = compute_counts(training_data, order)
	num_tokens = counts_tuple[0]
	lambda_list = []
	if order == 3:
		second_order = counts_tuple[4]
		transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		#C1 is ti, C2 is ti-1, C3 is ti-2
		C1 = counts_tuple[2]
		C2 = counts_tuple[3]
		C3 = counts_tuple[4]
		if use_smoothing == False:
			lambda_list = [0,0,1]
		else:
			lambda_list = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
		#Use the equation for trigram transition probabilities
		for prev_2tag in C3.keys():
			for prev_tag in C3[prev_2tag].keys():
				for curr_tag in C3[prev_2tag][prev_tag].keys():
					prob_val = lambda_list[2]*(C3[prev_2tag][prev_tag][curr_tag]/C2[prev_2tag][prev_tag]) + lambda_list[1]*(C2[prev_tag][curr_tag]/C1[prev_tag]) + lambda_list[0]*(C1[curr_tag]/num_tokens)
					transition_matrix[prev_2tag][prev_tag][curr_tag] = prob_val
	elif order == 2:
		second_order = counts_tuple[3]
		transition_matrix = defaultdict(lambda: defaultdict(int))
		#C1 is ti, C2 is ti-1, C3 is empty because bigram does not need it
		C1 = counts_tuple[2]
		C2 = counts_tuple[3]
		C3 = {}
		if use_smoothing == False:
			lambda_list = [0,1,0]
		else:
			lambda_list = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
		#Use the equation for bigram transition probabilities
		for prev_tag in C2.keys():
			for curr_tag in C2[prev_tag].keys():
				transition_matrix[prev_tag][curr_tag] = lambda_list[1]*(C2[prev_tag][curr_tag]/C1[prev_tag]) + lambda_list[0]*(C1[curr_tag]/num_tokens)
	emission_matrix = compute_emission_probabilities(unique_words, unique_tags, counts_tuple[1], counts_tuple[2])
	initial_distribution = compute_initial_distribution(training_data, order)
	hmm = HMM(order, initial_distribution, emission_matrix, transition_matrix)
	return hmm


#Test cases:
"""
hmm = build_hmm(test2, tags2, words2, 2, False)
print(hmm.initial_distribution)
print(hmm.emission_matrix)
print(hmm.transition_matrix)

hmm = build_hmm(test1, tags1, words1, 3, False)
print(hmm.initial_distribution)
print(hmm.emission_matrix)
print(hmm.transition_matrix)

hmm = build_hmm(test4, tags4, words4, 3, False)
print(hmm.initial_distribution)
print(hmm.emission_matrix)
print(hmm.transition_matrix)
"""

def update_hmm(emission, sentence, unique_word):
	"""
	Input:
	- emission: the emission matrix of training_data
	- sentence: a list representing an untagged sentence
	- unique_word: a list containing all unique words in training_data

	Output:
	- emission, with words that are not in unique_word and updated probability
	"""
	#Get all the words that are not in emission
	not_emission = []
	for word in sentence:
		if word not in unique_word:
			not_emission.append(word)
	
	#Update every state in emission (given there are words not in emission)
	if len(not_emission) > 0:
		for tag in emission:
			total_prob = 0
			for word in emission[tag]:
				emission[tag][word] += 0.00001
				total_prob += emission[tag][word]
			for word_n in not_emission:
				emission[tag][word_n] = 0.00001
				total_prob += emission[tag][word_n]
		#Normalize the emission values
			for word in emission[tag]:
				emission[tag][word] = emission[tag][word]/total_prob
	return emission

def trigram_viterbi(hmm, sentence: list) -> list:
	"""
    Run the Viterbi algorithm to tag a sentence assuming a trigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

	# Initialization
	viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	
	for tag in unique_tags:
		for tag2 in unique_tags:
			if (hmm.initial_distribution[tag2][tag] != 0) and (hmm.emission_matrix[tag][sentence[1]] != 0) and (hmm.emission_matrix[tag2][sentence[0]] != 0):
				viterbi[tag2][tag][0] = math.log(hmm.initial_distribution[tag2][tag]) + math.log(hmm.emission_matrix[tag][sentence[1]]) + math.log(hmm.emission_matrix[tag2][sentence[0]])
			else:
				viterbi[tag2][tag][0] = -1 * float('inf')
	
	
	#Dynamic Programming
	for t in range(1, len(sentence)):
		for s in unique_tags:
			backpointer["No_Path"][s][t] = "No_Path"
			for s_prime in unique_tags:
				max_value = -1 * float('inf')
				max_state = None
				for s_prime2 in unique_tags:
					val1= viterbi[s_prime2][s_prime][t-1]
					val2 = -1 * float('inf')
					if hmm.transition_matrix[s_prime2][s_prime][s] != 0:
						val2 = math.log(hmm.transition_matrix[s_prime2][s_prime][s])
					curr_value = val1 + val2
					if curr_value > max_value:
						max_value = curr_value
						max_state = s_prime2
				val3 = -1 * float('inf')
				if hmm.emission_matrix[s][sentence[t]] != 0:
					val3 = math.log(hmm.emission_matrix[s][sentence[t]])
				viterbi[s_prime][s][t] = max_value + val3
				if max_state == None:
					backpointer[s_prime][s][t] = "No_Path"
				else:
					backpointer[s_prime][s][t] = max_state
	for ut in unique_tags:
		string = ""
		for i in range(0, len(sentence)):
			for j in range(1, len(sentence)):
				if (viterbi[ut][i] != float("-inf")):
					string += str(int(viterbi[ut][i][j])) + "\t"
				else:
					string += str(viterbi[ut][i][j]) + "\t"

	# Termination
	max_value = -1 * float('inf')
	last_state = None
	final_time = len(sentence) - 1
	for s_prime in unique_tags:
		for s_prime2 in unique_tags:
			if viterbi[s_prime2][s_prime][final_time] > max_value:
				max_value = viterbi[s_prime2][s_prime][final_time]
				last_state = s_prime
				last_state2 = s_prime2
	if last_state == None:
		last_state = "No_Path"
		last_state2 = "No_Path"

	# Traceback
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence)-1], last_state))
	tagged_sentence.append((sentence[len(sentence)-2], last_state2))
	for i in range(len(sentence)-3, -1, -1):
		next_tag = tagged_sentence[-1][1]
		next_tag2 = tagged_sentence[-2][1]
		curr_tag = backpointer[next_tag][next_tag2][i+2]
		tagged_sentence.append((sentence[i], curr_tag))
	tagged_sentence.reverse()
	return tagged_sentence

#test cases:
"""
hmm = build_hmm(test3, tags3, words3, 3, True)
hmm.emission_matrix = update_hmm(hmm.emission_matrix, untag1, words3)
#print(hmm.emission_matrix)
result_data = trigram_viterbi(hmm, untag1)
print(result_data)

_trigram_initial_distribution = {'Coin1': {'Coin1': .01, 'Coin2': .49}, 'Coin2': {'Coin1': .01, 'Coin2': .49}}
_trigram_emission_probabilities = {'Coin1': {'Heads': .9, 'Tails': .1}, 'Coin2': {'Heads': .5, 'Tails': .5}}
_trigram_transition_matrix = {'Coin1': {'Coin1': {'Coin1': .35, 'Coin2': .65}, 'Coin2': {'Coin1': .5, 'Coin2': .5}}, 'Coin2': {'Coin1': {'Coin1': .10, 'Coin2': .90}, 'Coin2': {'Coin1': .65, 'Coin2': .35}}}

hmm = HMM(3, _trigram_initial_distribution, _trigram_emission_probabilities, _trigram_transition_matrix)
sentence = ['Heads', 'Heads', 'Tails', 'Heads', 'Tails', 'Heads', 'Tails']
"""
