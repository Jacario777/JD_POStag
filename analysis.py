# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 8, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...

import matplotlib.pyplot as plt
import pylab
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

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

#####################  STUDENT CODE BELOW THIS LINE  #####################

def compute_counts(training_data: list, order: int) -> tuple:
	"""
	Input:
	- training_data: a list of (word, POS-tag) pairs
	- order: an integer of either 2 or 3
	Output:
	- If order is 2, then return:
		- A tuple containing the number of tokens in training_data
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

#####################  CODE FOR ANALYSIS  #####################

def read_file(filename):
	"""
	Read the values from file; save the word and tag
	into unique_word, unique_tag, and training_data
	input:filename
	output: training_data
	"""
	file_representation = []
	f = open(str(filename), encoding="utf8")
	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append( (word, tag) )
	f.close()
	return file_representation

def read_sentence(filename):
	"""
	Given a text file, return the untagged sentence
	Input: filename
	Output: list sentence, which untagged
	"""
	sentence = []
	data_file = open(filename, encoding="utf8")
	data_list = data_file.read()
	sentence = data_list.split(" ")
	#Remove the last element, which is: '' (empty)
	sentence.pop()
	return sentence

def run_experiment(training_data, per_run, order, use_smoothing, test_data, true_data):
	"""
	Input:
	- training_data
	- per_run: a list containing all the percentage of data that will be run
	- order: 2 or 3
	- use_smoothing: true or false
	- test_data: sentence, untagged
	- true_data: sentence, tagged
	Output:
	- result_list: a dictionary that shows the result of all experiments
	"""
	run_len = len(training_data)
	n_per_run = []
	cent_run = []
	for ele in range(len(per_run)):
		n_per_run.append(int(run_len * per_run[ele]))
		#Multiply each element in per_run by 100 to get the percentage (%)
		cent_run.append(100*per_run[ele])
	#print(n_per_run)
	unique_word = set()
	unique_tag = set()
	n_training_data = []
	result_list = {}
	index = 0
	for run in range(run_len):
		run_time = n_per_run[index]
		n_training_data.append(training_data[run])
		unique_word.add(training_data[run][0])
		unique_tag.add(training_data[run][1])
		if run == run_time - 1:
			hmm = build_hmm(n_training_data, list(unique_tag), list(unique_word), order, use_smoothing)
			#print(hmm)
			hmm.emission_matrix = update_hmm(hmm.emission_matrix, test_data, list(unique_word))
			if order == 2:
				result_data = bigram_viterbi(hmm, test_data)
			elif order == 3:
				result_data = trigram_viterbi(hmm, test_data)
			result_list[cent_run[index]] = compare_data(result_data, true_data)
			index += 1
			if index > len(per_run) -1:
				return result_list

def compare_data(result_data, true_data):
	"""
	Input:
	result_data: the resulting tagged sentence from hmm
	true_data: sentence, tagged
	Output:
	per_correct: percentage of matching tag
	"""
	total_data = len(result_data)
	#print(total_data)
	#print(len(true_data))
	per_correct = 0
	correct_tag = 0
	for word in range(total_data):
		if result_data[word][1] == true_data[word][1]:
			correct_tag += 1
	#Multiply by 100 to get the result in percentage (%)
	per_correct = (correct_tag / total_data)*100
	return per_correct 

#####################  CODE FOR GRAPHING  #####################

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list

    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    plt.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


#####################  CODE FOR IMPLEMENTATION  #####################

training_data = read_file('training.txt')
true_data = read_file("testdata_tagged.txt")
test_data = read_sentence("testdata_untagged.txt")
experiment_data = []
per_run = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

#For order 2, no smoothing
order = 2
use_smoothing = False
result2_ns = run_experiment(training_data, per_run, order, use_smoothing, test_data, true_data)
print(result2_ns)
experiment_data.append(result2_ns)

#For order 3, no smoothing
order = 3
use_smoothing = False
result3_ns = run_experiment(training_data, per_run, order, use_smoothing, test_data, true_data)
print(result3_ns)
#experiment_data.append(result3_ns)

#For order 2, with smoothing
order = 2
use_smoothing = True
result2_s = run_experiment(training_data, per_run, order, use_smoothing, test_data, true_data)
print(result2_s)
experiment_data.append(result2_s)

#For order 3, with smoothing	
order = 3
use_smoothing = True
result3_s = run_experiment(training_data, per_run, order, use_smoothing, test_data, true_data)
print(result3_s)
#experiment_data.append(result3_s)

plot_lines(experiment_data, "Performance of Bigram and Trigram HMM with/without smoothing", "Percentage of training data used (%)", "Percentage of tags in HMM that agrees with the test set (%)", 
        labels=["Bigram without smoothing", "Trigram without smoothing", "Bigram with smoothing", "Trigram with smoothing"], filename=None)
