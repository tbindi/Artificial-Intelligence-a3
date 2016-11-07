# ###################################
# # CS B551 Fall 2016, Assignment #3
# #
# # Your names and user ids:
# # Mohit Galvankar mgalvank
# # Supreet S sushivan
# # Thanmai Bindia tbindi
# #
# # (Based on skeleton code by D. Crandall)
# #
# #
# ####
# ''''''''''

# # Report

# 1.a)Initialized all the needed states
# Initial state = #Creates a dict of tags and number of times they occured at the start of the sentence.
# Emission state = #Creates a dict of all words in train data and the number of times they occurred as a tag = noun,verb,etc.
# Transition state =  #Creates a dict of all tag transitions combination : - noun to noun, noun to verb, etc
# All the above state tables are not probabilities just counts. Probabilities are calculated when needed.

# 1.b)Simple model
# For simple model,given a sentence:
#     for each word in sentence:
#         for each tag :
#             calculate the max(no of times the word occured as a tag/no of times the word occured) and assign that tag to the word
# If word not in database, return a new tag as "na"


# 1.c)HMM model
# Implemented a viterbi algorithm.
# Initialize a viterbi matrix of N x T where is N is the words in the sentence and T is the number of tags.
# This viterbi matrix is a dictionary of dictionary where each word is a key in the main dictionary and has it's own dictionary with all tags as it's keys.
# Algorithm:-
# First compute the first col of the matrix.
# For each tag in the first word:
#     assigned that tag a value = initial probability of that tag * emission probability of that word

# For each word in viterbi matrix from the second word in sentence:
#     for each tag in each word:
#         compute max ( previous word value in word[tag] * transition probability from previous word[tag] to current word[tag] * emission probability of current word
#         also tag each state with the tag represented by the max calculated above inorder to backtrack and get the pos sequence.

# After the matrix is generated, backtrack from the last col in matrix and append all tags for each word and return tag tuple.


# 1.d.)Complex model
# For complex model, computed the current word[tag] value based on the max of all the possible combination of tags in previous two words. Similar to viterbi but comparing last two words.
# Basically used a trigram model. Computed a 12x12 tag transition matrix for each word[tag] value computation and multipled each by that specific state probability
# and also the emission probability of current state. Also tag each state with the tag represented by the max calculated above inorder to backtrack and get the pos sequence.

# Reference :-
# Referred to below video for viterbi algorithm concept clearance :-
# https://www.youtube.com/watch?v=O_q82UMtjoM
# '''''''''
# ####

import random
import math
from pprint import pprint
import copy
from collections import Counter
from collections import OrderedDict
from collections import defaultdict

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

smooth_constant = 10e-6

class Solver:

    WORD_DET = 'det'
    WORD_VERB = 'verb'
    WORD_ADP = 'adp'  # adposition
    WORD_ADV = 'adv'
    WORD_ADJ = 'adj'
    WORD_NOUN = 'noun'
    WORD_DOT = '.'
    WORD_PRON = 'pron'
    WORD_CONJ = 'conj'
    WORD_PRT = 'prt'
    WORD_NUM = 'num'
    WORD_X = 'x'


    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        prob = -math.log(smooth_constant)
        initial = -math.log(self.get_initial_prob(label[0])) - math.log(self.get_emission(0, label[0], sentence))
        for index in range(1,len(sentence)):
            try:
                emission = -math.log(self.get_emission(index,label[index],sentence))
            except:
                emission = -math.log(smooth_constant)
            try:
                if 'na' in label:
                    transition = -math.log(smooth_constant)
                else:
                    transition = -math.log(self.get_transition_prob_complex(label[index-1],label[index]))
            except:
                transition = -math.log(smooth_constant)
            prob -= emission * transition
        prob -= initial
        return prob

    def get_blank_dict(self):
        blank_dict = dict()
        blank_dict[Solver.WORD_DET] = smooth_constant
        blank_dict[Solver.WORD_VERB] = smooth_constant
        blank_dict[Solver.WORD_ADP] = smooth_constant
        blank_dict[Solver.WORD_ADV] = smooth_constant
        blank_dict[Solver.WORD_ADJ] = smooth_constant
        blank_dict[Solver.WORD_NOUN] = smooth_constant
        blank_dict[Solver.WORD_DOT] = smooth_constant
        blank_dict[Solver.WORD_PRON] = smooth_constant
        blank_dict[Solver.WORD_CONJ] = smooth_constant
        blank_dict[Solver.WORD_PRT] = smooth_constant
        blank_dict[Solver.WORD_NUM] = smooth_constant
        blank_dict[Solver.WORD_X] = smooth_constant
        return blank_dict

    def get_blank_dict_viterbi(self):
        blank_dict_viterbi = OrderedDict()
        blank_dict_viterbi[Solver.WORD_DET] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_VERB] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_ADP] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_ADV] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_ADJ] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_NOUN] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_DOT] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_PRON] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_CONJ] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_PRT] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_NUM] = [smooth_constant,0]
        blank_dict_viterbi[Solver.WORD_X] = [smooth_constant,0]
        return blank_dict_viterbi




    def __init__(self):
        self.word_dict = dict()
        self.tag_occurence = dict()
        self.initial_state_dict = ()
        self.transition_dict = dict()
        self.transition_dict_complex = dict()
        self.viterbi_matrix_dict = OrderedDict()
        self.complex_matrix_dict = OrderedDict()
        self.transition_dict_complex2 = dict()
    # Do the training!
    #



    def train(self, data):
        #Calculate the tag occurence :-
        #Creates a dict of all tags and number of times they occurred in the train data set
        self.cal_Tag_Occurence(data)

        #Calculate initial state matrix :-
        #Creates a dict of tags and number of times they occured at the start of the sentence.
        self.initial_state_dict = {}
        for sentence in data:
            tag = str(sentence[1][0])
            if tag not in self.initial_state_dict:
                self.initial_state_dict[tag] = smooth_constant
            self.initial_state_dict[tag] += 1

        # Intitialize Transition matrix
        self.transition_dict = self.get_blank_dict()
        #Calculate transition matrix
        #Creates a dict of all tag transitions combination : - noun to noun, noun to verb, etc
        for sentence in data:
            for word_index in range(0, len(sentence[1])-1):
                current_type = sentence[1][word_index]
                next_type = sentence[1][word_index+1]
                if type(self.transition_dict[current_type]) != dict:
                    self.transition_dict[current_type] = self.get_blank_dict()
                self.transition_dict[current_type][next_type] += 1

        # Intitialize Transition complex matrix
        self.transition_dict_complex = self.get_blank_dict()
        # Calculate transition matrix
        # Creates a dict of all tag transitions combination : - noun to noun skipping a word, noun to verb, etc
        for sentence in data:
            for word_index in range(0, len(sentence[1]) - 2):
                current_type = sentence[1][word_index]
                next_type = sentence[1][word_index + 2]
                if type(self.transition_dict_complex[current_type]) != dict:
                    self.transition_dict_complex[current_type] = self.get_blank_dict()
                self.transition_dict_complex[current_type][next_type] += 1

        # Intitialize Transition complex matrix 2
        self.transition_dict_complex2 = {}
        # Calculate transition matrix
        # Creates a dict of all tag transitions combination : - noun to noun skipping a word, noun to verb, etc
        for sentence in data:
            for word_index in range(2, len(sentence[1])):
                # print sentence[1]
                # print word_index
                prev1 = sentence[1][word_index-2]
                prev2 = sentence[1][word_index - 1]
                current = sentence[1][word_index]
                prev = prev1 + prev2

                if self.transition_dict_complex2.get(prev)==None:
                    self.transition_dict_complex2[prev] = self.get_blank_dict()
                self.transition_dict_complex2[prev][current] += 1



        #Calculate emission matrix
        #Creates a dict of all words in train data and the number of times they occurred as a tag = noun,verb,etc.
        for sentence in data:
            for word_index, word in enumerate(sentence[0]):
                if word not in self.word_dict:
                    self.word_dict[word] = self.get_blank_dict()
                self.word_dict[word][sentence[1][word_index]] += 1




    #Creates a dicitionary with each tag and the number of times it occurred in the train dataset.
    def cal_Tag_Occurence(self, data):
        tag_occurence = {}
        tag_list = []
        tag_flist = []
        for i in range(len(data)):
            tag_list.append(data[i][1])
        tag_flist = [item for sublist in tag_list for item in sublist]

        self.tag_occurence = dict(Counter(tag_flist))

        for i in self.tag_occurence:
            self.tag_occurence[i] += smooth_constant
        # raw_input()
        return

    #Calculate pos tag for a word by Simple method
    def get_posTagSimple(self,w):
        # Dict of emission probabily for that word
        word_dict = copy.deepcopy(self.word_dict.get(w))
        # raw_input()
        if word_dict == None:
            return "noun",0.5

        word_eprob_dict = {}

        #Get the total occurence of word
        max_word_occurence = 0
        max_word_occurence = sum(word_dict.values())

        #Compute emission probabilty of all pos tags for the word
        for key,value in word_dict.items():
            max_tag = 0
            max_tag = self.tag_occurence.get(key)
            # *((float(value) / max_tag))
            word_eprob_dict[key] = ((float(value)/max_word_occurence))

        posTag = max(word_eprob_dict, key=word_eprob_dict.get)
        posprob = round(max(word_eprob_dict.values()),2)

        return posTag,posprob

    #Calculate emission prob for a word
    def get_emission(self,word1,tag_row,sentence):
        # Dict of emission probabily for that word
        word = 0
        emission_prob = 0
        emission_value = 0
        word = sentence[word1]
        emission_value_dict = copy.deepcopy(self.word_dict.get(word))
        if emission_value_dict == None:
            return smooth_constant

        emission_value = emission_value_dict.get(tag_row)

        if emission_value == None:
            return smooth_constant


        #Get the total occurence of word
        max_tag_occurence = self.tag_occurence.get(tag_row)
        emission_prob = ((float(emission_value)/ max_tag_occurence))

        return emission_prob


    def get_initial_prob(self,key):
        key_occurred = 0
        initial_key = 0
        initial_prob = 0
        key_occurred = self.tag_occurence[key]
        initial_key = self.initial_state_dict[key]
        initial_prob = ((float(initial_key)/key_occurred))
        return initial_prob


    def get_max_viterbi(self,index,tag,sentence):
        max_transition_viterbi = defaultdict(float)
        viterbi_column = copy.deepcopy(self.viterbi_matrix_dict.get(index))

        if viterbi_column == None:
            return -math.log10(smooth_constant),"noun"

        for i in viterbi_column:
            try:
                max_transition_viterbi[i] = self.viterbi_matrix_dict[index][i][0] - math.log10(self.get_transition_prob_viterbi(i,tag)) - math.log10(self.get_emission(index+1,tag,sentence))
            except:

                max_transition_viterbi[i] = self.viterbi_matrix_dict[index][i][0] - math.log10(self.get_transition_prob_viterbi(i,tag)) - math.log10(smooth_constant)

        posTag = min(max_transition_viterbi, key=max_transition_viterbi.get)
        posprob = min(max_transition_viterbi.values())
        return posprob,posTag


    def get_max_viterbi_complex(self,index,tag,sentence):
        max_transition_complex = defaultdict(float)
        viterbi_column = copy.deepcopy(self.complex_matrix_dict.get(index))

        if viterbi_column == None:
            return -math.log10(smooth_constant)
        for i in viterbi_column:
            try:
                max_transition_complex[i] = self.complex_matrix_dict[index][i] - math.log10(self.get_transition_prob_viterbi(i,tag)) - math.log10(self.get_emission(index+1,tag,sentence))
            except:
                max_transition_complex[i] = self.complex_matrix_dict[index][i] - math.log10(self.get_transition_prob_viterbi(i,tag)) - math.log10(smooth_constant)

        posprob = min(max_transition_complex.values())
        return posprob

    def get_max_viterbi_complex1(self,index,tag,sentence):
        max_transition_complex = defaultdict(float)
        viterbi_column = copy.deepcopy(self.complex_matrix_dict.get(index))

        if viterbi_column == None:
            return -math.log10(smooth_constant)

        for i in viterbi_column:
            try:
                max_transition_complex[i] = self.complex_matrix_dict[index][i] - math.log10(self.get_transition_prob_complex(i,tag)) - math.log10(self.get_emission(index+2,tag,sentence))
            except:

                max_transition_complex[i] = self.complex_matrix_dict[index][i] - math.log10(self.get_transition_prob_complex(i,tag)) - math.log10(smooth_constant)

        # posTag = sum(max_transition_complex, key=max_transition_viterbi.get)
        posprob = min(max_transition_complex.values())
        return posprob


    #Return transition probability given 2 tags
    def get_transition_prob_viterbi(self,i,tag):
        temp = 0
        temp = float(self.transition_dict[i][tag])/self.tag_occurence.get(tag)
        if temp == 0.0:
            return smooth_constant
        else : return temp

    #Return transition probability given 2 tags
    def get_transition_prob_complex(self,i,tag):
        temp = 0
        temp = float(self.transition_dict_complex[i][tag])/self.tag_occurence.get(tag)
        if temp == 0.0:
            return smooth_constant
        else : return temp


    def backTrack_viterbi(self,sentence):
        n = len(sentence)
        max_last_col = self.viterbi_matrix_dict.get(n-1)
        tag = min(max_last_col, key=max_last_col.get)
        tagsequence = []
        tagsequence.insert(0,tag)
        for i in range(n-1,0,-1):
            tag = self.viterbi_matrix_dict[i][tag][1]
            tagsequence.insert(0,tag)

        return tagsequence

    def backTrack_complex(self,sentence):
        n = len(sentence)
        tagsequence = []
        for word in self.complex_matrix_dict:
            temp = min(self.complex_matrix_dict[word], key=self.complex_matrix_dict[word].get)
            tagsequence.append(temp)
        return tagsequence






    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        posTag_Sentence = []
        posTag_Prob = []
        for word in sentence:
             posTag,posprob = self.get_posTagSimple(word)
             posTag_Sentence.append(posTag)
             posTag_Prob.append(posprob)

        return [[posTag_Sentence ], [posTag_Prob , ]]



    def hmm(self, sentence):
        self.viterbi_matrix_dict ={}
        sentence_copy = copy.deepcopy(sentence)
        n = len(sentence)
        temp_dict = {}
        #Initialize a blank viterbi matrix
        for index,word in enumerate(sentence):
            if index not in self.viterbi_matrix_dict:
                self.viterbi_matrix_dict[index] = self.get_blank_dict_viterbi()


        #get the first word in the sentence
        first_word = 0

        #calculate the first column of the matrix
        for tag_row in self.viterbi_matrix_dict[first_word]:
            value = 0
            emission = 0
            value = -math.log10( self.get_initial_prob(tag_row))

            try:
                # print "in try of hmm"
                emission = -math.log10( self.get_emission(first_word,tag_row,sentence))
                # print "emission",emission
                temp = value + emission
                self.viterbi_matrix_dict[first_word][tag_row][0] = temp
                self.viterbi_matrix_dict[first_word][tag_row][1] = "start"
            except:
                # print "sentence",sentence
                # print "value",value
                # print "in except"
                print ""
                # raw_input()


        #calculate the whole viterbi matrix
        for word_col in self.viterbi_matrix_dict:
            if int(word_col)>0:
                for tag_row in self.viterbi_matrix_dict[word_col]:
                    max_transition,posTag = self.get_max_viterbi(word_col-1,tag_row,sentence)
                    if max_transition == 0.0:
                        self.viterbi_matrix_dict[word_col][tag_row][1] = posTag
                        self.viterbi_matrix_dict[word_col][tag_row][0] = -math.log10(smooth_constant)
                    else:
                        self.viterbi_matrix_dict[word_col][tag_row][1] = posTag
                        self.viterbi_matrix_dict[word_col][tag_row][0] = (max_transition)


        Tag_Sequence = 0
        Tag_Sequence = tuple(self.backTrack_viterbi(sentence))
        # print Tag_Sequence
        # raw_input()
        return [  [Tag_Sequence ] , [] ]




    def complex(self, sentence):

        self.complex_matrix_dict = {}
        sentence_copy = copy.deepcopy(sentence)
        n = len(sentence)
        temp_dict = {}
        # Initialize a blank complex matrix
        for index, word in enumerate(sentence_copy):
            if index not in self.complex_matrix_dict:
                self.complex_matrix_dict[index] = self.get_blank_dict()

        # get the first word in the sentence
        first_word = 0

        # calculate the first column of the matrix
        for tag_row in self.complex_matrix_dict[first_word]:
            value = 0
            emission = 0
            value = -math.log10(self.get_initial_prob(tag_row))

            try:
                emission = -math.log10(self.get_emission(first_word, tag_row, sentence_copy))
                temp = value + emission
                self.complex_matrix_dict[first_word][tag_row] = temp
            except:
                print ""

        #Calculate for second col of the matrix
        if len(sentence_copy)== 1:
            Tag_Sequence = tuple(self.backTrack_complex(sentence_copy))
            return [ [ Tag_Sequence], [[0] * len(sentence_copy),] ]

        for tag_row1 in self.complex_matrix_dict[1]:
            max_transition = self.get_max_viterbi_complex(0, tag_row1, sentence_copy)
            if max_transition == 0.0:
                self.complex_matrix_dict[1][tag_row1] = -math.log10(smooth_constant)
            else:
                self.complex_matrix_dict[1][tag_row1] = (max_transition)



        if len(sentence_copy)== 2:
            Tag_Sequence = tuple(self.backTrack_complex(sentence_copy))
            return [ [ Tag_Sequence], [[0] * len(sentence_copy),] ]


        # calculate the whole viterbi matrix
        max_transition = 0
        for word_col in self.complex_matrix_dict:
            if int(word_col) > 1:
                for tag_row2 in self.complex_matrix_dict[word_col]:
                    max_transition1 = self.get_max_viterbi_complex(word_col - 1, tag_row2, sentence_copy)
                    max_transition2 = self.get_max_viterbi_complex1(word_col - 2, tag_row2, sentence_copy)
                    max_transition = max_transition1 + max_transition2
                    if max_transition == 0.0:
                        self.complex_matrix_dict[word_col][tag_row2] = -math.log10(smooth_constant)
                    else:
                        self.complex_matrix_dict[word_col][tag_row2] = (max_transition)


        Tag_Sequence = []
        Complex_prob = []
        Tag_Sequence = tuple(self.backTrack_complex(sentence_copy))
        for word in self.complex_matrix_dict:
            a = round(self.complex_matrix_dict[word][Tag_Sequence[word]]*smooth_constant,7)
            Complex_prob.append(a)


        return [ [ Tag_Sequence], [Complex_prob , ]]
        # return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
