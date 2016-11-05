###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
from pprint import pprint
import copy
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
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
        return 0

    def get_blank_dict(self):
        blank_dict = dict()
        blank_dict[Solver.WORD_DET] = 0
        blank_dict[Solver.WORD_VERB] = 0
        blank_dict[Solver.WORD_ADP] = 0
        blank_dict[Solver.WORD_ADV] = 0
        blank_dict[Solver.WORD_ADJ] = 0
        blank_dict[Solver.WORD_NOUN] = 0
        blank_dict[Solver.WORD_DOT] = 0
        blank_dict[Solver.WORD_PRON] = 0
        blank_dict[Solver.WORD_CONJ] = 0
        blank_dict[Solver.WORD_PRT] = 0
        blank_dict[Solver.WORD_NUM] = 0
        blank_dict[Solver.WORD_X] = 0
        return blank_dict



    def __init__(self):
        self.word_dict = dict()
        self.tag_occurence = dict()
        self.initial_state_dict = ()
        self.transition_dict = dict()
    # Do the training!
    #



    def train(self, data):
        #Calculate the tag occurence :-
        self.cal_Tag_Occurence(data)

        #Calculate initial state matrix :-
        self.initial_state_dict = {}
        for sentence in data:
            tag = ""
            # print sentence[1][0]
            tag = str(sentence[1][0])
            # print type(tag)
            # print tag
            # raw_input()
            if tag not in self.initial_state_dict:
                self.initial_state_dict[tag] = 1
            if tag == Solver.WORD_DET:
                self.initial_state_dict[tag] += 1
            elif tag == Solver.WORD_VERB:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_ADP:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_ADV:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_ADJ:
                self.initial_state_dict[tag] += 1
            elif tag == Solver.WORD_NOUN:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_DOT:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_PRON:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_CONJ:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_PRT:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_NUM:
                self.initial_state_dict[tag] += 1
            elif tag  == Solver.WORD_X:
                self.initial_state_dict[tag] += 1
        # print self.initial_state_dict
        # raw_input()


        # Intitialize Transition matrix
        self.transition_dict = self.get_blank_dict()

        #Calculate transition matrix
        for sentence in data:
            for word_index in range(0, len(sentence[1])-1):
                current_type = sentence[1][word_index]
                next_type = sentence[1][word_index+1]
                if type(self.transition_dict[current_type]) != dict:
                    self.transition_dict[current_type] = self.get_blank_dict()
                self.transition_dict[current_type][next_type] += 1


        #Calculate emission matrix
        for sentence in data:
            for word_index, word in enumerate(sentence[0]):
                if word not in self.word_dict:
                    self.word_dict[word] = self.get_blank_dict()
                self.word_dict[word][sentence[1][word_index]] += 1


                # if sentence[1][word_index] == Solver.WORD_DET:
                #     self.word_dict[word][Solver.WORD_DET] += 1
                # elif sentence[1][word_index] == Solver.WORD_VERB:
                #     self.word_dict[word][Solver.WORD_VERB] += 1
                # elif sentence[1][word_index] == Solver.WORD_ADP:
                #     self.word_dict[word][Solver.WORD_ADP] += 1
                # elif sentence[1][word_index] == Solver.WORD_ADV:
                #     self.word_dict[word][Solver.WORD_ADV] += 1
                # elif sentence[1][word_index] == Solver.WORD_ADJ:
                #     self.word_dict[word][Solver.WORD_ADJ] += 1
                # elif sentence[1][word_index] == Solver.WORD_NOUN:
                #     self.word_dict[word][Solver.WORD_NOUN] += 1
                # elif sentence[1][word_index] == Solver.WORD_DOT:
                #     self.word_dict[word][Solver.WORD_DOT] += 1
                # elif sentence[1][word_index] == Solver.WORD_PRON:
                #     self.word_dict[word][Solver.WORD_PRON] += 1
                # elif sentence[1][word_index] == Solver.WORD_CONJ:
                #     self.word_dict[word][Solver.WORD_CONJ] += 1
                # elif sentence[1][word_index] == Solver.WORD_PRT:
                #     self.word_dict[word][Solver.WORD_PRT] += 1
                # elif sentence[1][word_index] == Solver.WORD_NUM:
                #     self.word_dict[word][Solver.WORD_NUM] += 1
                # elif sentence[1][word_index] == Solver.WORD_X:
                #     self.word_dict[word][Solver.WORD_X] += 1



        # for word in self.word_dict:
        #     pprint(word)
        #     pprint(self.word_dict[word])

        # pprint(self.word_dict['the'])

    def cal_Tag_Occurence(self, data):
        tag_occurence = {}
        tag_list = []
        tag_flist = []
        for i in range(len(data)):
            tag_list.append(data[i][1])
        tag_flist = [item for sublist in tag_list for item in sublist]

        self.tag_occurence = dict(Counter(tag_flist))
        return

    def get_posTag(self,w):
        # Dict of emission probabily for that word
        word_dict = copy.deepcopy(self.word_dict.get(w))
        # print "word_dict",word_dict
        if word_dict == None:
            return "na"

        word_eprob_dict = {}
        # print w
        # print word_dict

        max_tag_occurence_dict = copy.deepcopy(self.tag_occurence)
        # print "max tag occurence",max_tag_occurence_dict
        #Get the total occurence of word
        max_word_occurence = 0
        max_word_occurence = sum(word_dict.values())

        #Compute emission probabilty of all pos tags for the word
        for key,value in word_dict.items():
            max_tag = 0
            max_tag = max_tag_occurence_dict.get(key)
            word_eprob_dict[key] = (float(value)/max_word_occurence) * (float(value)/max_tag)
        # print word_eprob_dict
        # raw_input()
        #Return max emission probability pos tag
        posTag = max(word_eprob_dict, key=word_eprob_dict.get)

        return posTag





    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        posTag_Sentence = ""
        for word in sentence:
             posTag = self.get_posTag(word)
             posTag_Sentence += posTag + " "

             # raw_input()
        posTag_Sentence_list = posTag_Sentence.split()

        return [[posTag_Sentence_list ], [[0] * len(sentence), ]]


    def hmm(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [] ]

    def complex(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]


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
