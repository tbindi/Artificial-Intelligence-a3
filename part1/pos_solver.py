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

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    WORD_DET = 'det'
    WORD_VERB = 'verb'
    WORD_ADP = 'adp'
    WORD_ADV = 'adv'
    WORD_ADJ = 'adj'
    WORD_NOUN = 'noun'
    WORD_DOT = '.'


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
        return blank_dict

    def __init__(self):
        self.word_dict = dict()

    # Do the training!
    #
    def train(self, data):

        for sentence in data:
            for word_index, word in enumerate(sentence[0]):
                if word not in self.word_dict:
                    self.word_dict[word] = self.get_blank_dict()

                if sentence[1][word_index] == Solver.WORD_DET:
                    self.word_dict[word][Solver.WORD_DET] += 1
                elif sentence[1][word_index] == Solver.WORD_VERB:
                    self.word_dict[word][Solver.WORD_VERB] += 1
                elif sentence[1][word_index] == Solver.WORD_ADP:
                    self.word_dict[word][Solver.WORD_ADP] += 1
                elif sentence[1][word_index] == Solver.WORD_ADJ:
                    self.word_dict[word][Solver.WORD_ADJ] += 1
                elif sentence[1][word_index] == Solver.WORD_ADV:
                    self.word_dict[word][Solver.WORD_ADV] += 1
                elif sentence[1][word_index] == Solver.WORD_NOUN:
                    self.word_dict[word][Solver.WORD_NOUN] += 1
                elif sentence[1][word_index] == Solver.WORD_DOT:
                    self.word_dict[word][Solver.WORD_DOT] += 1

        for word in self.word_dict:
            pprint(word)
            pprint(self.word_dict[word])

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

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

