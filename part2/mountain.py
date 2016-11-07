#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#
"""
(1) a description of how you formulated the problem, including precisely
defining the abstractions (e.g. HMM formulation);
    HMM Formulation:
        - First I choose the initial state as a random column.
        - Later I will consider the point(row) with the maximum edge weight.
        - Emission probability is done initially with probability distribution
        of the edge weight. Edge_weight[Row][Column]/ Sum of EdgeWeight[Column]
        - I am using Gibbs Sampling to get a sample and move forward while
        doing n iterations. (our case 50 iterations)
        - Probability to consider the next point is:
            Prob = Emission_Probability / (Row Diff)
        - The Row difference or distance is indirectly proportional to the
        probability of choosing that row.

(2) a brief description of how your program works;
    Max Ridge will give us the Maximum value at each column. That is
    represented by red. Blue represents gibbs sampling which will calculate
    the probability at the next row compared to the current row. Gibbs sample
    takes number of iterations, edge weight and rows of top edge weight.
    Once it will get the column it will move ahead and behind creating the
    ridge.
    Green ridge will be using the same formula but instead of picking random
    point it will start at a point and traverse. This will also make the
    gibbs sample as just one iteration.

(3) a discussion of any problems, assumptions, simplifications, and/or design
decisions you made;
    This problem assumes that the point given by the user will always lie on
    the ridge so that we traverse only the rows + or - n number of the times.
    For design we used numpy array for each of the sample, After all sample
    gets calculated we select the point which is most frequent in each of the
    column.

(4) answers to any questions asked below in the assignment.
    --

"""
from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import numpy as np
import random
import sys
from copy import deepcopy


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return filtered_y**2


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to
# the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure
# red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(max(y-thickness/2,0),min(y+thickness/2,image.size[1]-1)):
            image.putpixel((x, t), color)
    return image


def distribution(data):
    probability_dist = np.zeros(data.shape)
    (row, col) = data.shape
    for j in range(0, col):
        sum_ = np.sum(data[:, j])
        for i in range(0, row):
            probability_dist[i][j] = data[i][j] / sum_
    return probability_dist


def get_random_col():
    return random.randint(0, m-1)


def get_probability(prob_dist, row, col, row1, col1):
    if abs(row1 - row) > 5:
        return prob_dist[row1][col1] / ((1 + abs(row1 - row)) * 50)
    else:
        return prob_dist[row1][col1] / (1 + abs(row1 - row))


def get_max_row(edge_weight, col):
    return np.argmax(edge_weight[:, col])


def get_top_row(edge_weight, col, top, prev_row, flag):
    # return edge_weight[:, col].argsort()[-top:][::-1]
    if flag:
        return [prev_row-i for i in range(1, top)] + [prev_row] + [prev_row+i
                                                                 for i in
                                                                 range(1, top)]
    else:
        return edge_weight[:, col].argsort()[-top:][::-1]


def create_sample(prob_dist, edge_weight, rand_col, prev_row, top_rows, flag):
    sample = np.zeros((m, ), dtype=np.int)
    sample[rand_col] = prev_row
    cur_row = deepcopy(prev_row)
    # Move Ahead
    for col in range(rand_col+1, m):
        max_prob = -1
        max_row = -1
        prev_col = col-1
        for row in get_top_row(edge_weight, col, top_rows, cur_row, flag):
            cur_prob = get_probability(prob_dist, cur_row, prev_col, row, col)
            if max_prob < cur_prob:
                max_prob = cur_prob
                max_row = row
        cur_row = max_row
        sample[col] = max_row

    cur_row = deepcopy(prev_row)
    # prev_row = get_max_row(edge_weight, rand_col)
    # Move behind
    for col in range(rand_col-1, -1, -1):
        max_prob = -1
        max_row = -1
        prev_col = col - 1
        for row in get_top_row(edge_weight, col, top_rows, cur_row, flag):
            cur_prob = get_probability(prob_dist, cur_row, prev_col, row, col)
            if max_prob < cur_prob:
                max_prob = cur_prob
                max_row = row
        cur_row = max_row
        sample[col] = max_row
    return sample


def gibbs_sample(iter, edge_weight, top_rows, gt_row, gt_col):
    result = np.zeros(shape=(iter, m), dtype=np.int)
    flag = False
    if gt_col != -1:
        flag = True
    for i in range(0, iter):
        rand_col = get_random_col() if gt_col == -1 else int(gt_col)
        prev_row = get_max_row(edge_weight, rand_col) if gt_row == -1 else \
            int(gt_row)
        result[i] = create_sample(prob_dist, edge_weight, rand_col, prev_row,
                                  top_rows, flag)
    return result


# main program
#
gt_row = -1
gt_col = -1
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
m = -1
n = -1
iterations = 20
top_rows = 5
# load in image
inp_data = [[input_filename, output_filename]]
for i in inp_data:
    input_image = Image.open(i[0])

    # compute edge strength mask
    edge_strengt = edge_strength(input_image)
    imsave('edges.jpg', edge_strengt)

    # You'll need to add code here to figure out the results! For now,
    # just create a horizontal centered line.

    prob_dist = distribution(edge_strengt)
    (n, m) = edge_strengt.shape

    # Red Line
    max_ridge = edge_strengt.argmax(axis=0).tolist()

    # Blue Line
    samples = gibbs_sample(iterations, edge_strengt, top_rows, -1, -1)
    final_ridge = np.zeros((m,), dtype=np.int)
    for j in range(0, m):
        final_ridge[j] = np.bincount(samples[:, j]).argmax()

    # Green Line
    green_ridge = gibbs_sample(1, edge_strengt, top_rows, gt_row, gt_col)

    # output answer
    imsave(i[1], draw_edge(input_image, max_ridge, (255, 0, 0), 5))
    imsave(i[1], draw_edge(input_image, final_ridge.tolist(), (0, 0, 255), 5))
    imsave(i[1], draw_edge(input_image, green_ridge[0].tolist(), (0, 255, 0),
                           5))
