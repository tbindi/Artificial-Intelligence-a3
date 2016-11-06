#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import numpy as np
from copy import deepcopy


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
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


def get_init_state(data):
    return data.argmax(axis=0).tolist()


def total_probability(sample_i, prob_dist):
    return sum([prob_dist[sample_i[x]][x] for x in range(0, len(sample_i))])


def get_probability(row, col, prob_dist, state):
    if col > 0 and col < m-1:
        return (prob_dist[row][col] * n * 2) / 10*(1 + abs(state[col-1] - state[col]) + abs(state[col+1] - state[col]))
    elif col == 0:
        return (prob_dist[row][col] * n) / 10*(1 + abs(state[col+1] - state[col]))
    elif col == m-1:
        return (prob_dist[row][col] * n) / 10*(1 + abs(state[col-1] - state[col]))


# data = [ Edge Weight ]
# prob_dist = [ Probability Distribution ]
# sample = [ row_numbers ]
def gibbs_sample(edge_s, prob_dist, sample, T, row_list):
    for t in range(1, T):
        sample.append(sample[t-1])
        max_total_prob = sample[t]["PROB"]
        max_total_state = sample[t]["STATE"]
        for col in range(0, m):
            max_col_prob = -1
            max_state = deepcopy(max_total_state)
            for row in row_list[col]:
                cur_state = deepcopy(max_state)
                cur_state[col] = row
                cur_col_prob = get_probability(row, col, prob_dist, cur_state)
                if cur_col_prob > max_col_prob:
                    max_col_prob = cur_col_prob
                    max_state = cur_state
            cur_total_probability = total_probability(max_state, prob_dist)
            if cur_total_probability > max_total_prob:
                max_total_prob = cur_total_probability
                max_total_state = max_state
        sample[t] = {"PROB": max_total_prob, "STATE": max_total_state}
    return sample


def get_top_column(edge_strengt, top):
    return [col.argsort()[-top:][::-1] for col in edge_strengt.T]


# main program
#
# (input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
inp_data = [["test_images/mountain.jpg", "mountain_out.jpg"],]
                # ["test_images/mountain2.jpg", "out/mountain_out2.jpg"],
                # ["test_images/mountain3.jpg", "out/mountain_out3.jpg"],
                # ["test_images/mountain4.jpg", "out/mountain_out4.jpg"],
                # ["test_images/mountain5.jpg", "out/mountain_out5.jpg"],
                # ["test_images/mountain6.jpg", "out/mountain_out6.jpg"],
                # ["test_images/mountain7.jpg", "out/mountain_out7.jpg"],
                # ["test_images/mountain8.jpg", "out/mountain_out8.jpg"],
                # ["test_images/mountain9.jpg", "out/mountain_out9.jpg"]]
m = -1
n = -1
# load in image
for i in inp_data:
    input_image = Image.open(i[0])

    # compute edge strength mask
    edge_strengt = edge_strength(input_image)
    imsave('edges.jpg', edge_strengt)

    # You'll need to add code here to figure out the results! For now,
    # just create a horizontal centered line.
    init_ridge = get_init_state(edge_strengt)

    prob_dist = distribution(edge_strengt)
    (n, m) = edge_strengt.shape
    sample = [{"PROB": total_probability(init_ridge, prob_dist), "STATE": init_ridge}]
    final_ridge = gibbs_sample(edge_strengt, prob_dist, sample, 10, get_top_column(edge_strengt, 5))

    # output answer
    imsave(i[1], draw_edge(input_image, final_ridge[-1]["STATE"], (255, 0, 0), 5))
