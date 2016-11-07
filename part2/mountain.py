#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import numpy as np
import random
import sys


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


def get_probability(prob_dist, row, col, row1, col1, edge_weight, flag):
    if flag:
        return prob_dist[row1][col1] / ((1 + abs(row1 - row)) * (1 + abs(
            (edge_weight[row1][col1])/np.sum(edge_weight[:, col1]) -
            (edge_weight[row][col])/np.sum(edge_weight[:, col]))))
    else:
        return prob_dist[row1][col1] / (1 + abs(row1 - row))


def get_max_row(edge_weight, col):
    return np.argmax(edge_weight[:, col])


def get_top_row(edge_weight, col, top):
    return edge_weight[:, col].argsort()[-top:][::-1]


def create_sample(prob_dist, edge_weight, rand_col, prev_row, top_rows, flag):
    sample = np.zeros((m, ), dtype=np.int)
    sample[rand_col] = prev_row
    # Move Ahead
    for col in range(rand_col+1, m):
        max_prob = -1
        max_row = -1
        prev_col = col-1
        for row in get_top_row(edge_weight, col, top_rows):
            cur_prob = get_probability(prob_dist, prev_row, prev_col, row,
                                       col, edge_weight, flag)
            if max_prob < cur_prob:
                max_prob = cur_prob
                max_row = row
        prev_row = max_row
        sample[col] = max_row

    prev_row = get_max_row(edge_weight, rand_col)
    # Move behind
    for col in range(rand_col-1, -1, -1):
        max_prob = -1
        max_row = -1
        prev_col = col - 1
        for row in get_top_row(edge_weight, col, top_rows):
            cur_prob = get_probability(prob_dist, prev_row, prev_col, row,
                                       col, edge_weight, flag)
            if max_prob < cur_prob:
                max_prob = cur_prob
                max_row = row
        prev_row = max_row
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
iterations = 30
top_rows = 100
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
