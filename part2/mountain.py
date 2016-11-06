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


def get_random_col():
    return random.randint(0, m-1)


def get_probability(prob_dist, row, col, row1, col1):
    return prob_dist[row1][col1] / (1 + abs(row1 - row))


def get_max_row(edge_weight, col):
    return np.argmax(edge_weight[:, col])


def get_top_row(edge_weight, col, top):
    return edge_weight[:, col].argsort()[-top:][::-1]


def create_sample(prob_dist, edge_weight, rand_col, top_rows):
    sample = np.zeros((m, ), dtype=np.int)
    prev_row = get_max_row(edge_weight, rand_col)
    sample[rand_col] = prev_row
    # Move Ahead
    for col in range(rand_col+1, m):
        max_prob = -1
        max_row = -1
        prev_col = col-1
        for row in get_top_row(edge_weight, col, top_rows):
            cur_prob = get_probability(prob_dist, prev_row, prev_col, row, col)
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
            cur_prob = get_probability(prob_dist, prev_row, prev_col, row, col)
            if max_prob < cur_prob:
                max_prob = cur_prob
                max_row = row
        prev_row = max_row
        sample[col] = max_row
    return sample


def gibbs_sample(iter, edge_weight, top_rows):
    result = np.zeros(shape=(iter, m), dtype=np.int)
    for i in range(0, iter):
        rand_col = get_random_col()
        result[i] = create_sample(prob_dist, edge_weight, rand_col, top_rows)
    return result


# main program
#
# (input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
inp_data = [["test_images/mountain.jpg", "out/mountain_out.jpg"],
                ["test_images/mountain2.jpg", "out/mountain_out2.jpg"],
                ["test_images/mountain3.jpg", "out/mountain_out3.jpg"],
                ["test_images/mountain4.jpg", "out/mountain_out4.jpg"],
                ["test_images/mountain5.jpg", "out/mountain_out5.jpg"],
                ["test_images/mountain6.jpg", "out/mountain_out6.jpg"],
                ["test_images/mountain7.jpg", "out/mountain_out7.jpg"],
                ["test_images/mountain8.jpg", "out/mountain_out8.jpg"],
                ["test_images/mountain9.jpg", "out/mountain_out9.jpg"]]
m = -1
n = -1
iterations = 50
top_rows = 100
# load in image
for i in inp_data:
    input_image = Image.open(i[0])

    # compute edge strength mask
    edge_strengt = edge_strength(input_image)
    imsave('edges.jpg', edge_strengt)

    # You'll need to add code here to figure out the results! For now,
    # just create a horizontal centered line.

    prob_dist = distribution(edge_strengt)
    (n, m) = edge_strengt.shape
    max_ridge = edge_strengt.argmax(axis=0).tolist()
    samples = gibbs_sample(iterations, edge_strengt, top_rows)
    final_ridge = np.zeros((m,), dtype=np.int)
    for j in range(0, m):
        final_ridge[j] = np.bincount(samples[:, j]).argmax()

    # output answer
    imsave(i[1], draw_edge(input_image, max_ridge, (255, 0, 0), 5))
    imsave(i[1], draw_edge(input_image, final_ridge.tolist(), (0, 0, 255), 5))
