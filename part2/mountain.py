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


# main program
#
# (input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
def main(inp_data):
    # load in image
    for i in inp_data:
        input_image = Image.open(i[0])

        # compute edge strength mask
        edge_strengt = edge_strength(input_image)
        prob_dist = distribution(edge_strengt)
        imsave('edges.jpg', edge_strengt)

        # You'll need to add code here to figure out the results! For now,
        # just create a horizontal centered line.
        ridge = edge_strengt.argmax(axis=0).tolist()

        # output answer
        imsave(i[1], draw_edge(input_image, ridge, (255, 0, 0), 5))


if __name__ == "__main__":
    inp_data = [["test_images/mountain.jpg", "out/mountain_out.jpg"],
                ["test_images/mountain2.jpg", "out/mountain_out2.jpg"],
                ["test_images/mountain3.jpg", "out/mountain_out3.jpg"],
                ["test_images/mountain4.jpg", "out/mountain_out4.jpg"],
                ["test_images/mountain5.jpg", "out/mountain_out5.jpg"],
                ["test_images/mountain6.jpg", "out/mountain_out6.jpg"],
                ["test_images/mountain7.jpg", "out/mountain_out7.jpg"],
                ["test_images/mountain8.jpg", "out/mountain_out8.jpg"],
                ["test_images/mountain9.jpg", "out/mountain_out9.jpg"]]
    main(inp_data)
