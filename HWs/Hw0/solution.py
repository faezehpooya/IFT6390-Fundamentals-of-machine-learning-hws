# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sEqVIVCp0j_hnLhokIWB0dkfsZLjjhVv
"""

import numpy as np

import numpy as np


def make_array_from_list(some_list):
    return np.array(some_list)


def make_array_from_number(num):
    return np.array([num])


class NumpyBasics:
    def add_arrays(self, a, b):
      return a+b

    def add_array_number(self, a, num):
        return a + num

    def multiply_elementwise_arrays(self, a, b):
        return np.multiply(a,b)

    def dot_product_arrays(self, a, b):
        return np.dot(a,b)

    def dot_1d_array_2d_array(self, a, m):
        # consider the 2d array to be like a matrix
        return np.dot(a,m)