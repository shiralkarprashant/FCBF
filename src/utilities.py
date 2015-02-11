#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Prashant Shiralkar on 2015-02-06.

Utility methods to compute information-theoretic concepts such as 
entropy, information gain, symmetrical uncertainty.

"""

import sys
import os
import random 
import numpy as np
import scipy as sp

def entropy(vec, base=2):
	" Returns the empirical entropy H(X) in the input vector."
	_, vec = np.unique(vec, return_counts=True)
	prob_vec = np.array(vec/float(sum(vec)))
	if base == 2:
		logfn = np.log2
	elif base == 10:
		logfn = np.log10
	else:
		logfn = np.log
	return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
	"Returns H(X|Y)."
	uy, uyc = np.unique(y, return_counts=True)
	prob_uyc = uyc/float(sum(uyc))
	cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
	return prob_uyc.dot(cond_entropy_x)
	
def mutual_information(x, y):
	" Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
	return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
	" Returns 'symmetrical uncertainty' - a symmetric mutual information measure."
	return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))
	
if __name__ == '__main__':
	vec1 = np.linspace(1,20,20)
	print "Vec 1:", vec1
	print "Entropy:", entropy(vec1)
	
	vec2 = np.tile([4,5,6,7], 5)
	print "Vec 2:", vec2
	print "Entropy:", entropy(vec2)
	
	mi = mutual_information(vec1, vec2)
	print "Mutual information: {0}".format(mi)
	
	su = symmetrical_uncertainty(vec1, vec2)
	print "Symmetrical uncertainty: {0}". format(su)

