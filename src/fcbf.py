#!/usr/bin/env python
# encoding: utf-8
"""
fcbf.py

Created by Prashant Shiralkar on 2015-02-06.

Fast Correlation-Based Filter (FCBF) algorithm as described in 
Feature Selection for High-Dimensional Data: A Fast Correlation-Based
Filter Solution. Yu & Liu (ICML 2003)

"""

import sys
import os
import numpy as np
import time

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

def getFirstElement(d):
	"""Returns tuple corresponding to first 'unconsidered' feature
	Parameters:
	----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	
	Returns:
	-------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	
	t = np.where(d[:,2]>0)[0]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None

def getNextElement(d, idx):
	"""Returns tuple corresponding to the next 'unconsidered' feature
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature whose next element is required.
		
	Returns:
	--------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	t = np.where(d[:,2]>0)[0]
	t = t[t > idx]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None
	
def removeElement(d, idx):
	"""Returns data with requested feature removed.
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature which needs to be removed.
		
	Returns:
	--------
	d : ndarray
		Same as input, except with specific feature removed.
	"""
	d[idx,2] = 0
	return d

def fcbf(X, y, thresh):
	n = X.shape[1]
	slist = np.zeros((n, 3))
	slist[:, -1] = 1

	# identify relevant features
	t1 = time.time()
	for i in xrange(n):
		slist[i,0] = symmetrical_uncertainty(X[:,i], y)
	print "Time for SU[i,c]: {0}".format(time.time()-t1)
	idx = slist[:,0].argsort()[::-1]
	slist = slist[idx, ]
	slist[:,1] = idx
	slist = slist[slist[:,0]>thresh,:]
	print "Ordered:\n", slist
	
	# identify redundant features among the relevant ones
	cache = {}
	m = len(slist)
	p_su, p, p_idx = getFirstElement(slist)
	print "First:", p_su, p, p_idx
	for i in xrange(m):
		q_su, q, q_idx = getNextElement(slist, p_idx)
		print "Outer q:", q_su, q, q_idx
		if q:
			while q:
				if (p, q) in cache:
					pq_su = cache[(p,q)]
				else:
					pq_su = symmetrical_uncertainty(X[:,p], X[:,q])
					cache[(p,q)] = pq_su
				print pq_su, (pq_su >= q_su)
				if pq_su >= q_su:
					slist = removeElement(slist, q_idx)
					# print slist
				q_su, q, q_idx = getNextElement(slist, q_idx)
				print "Inner q:", q_su, q, q_idx
		p_su, p, p_idx = getNextElement(slist, p_idx)
		print "Next p:", p_su, p, p_idx
		print "========================="
		if not p_idx:
			break
	
	print "\nFinal:\n", slist
	return slist[slist[:,2]>0, :2]
	
def main():
	## ================= PARAMS =================
	fname = '../data/bot_online_dataset.dat'
	delim = '\t'
	thresh = 0.01
	header = True
	## ==========================================
	if os.path.exists(fname):
		try:
			print "Reading file. Please wait ..."
			if header:
				d = np.loadtxt(fname, delimiter=delim, skiprows=1)
			else:
				d = np.loadtxt(fname, delimiter=delim)
		except Exception, e:
			print "Input file loading failed. Please check the file."
			raise e
		print "File read successfully. Dimensions: {0} x {1}".format(d.shape[0], d.shape[1])
		
		X = d[:, :d.shape[1]-1]
		y = d[:,-1]

		sbest = fcbf(X, y, thresh)
		if sbest.shape[0] > 0:
			print "\n#Features selected: {0}".format(len(sbest))
			print "Selected feature indices:\n{0}".format(sbest)
		

if __name__ == '__main__':
	main()

