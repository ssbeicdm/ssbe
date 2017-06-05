# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

#Compute the average cosine similarity withen a list of vectors
def avg_inner_sim(lst):
	sum_sim = 0
	compare_count = 0
	count = len(lst)
	for i in range(count-1):
		sim = cosine_similarity(lst[i].reshape(1,-1), lst[i+1:count])
		compare_count += count - (i+1)
		sum_sim += np.sum(sim)
	return	sum_sim / compare_count

#Take a list and randomly extract negative samples (index > positive samples maximum index)
#The expected list has the positive samples in lst[:count_pos+1] and negatives in [count_pos:]
#Then, compute the average inner similarity of the negative samples
#This runs n times and returns the average
def n_neg_sampling_avg_inner_sim(lst, count_pos, n=100):
	nb_neg_samples = count_pos # for balancing the number positive and negative samples
	sum_neg = 0
	# just in case we have less negative examples than the positives
	if len(range(count_pos + 1, len(lst))) < nb_neg_samples:
		nb_neg_samples = len(range(count_pos + 1, len(lst)))
	for i in range(n):
		rand_ind = random.sample(range(count_pos + 1, len(lst)), nb_neg_samples)
		rand_lst = lst[rand_ind]
		avg_neg = avg_inner_sim(rand_lst)
		sum_neg += 	avg_neg
	return sum_neg / n