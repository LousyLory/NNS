"""
main wrapper for baselines
"""
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from utils import fvecs_read, ivecs_read

def read_data(full_data_path):
	"""
	grabs files from data path
	"""
	all_files = glob(full_data_path+"/*")
	query_file = [x for x in all_files if "query" in x]
	base_file = [x for x in all_files if "base" in x]

	if "ivecs" in query_file:
		query_vectors = ivecs_read(query_file)
	else:
		query_vectors = fvecs_read(query_file)

	if "ivecs" in base_file:
		base_vectors = ivecs_read(base_file)
	else:
		base_vectors = fvecs_read(base_file)

	return base_vectors, query_vectors


path_to_data = "./data/"
data = "siftsmall"
full_data_path = os.path.join(path_to_data, data)

base_vectors, query_vectors = read_data(full_data_path)
print("data shapes:", base_vectors.shape, query_vectors.shape)
