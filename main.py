"""
main wrapper for baselines
"""
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from utils import fvecs_read, ivecs_read
from indexed_structure import compute_online

def read_data(full_data_path):
	"""
	grabs files from data path
	"""
	all_files = glob(full_data_path+"/*")
	query_file = [x for x in all_files if "query" in x]
	base_file = [x for x in all_files if "base" in x]
        
	if "ivecs" in query_file:
		query_vectors = ivecs_read(query_file[0])
	else:
		query_vectors = fvecs_read(query_file[0])

	if "ivecs" in base_file:
		base_vectors = ivecs_read(base_file[0])
	else:
		base_vectors = fvecs_read(base_file[0])

	return base_vectors, query_vectors


path_to_data = "./data/"
data = "siftsmall"
full_data_path = os.path.join(path_to_data, data)

base_vectors, query_vectors = read_data(full_data_path)

# compute indexed structure
method_choices = ["AK", "ORS"]
selected_vectors = compute_online(base_vectors, method=method_choices[0])
