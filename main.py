"""
main wrapper for baselines
"""
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from utils import fvecs_read, ivecs_read
from indexed_structure import compute_online
import argparse

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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path_to_data", default="./data", help="path where all data is stored", type=str)
    parser.add_argument("--dataset", metavar="NAME", help="the dataset to load training points from", default="siftsmall", type=str, choices=["siftsmall"])
    parser.add_argument("-k", "--count", default=1, type=positive_int, help="the number of near neighbours to search for")
    parser.add_argument("--method", default="AK", type=str, choices=["AK", "ORS"]
    args = parser.parse_args()

    path_to_data = args.path_to_data
    data = args.dataset
    full_data_path = os.path.join(path_to_data, data)

    base_vectors, query_vectors = read_data(full_data_path)

    # compute indexed structure
    method_choice = args.method
    selected_vectors = compute_online(base_vectors, method=method_choice)

    return None

if name == "__main__":
    main()
