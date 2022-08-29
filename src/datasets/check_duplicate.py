import argparse
import hashlib


# Revised from https://github.com/VHellendoorn/Code-LMs/blob/main/Data/deduplicate.py
import json


def _get_hash(file=None, bytes=None):
    if file:
        bytes = open(file, 'rb').read()
    hash = hashlib.sha256(bytes).hexdigest()
    return hash, bytes


def check_polycode_duplicate(dataset):
    with open(dataset, 'r') as f:
        files = f.readlines()

    map_hash_to_files = {}

    for num, file in enumerate(files):
        #  for some reason certain files do not have repo name, or might be missing has values
        try:
            language, repo_owner, project = file.split("__")[0], file.split("__")[1], file.split("__")[2]
            raw_file_path = "__".join(file.split("__")[3:]).strip()
            raw_file_path, hash_v = raw_file_path.split()[0], raw_file_path.split()[-1]
            map_hash_to_files[hash_v] = repo_owner + "__" + project + "__" + raw_file_path
        except:
            print("Malformed Line {} : {}".format(num, file))

    # load json file
    with open("../Defects4j/file_hash.json", 'r') as f:
        d4j_file_hash = json.load(f)

    for bug, s in d4j_file_hash.items():
        for d4j_hash in s['buggy']:
            if d4j_hash['hash'] in map_hash_to_files:
                print("Found in polycode dataset: {} Defects4j bug: {} Defects4j file: {}"
                      .format(map_hash_to_files[d4j_hash['hash']], bug, d4j_hash['path']))
        for d4j_hash in s['fixed']:
            if d4j_hash['hash'] in map_hash_to_files:
                print("Found in polycode dataset: {} Defects4j patch: {} Defects4j file: {}"
                      .format(map_hash_to_files[d4j_hash['hash']], bug, d4j_hash['path']))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset to check, currently support: polycode')
    parser.add_argument("--location", type=str,
                        help="Location of the dataset files")
    args = parser.parse_args()
    if args.dataset == 'polycode':
        check_polycode_duplicate(args.location)
    else:  # TODO
        raise NotImplementedError


if __name__ == '__main__':
    main()
