import os
import pickle
import time
import torch
import numpy as np
from tqdm import tqdm
from subprocess import check_call, check_output, CalledProcessError

def solve_concorde_log(executable, directory, name, loc, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.tsp".format(name))
    tour_filename = os.path.join(directory, "{}.tour".format(name))
    output_filename = os.path.join(directory, "{}.concorde.pkl".format(name))
    log_filename = os.path.join(directory, "{}.log".format(name))

    # if True:
    try:
        write_tsplib(problem_filename, loc, name=name)

        with open(log_filename, 'w') as f:
            start = time.time()
            try:
                # Concorde is weird, will leave traces of solution in current directory so call from target dir
                check_call([executable, '-s', '1234', '-x', '-o',
                            os.path.abspath(tour_filename), os.path.abspath(problem_filename)],
                        stdout=f, stderr=f, cwd=directory)
            except CalledProcessError as e:
                # Somehow Concorde returns 255
                assert e.returncode == 255
            duration = time.time() - start

        tour = read_concorde_tour(tour_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None

def load_dataset(filename):

    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_concorde_tour(filename):
    with open(filename, 'r') as f:
        n = None
        tour = []
        for line in f:
            if n is None:
                n = int(line)
            else:
                tour.extend([int(node) for node in line.rstrip().split(" ")])
    assert len(tour) == n, "Unexpected tour length"
    return tour

def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def run_concorde(dataset_path):
    executable = os.path.abspath(os.path.join('concorde_baseline', 'concorde', 'concorde', 'TSP', 'concorde'))
    dataset = torch.load(dataset_path)
    dataset_folder, dataset_file = os.path.split(dataset_path)
    dataset_name = os.path.splitext(dataset_file)[0]
    save_path = os.path.join(dataset_folder, "results")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    len_list = []
    data_list = dataset.__data_list__
    for i, data in tqdm(enumerate(data_list)):
        name = f"{dataset_name}-{i}"
        loc = data.pos.numpy().tolist()
        len, _, _ = solve_concorde_log(executable, save_path, name, loc)
        data.len = torch.tensor(len)
        len_list.append(len)

    return dataset, len_list