import torch
import argparse
import os
from tqdm import tqdm
from concorde.tsp import TSPSolver

FLOAT_SCALE = 10000


def generate_data(num_samples: int, graph_size: int, dataset_name: str):

    points_list = []
    solutions = []
    opt_dists = []

    data_iter = tqdm(range(num_samples), unit="data")
    for i, _ in enumerate(data_iter):
        data_iter.set_description("Generating data points %i/%i" % (i + 1, num_samples))

        points = torch.empty(size=(graph_size, 2)).uniform_(0, 1)

        points_list.append(points)

    # solutions_iter: for tqdm
    solutions_iter = tqdm(points_list, unit="solve")
    for i, points in enumerate(solutions_iter):
        solutions_iter.set_description("Solved %i/%i" % (i + 1, len(points_list)))

        points_scaled = points.numpy() * FLOAT_SCALE
        solver = TSPSolver.from_data(points_scaled[:, 0], points_scaled[:, 1], "EUC_2D")

        sol = solver.solve(time_bound=-1, verbose=False)

        opt_tour, opt_dist = sol.tour, sol.optimal_value / FLOAT_SCALE
        solutions.append(opt_tour)
        opt_dists.append(opt_dist)

    data = {"Points": points_list, "OptTour": solutions, "OptDistance": opt_dists}
    file_name = f"tsp_{graph_size}_{dataset_name}_{num_samples}.pt"
    path = os.path.join(os.getcwd(), file_name)
    torch.save(data, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Optimal Solutions for TSP instance.")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--name", type=str, default="validation")
    args = parser.parse_args()

    generate_data(args.num_samples, args.graph_size, args.name)
