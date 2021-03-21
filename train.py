import json
import math
import os
import pprint as pp
import time
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from args import get_args
from datasets.tsp import TSPDataset
from environments.tsp import TSPEnv
from models.tsp_agent import TSPAgent
from models.tsp_baseline import CriticBaseline, ExponentialBaseline, RolloutBaseline
from rl_algorithms.reinforce import reinforce_train_batch


def warmup_baseline(baseline, dataset, env, optimizer, args):
    print(f"Warmup Critic Baseline with training dataset")
    for batch in tqdm(DataLoader(dataset, args.warmup_batch_size)):
        batch = batch.to(args.device)
        node_pos = to_dense_batch(batch.pos, batch.batch)[0]
        done = False
        reward = None
        env.reset(node_pos)
        while not done:
            a = env.random_action()
            _, reward, done, _ = env.step(a)

        cost = -reward.unsqueeze(-1)
        _, bl_loss = baseline.eval(batch, cost)

        optimizer.zero_grad()
        bl_loss.backward()
        optimizer.step()


def validate(model, dataset, env, args):
    # Validate
    print("Validating...")
    cost = rollout(model, dataset, env, args)
    assert cost.size(0) == len(dataset)
    avg_cost = -cost.mean()
    print("Validation overall avg_cost: {} +- {}".format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, env, args):
    # Put in greedy evaluation mode!

    model.set_decode_type("greedy")
    model.eval()

    def eval_model_bat(bat):
        node_pos = to_dense_batch(bat.pos, bat.batch)[0]
        reward_s = []
        done = False
        state = env.reset(node_pos)
        embed_data = model.init_embed(bat)
        node_embeddings, graph_feat = model.encoder(embed_data)
        fixed = model.precompute_fixed(node_embeddings, graph_feat)
        while not done:
            action, _ = model(state, fixed)
            state, reward, done, _ = env.step(action)
            reward_s.append(reward)
        return reward_s[-1].cpu()

    return torch.cat(
        [eval_model_bat(bat.to(args.device)) for bat in DataLoader(dataset, batch_size=args.eval_batch_size)], 0,
    )


if __name__ == "__main__":
    args = get_args(None)
    # Pretty print the run args
    pp.pprint(vars(args))

    # Set the random seed
    torch.manual_seed(args.seed)

    # Optionally configure tensorboard
    tb_logger = SummaryWriter(os.path.join(args.log_dir, "{}_{}".format(args.problem, args.graph_size), args.run_name))
    tb_logger.add_hparams(vars(args), {"date": np.array((datetime.today().day))}, run_name=args.run_name)

    os.makedirs(args.save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=True)

    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    load_data = None
    # Load data from load_path
    if args.load_path:
        load_path = args.load_path
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch.load(load_path)

    env = TSPEnv()
    model = TSPAgent(args).to(args.device)
    baseline = ExponentialBaseline(args).to(args.device)
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": args.lr_model}]
        # + [{"params": baseline.parameters(), "lr": args.lr_critic}]
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)

    # Load saved data
    if load_data:
        model.load_state_dict(load_data["model"])
        baseline.load_state_dict(load_data["baseline"])
        optimizer.load_state_dict(load_data["optimizer"])
        torch.set_rng_state(load_data["rng_state"])
        if args.device is torch.device("cuda"):
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
    if args.load_path:
        epoch_resume = int(os.path.splitext(os.path.split(args.load_path)[-1])[0].split("-")[1])
        args.epoch_start = epoch_resume + 1
    val_dataset = TSPDataset(size=args.val_size, args=args, load_path=args.val_dataset)

    if args.eval_only:
        validate(model, val_dataset, env, args)
    else:
        for epoch in range(args.epoch_start, args.epoch_start + args.n_epochs):
            print(
                "Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]["lr"], args.run_name)
            )
            step = epoch * (args.epoch_size // args.batch_size)
            start_time = time.time()
            tb_logger.add_scalar("learnrate_pg0", optimizer.param_groups[0]["lr"], step)
            training_dataset = TSPDataset(size=args.epoch_size, args=args, load_path=args.train_dataset)
            training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size)

            if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                warmup_baseline(baseline, training_dataset, env, optimizer, args)

            model.train()
            model.set_decode_type(args.decode_type)
            for batch_id, batch in enumerate(tqdm(training_dataloader, disable=args.no_progress_bar)):
                reinforce_train_batch(
                    model, baseline, optimizer, batch, epoch, batch_id, step, env, tb_logger, args,
                )
                step += 1

            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))))

            if (args.checkpoint_epochs != 0 and epoch % args.checkpoint_epochs == 0) or epoch == args.n_epochs - 1:
                print("Saving model and state...")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        "baseline": baseline.state_dict(),
                    },
                    os.path.join(args.save_dir, "epoch-{}.pt".format(epoch)),
                )

            avg_reward = validate(model, val_dataset, env, args)
            tb_logger.add_scalar("val_avg_reward", avg_reward, step)

            # lr_scheduler should be called at end of epoch
            lr_scheduler.step()
