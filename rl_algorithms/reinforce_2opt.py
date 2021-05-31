import math
from pprint import pprint
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from environments import TSP2OPTEnv
from environments.tsp_2opt import TSP2OPTState
from torch.utils.tensorboard.writer import SummaryWriter
from torch_discounted_cumsum import discounted_cumsum_right
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch


# buffer to store experiences
class Buffer:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def clear_buffer(self):
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.entropies[:]


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def reinforce_train_batch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.grad_scaler.GradScaler,
    batch: Batch,
    epoch: int,
    batch_id: int,
    step: int,
    learn_count: int,
    env: TSP2OPTEnv,
    logger: SummaryWriter,
    args,
) -> int:
    batch = batch.to(args.device)
    node_pos = to_dense_batch(batch.pos, batch.batch)[0]
    buffer = Buffer()
    done = False
    batch_reward = 0
    state = env.reset(T=args.max_num_steps, node_pos=node_pos)
    # embed_data = model.init_embed(batch)
    # node_embeddings, _ = model.encoder(embed_data)
    while not done:
        count = 0
        with amp.autocast_mode.autocast(enabled=True):
            embed_data = model.init_embed(batch)
            node_embeddings, _ = model.encoder(embed_data)
        while count < args.horizon and not done:
            with amp.autocast_mode.autocast(enabled=True):
                action, log_p, value  = model(state, node_embeddings, embed_data.batch)
            state, reward, done, _ = env.step(action.squeeze())
            batch_reward += reward
            buffer.actions.append(action)
            buffer.log_probs.append(log_p)
            buffer.rewards.append(reward)
            buffer.values.append(value)
            buffer.entropies.append(_)
            count += 1
        learn_count = update_model(optimizer, scaler, buffer, state, done, epoch, count, learn_count, step, logger, args)

    logger.add_scalar("batch_rewards_train", batch_reward.mean().item(), step)

    return learn_count


def update_model(
    optimizer: optim.Optimizer,
    scaler: amp.grad_scaler.GradScaler,
    buffer: Buffer,
    state: TSP2OPTState,
    done: bool,
    epoch: int,
    count: int,
    learn_count: int,
    global_step: int,
    logger: SummaryWriter,
    args,
):

    rewards = torch.stack(buffer.rewards, dim=0)  # [horizon, batch_size, 1]
    returns = discounted_return(rewards, args.gamma, count)  # [horizon, batch_size, 1]
    if not args.no_norm_return:
        r_mean = returns.mean()
        r_std = returns.std()
        eps = torch.finfo(torch.float).eps  # small number to avoid div/0
        returns = (returns - r_mean) / (r_std + eps)
    values = torch.stack(buffer.values, dim=0)  # [horizon, batch_size, 1]
    advantages = (returns - values).detach()  # [horizon, batch_size, 1]

    logps = torch.stack(buffer.log_probs, dim=0)  # [horizon, batch_size, 2, graph_size]
    actions = torch.stack(buffer.actions, dim=0)  # [horizon, batch_size, 2, 1]
    log_likelihood = logps.gather(-1, actions).squeeze(-1)  # [horizon, batch_size, 2]
    log_likelihood = log_likelihood.mean(2).unsqueeze(2)  # [horizon, batch_size, 1]

    entropies = log_p_to_entropy(logps).mean(2).unsqueeze(2)  # [horizon, batch_size, 1]

    p_loss = (-log_likelihood * advantages).mean()
    v_loss = args.value_beta * (returns - values).pow(2).mean()
    e_loss = (0.9 ** (epoch + 1)) * args.entropy_beta * entropies.sum(0).mean()
    r_loss = -e_loss + v_loss
    loss = p_loss + r_loss

    optimizer.zero_grad()
    scaler.scale(p_loss).backward(retain_graph=True)
    # scaler.unscale_(optimizer)
    grad_norms = clip_grad_norms(optimizer.param_groups) #, args.max_grad_norm)
    scaler.scale(r_loss).backward(retain_graph=False)
    scaler.step(optimizer)
    scaler.update()

    buffer.clear_buffer()
    log_values(
        cost=state.best_tour_len,
        grad_norms=grad_norms,
        done=done,
        epoch=epoch,
        global_step=global_step,
        learn_count=learn_count,
        p_loss=p_loss,
        v_loss=v_loss,
        e_loss=e_loss,
        loss=loss,
        returns=returns.mean(),
        value=values.mean(),
        entropy=entropies.detach().mean(),
        logger=logger,
        args=args,
    )

    learn_count += 1

    return learn_count


def discounted_return(rewards: torch.Tensor, gamma: int, count: int):
    assert rewards.dim() == 3
    assert rewards.size(0) == count
    assert rewards.size(2) == 1

    # transpose `rewards` as function `discounted_cumsum` apply on second dim
    # squeeze `rewards` as function `discounted_cumsum` only work for 2-dim-tensor
    returns = discounted_cumsum_right(rewards.squeeze(2).T, gamma).T

    return returns.unsqueeze(2)


def log_p_to_entropy(log_probs):
    min_real = torch.finfo(log_probs.dtype).min
    clamped_log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs.exp() * clamped_log_probs

    return -p_log_p.sum(-1)


def log_values(
    cost,
    grad_norms,
    done,
    epoch,
    global_step,
    learn_count,
    p_loss,
    v_loss,
    e_loss,
    loss,
    returns,
    value,
    entropy,
    logger: SummaryWriter,
    args,
):
    avg_len = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms
    if done:
        # Log values to screen
        print("epoch: {}, learn_count: {}, avg_best_cost: {}".format(epoch, learn_count, avg_len))
        print("grad_norm: {}, clipped: {}".format(grad_norms, grad_norms_clipped))
        logger.add_scalar("tour_len_train", avg_len, global_step)
        wandb.log({"tour_len_train": avg_len})

    # Log values to tensorboard

    logger.add_scalar("grad_norm", grad_norms[0], learn_count)
    logger.add_scalar("grad_norm_clipped", grad_norms_clipped[0], learn_count)

    logger.add_scalar("loss_policy", p_loss.item(), learn_count)
    logger.add_scalar("loss_baseline", v_loss.item(), learn_count)
    logger.add_scalar("loss_entropy", e_loss.item(), learn_count)
    logger.add_scalar("loss", loss.item(), learn_count)
    logger.add_scalar("env_returns", returns.item(), learn_count)
    logger.add_scalar("predict_value", value.item(), learn_count)

    wandb.log(
        {
            "grad_norm": grad_norms[0],
            "loss_policy": p_loss.item(),
            "loss_baseline": v_loss.item(),
            "loss_entropy": e_loss.item(),
            "entropy":entropy.item(),
        }
    )
