import math
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from environments import TSP2OPTEnv
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch


# buffer to store experiences
class Buffer:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

    def clear_buffer(self):
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]


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
    batch: Batch,
    epoch: int,
    batch_id: int,
    step: int,
    env: TSP2OPTEnv,
    logger,
    args,
) -> None:
    batch = batch.to(args.device)
    node_pos = to_dense_batch(batch.pos, batch.batch)[0]
    buffer = Buffer()
    done = False
    state = env.reset(T=args.max_num_steps, node_pos=node_pos)
    # embed_data = model.init_embed(batch)
    # node_embeddings, _ = model.encoder(embed_data)
    p_losses = []
    v_losses = []
    e_losses = []
    grad_norms = None
    batch_reward = 0
    batch_value = 0
    while not done:
        count = 0
        embed_data = model.init_embed(batch)
        node_embeddings, _ = model.encoder(embed_data)
        while count < args.horizon and not done:
            action, log_p, value = model(state, node_embeddings, embed_data.batch)
            state, reward, done, _ = env.step(action.squeeze())
            buffer.actions.append(action)
            buffer.log_probs.append(log_p)
            buffer.rewards.append(reward)
            buffer.values.append(value)
            batch_reward += reward.mean().item()
            batch_value += value.mean().item()
            count += 1
        p_loss, v_loss, e_loss, grad_norms = accumulate_loss(optimizer, buffer, epoch, args)
        p_losses.append(p_loss)
        v_losses.append(v_loss)
        e_losses.append(e_loss)

    p_loss = torch.stack(p_losses).mean()
    v_loss = torch.stack(v_losses).mean()
    e_loss = torch.stack(e_losses).mean()

    # Logging
    if step % int(args.log_step) == 0:
        log_values(
            cost=state.best_tour_len,
            grad_norms=grad_norms,
            epoch=epoch,
            batch_id=batch_id,
            step=step,
            p_loss=p_loss,
            v_loss=v_loss,
            e_loss=e_loss,
            reward=batch_reward,
            value=batch_value,
            logger=logger,
            args=args,
        )


def accumulate_loss(optimizer: optim.Optimizer, buffer: Buffer, epoch: int, args):

    returns = []
    R = torch.zeros_like(buffer.rewards[0])
    for i in reversed(range(len(buffer.rewards))):
        R = buffer.rewards[i] + args.gamma * R
        returns.insert(0, R)

    returns = torch.stack(returns, dim=0).detach()  # [horizon, batch_size, 1]
    values = torch.stack(buffer.values, dim=0)  # [horizon, batch_size, 1]
    advantages = (returns - values).detach()  # [horizon, batch_size, 1]

    logps = torch.stack(buffer.log_probs, dim=0)  # [horizon, batch_size, 2, graph_size]
    actions = torch.stack(buffer.actions, dim=0)  # [horizon, batch_size, 2, 1]
    log_likelihood = logps.gather(-1, actions).squeeze(-1)  # [horizon, batch_size, 2]
    log_likelihood = log_likelihood.mean(2).unsqueeze(2)  # [horizon, batch_size, 1]

    entropies = log_p_to_entropy(logps).mean(2)  # [horizon, batch_size]

    p_loss = (-log_likelihood * advantages).mean()
    v_loss = args.value_beta * (returns - values).pow(2).mean()
    e_loss = (0.9 ** (epoch + 1)) * args.entropy_beta * entropies.sum(0).mean()

    loss = p_loss + v_loss - e_loss

    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    grad_norms = clip_grad_norms(optimizer.param_groups, args.max_grad_norm)

    optimizer.step()

    buffer.clear_buffer()

    return p_loss.detach(), v_loss.detach(), e_loss.detach(), grad_norms


def log_p_to_entropy(log_probs):
    min_real = torch.finfo(log_probs.dtype).min
    clamped_log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs.exp() * clamped_log_probs

    return -p_log_p.sum(-1)


def log_values(
    cost, grad_norms, epoch, batch_id, step, p_loss, v_loss, e_loss, reward, value, logger: SummaryWriter, args,
):
    avg_len = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print("epoch: {}, train_batch_id: {}, avg_best_cost: {}".format(epoch, batch_id, avg_len))

    print("grad_norm: {}, clipped: {}".format(grad_norms, grad_norms_clipped))

    # Log values to tensorboard
    logger.add_scalar("avg_len", avg_len, step)

    logger.add_scalar("grad_norm", grad_norms[0], step)
    logger.add_scalar("grad_norm_clipped", grad_norms_clipped[0], step)

    logger.add_scalar("actor_loss", p_loss.item(), step)
    logger.add_scalar("baseline_loss", v_loss.item(), step)
    logger.add_scalar("entropy_loss", e_loss.item(), step)
    logger.add_scalar("env_reward", reward, step)
    logger.add_scalar("predict_value", value, step)
