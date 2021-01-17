import math
import torch
import torch.nn as nn
import torch.optim as optim
from environments import TSPEnv
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

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
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def reinforce_train_batch(
    model: nn.Module,
    baseline: nn.Module,
    optimizer: optim.Optimizer,
    batch: Batch,
    epoch: int,
    batch_id: int,
    step: int,
    env,
    logger,
    args,
) -> None:
    batch = batch.to(args.device)
    node_pos = to_dense_batch(batch.pos, batch.batch)[0]
    log_p_s = []
    action_s = []
    reward_s = []
    done = False
    state = env.reset(node_pos)
    model.encode(batch)
    while not done:
        action, log_p = model(state)
        state, reward, done, _ = env.step(action)
        log_p_s.append(log_p)
        action_s.append(action)
        reward_s.append(reward)
    log_p = torch.stack(log_p_s, 1)
    a = torch.stack(action_s, 1)
    # Calculate policy's log_likelihood and reward
    log_likelihood = _calc_log_likelihood(log_p, a)
    # reward is a negative value of tour lenth
    cost = -(reward_s[-1].unsqueeze(-1))
    # let baseline to predict positive value
    bl_val, bl_loss = baseline.eval(batch, cost)
    rl_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = rl_loss + bl_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, args.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(args.log_step) == 0:
        log_values(
            cost=cost,
            grad_norms=grad_norms,
            bl_val=bl_val,
            epoch=epoch,
            batch_id=batch_id,
            step=step,
            log_likelihood=log_likelihood,
            reinforce_loss=rl_loss,
            bl_loss=bl_loss,
            logger=logger,
        )


def _calc_log_likelihood(_log_p, a):
    # Get log_p corresponding to selected actions
    log_p = _log_p.gather(2, a).squeeze(-1)

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    return log_p.sum(1)


def log_values(
    cost, grad_norms, bl_val, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, logger
):
    avg_cost = cost.mean().item()
    bl_cost = bl_val.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}, baseline predict: {}".format(epoch, batch_id, avg_cost, bl_cost)
    )

    print('grad_norm: {}, clipped: {}'.format(grad_norms, grad_norms_clipped))

    # Log values to tensorboard
    logger.add_scalar("avg_cost", avg_cost, step)

    logger.add_scalar('grad_norm', grad_norms[0], step)
    logger.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)

    logger.add_scalar("actor_loss", reinforce_loss.item(), step)
    logger.add_scalar("nll", -log_likelihood.mean().item(), step)

    logger.add_scalar("critic_loss", bl_loss.item(), step)
