import torch
import torch.nn as nn
import torch.optim as optim
from environments import TSPEnv
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch


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
    _log_p = torch.stack(log_p_s, 1)
    a = torch.stack(action_s, 1)
    # Calculate policy's log_likelihood and reward
    log_likelihood = _calc_log_likelihood(_log_p, a)
    # reward is a negative value of tour lenth
    reward = -(reward_s[-1].unsqueeze(-1))
    # let baseline to predict positive value
    bl_val, bl_loss = baseline.eval(batch, reward)
    rl_loss = ((reward - bl_val) * log_likelihood).mean()
    loss = rl_loss + bl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if step % int(args.log_step) == 0:
        log_values(
            cost=-reward,
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
    cost, bl_val, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, logger
):
    avg_cost = cost.mean().item()
    bl_cost = bl_val.mean().item()

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}, baseline predict: {}".format(epoch, batch_id, avg_cost, bl_cost)
    )

    # Log values to tensorboard
    logger.add_scalar("avg_cost", avg_cost, step)

    logger.add_scalar("actor_loss", reinforce_loss.item(), step)
    logger.add_scalar("nll", -log_likelihood.mean().item(), step)

    logger.add_scalar("critic_loss", bl_loss.item(), step)
