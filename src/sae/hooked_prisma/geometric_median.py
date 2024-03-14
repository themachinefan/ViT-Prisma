import torch
from types import SimpleNamespace
import torch
import tqdm

def weighted_average(points, weights):
    weights = weights / weights.sum()
    return (points*weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(median, points, weights):

    norms = torch.linalg.norm(points-median.view(1, -1), dim=1)


    return (norms*weights).sum()

def compute_geometric_median( points:torch.Tensor,
    weights:torch.Tensor=None,
    eps=1e-6,
    maxiter=100,
    ftol=1e-20,
    do_log=False):
    """
    :param points: ``torch.Tensor`` of shape ``(n, d)``
    :param weights: Optional ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
        Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :param do_log: If true will return a log of function values encountered through the course of the algorithm 
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list (None if do_log is false).
    """
    with torch.no_grad():

        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if do_log:
            logs = [objective_value]
        else:
            logs = None 

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points-median.view(1, -1), dim=1)
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if do_log:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )



