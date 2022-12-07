import torch


class FieldFactor:

    def __init__(
            self,
            dof,
            sigma,
    ):
        self.sigma = sigma
        self.dof = dof
        self.K = 1. / (sigma**2)

    def get_error(
            self,
            q_traj,
            field,
            x_traj=None,
            calc_jacobian=True,
            **observations
    ):
        batch, horizon = q_traj.shape[0], q_traj.shape[1]

        if x_traj is not None:
            states = x_traj
        else:
            states = q_traj[:, :, :self.dof].reshape(-1, self.dof)
        error = field.compute_cost(states, **observations).reshape(batch, horizon)

        if calc_jacobian:
            H = -torch.autograd.grad(error.sum(), q_traj)[0]
            error = error.detach()
            error.requires_grad = False
            field.zero_grad()
            return error, H.reshape(batch, horizon, self.dof)
        else:
            return error
