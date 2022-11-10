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
            x_traj,
            field_func,
            FK=None,
            calc_jacobian=True,
    ):
        batch, horizon = x_traj.shape[0], x_traj.shape[1]

        states = x_traj[:, :, :self.dof].reshape(-1, self.dof)
        if FK is not None:
            states = FK(states)
        error = field_func(states).reshape(batch, horizon)

        if calc_jacobian:
            H = -1. * torch.autograd.grad(error.sum(), x_traj)[0]
            error = error.detach()
            error.requires_grad = False
            return error, H.reshape(batch, horizon, self.dof)
        else:
            return error
