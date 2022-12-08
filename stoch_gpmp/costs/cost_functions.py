
from abc import ABC, abstractmethod
import torch
from stoch_gpmp.costs.factors.gp_factor import GPFactor
from stoch_gpmp.costs.factors.unary_factor import UnaryFactor
from stoch_gpmp.costs.factors.field_factor import FieldFactor



class Cost(ABC):
    def __init__(self, n_dof, traj_len):
        self.n_dof = n_dof
        self.dim = 2 * n_dof  # Pos + Vel
        self.traj_len = traj_len

    def set_cost_factors(self):
        pass

    @abstractmethod
    def eval(self, trajs, observation=None):
        pass



class CostComposite(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        cost_func_list,
        FK=None,
    ):
        super().__init__(n_dof, traj_len)
        self.cost_func_list = cost_func_list
        self.FK = FK

    def eval(self, trajs, observation=None):
        trajs = trajs.reshape(-1, self.traj_len, self.dim)
        batch_size = trajs.shape[0]
        x_trajs = None
        if self.FK is not None:  # NOTE(an): only works with SE(3) FK for now
            x_trajs = self.FK(trajs.view(-1, self.dim)[:, :self.n_dof]).reshape(batch_size, self.traj_len, -1, 4, 4)
        costs = 0

        for cost_func in self.cost_func_list:
            costs += cost_func(trajs, x_trajs=x_trajs, observation=observation)

        return costs


class CostGP(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        start_state,
        dt,
        sigma_params,
        tensor_args,
    ):
        super().__init__(n_dof, traj_len)
        self.start_state = start_state
        self.dt = dt

        self.sigma_start = sigma_params['sigma_start']
        self.sigma_gp = sigma_params['sigma_gp']
        self.tensor_args = tensor_args

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

    def eval(self, trajs, x_trajs=None, observation=None):
        # trajs = trajs.reshape(-1, self.traj_len, self.dim)
        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0] # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        return costs


class CostCollision(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        field=None,
        sigma_coll=None,
        tensor_args=None,
    ):
        super().__init__(n_dof, traj_len)
        self.field = field
        self.sigma_coll = sigma_coll
        self.tensor_args = tensor_args

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.obst_factor = FieldFactor(
            self.n_dof,
            self.sigma_coll,
        )

    def eval(self, trajs, x_trajs=None, observation=None):
        costs = 0
        if self.field is not None:
            if x_trajs is not None:
                x_trajs = x_trajs[:, 1:]
            err_obst = self.obst_factor.get_error(
                trajs[:, 1:, :self.n_dof],
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs


class CostGoal(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        field=None,
        sigma_goal=None,
        tensor_args=None,
    ):
        super().__init__(n_dof, traj_len)
        self.field = field
        self.sigma_goal = sigma_goal
        self.tensor_args = tensor_args

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.goal_factor = FieldFactor(
            self.n_dof,
            self.sigma_goal,
        )

    def eval(self, trajs, x_trajs=None, observation=None):
        costs = 0
        if self.field is not None:
            if x_trajs is not None:
                x_trajs = x_trajs[:, [-1]]
            err_obst = self.goal_factor.get_error(
                trajs[:, [-1], :self.n_dof],  # only take last point
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
            )
            w_mat = self.goal_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs
