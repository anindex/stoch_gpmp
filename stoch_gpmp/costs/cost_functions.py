
from abc import ABC, abstractmethod
import torch
from torch._vmap_internals import _vmap
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
    def eval(self, trajs, **observation):
        pass

    @abstractmethod
    def get_linear_system(self, trajs, **observation):
        pass



class CostComposite(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        cost_list,
        FK=None,
        tensor_args=None
    ):
        super().__init__(n_dof, traj_len)
        self.cost_list = cost_list
        self.FK = FK
        self.tensor_args = tensor_args

    def eval(self, trajs, **observation):
        trajs = trajs.reshape(-1, self.traj_len, self.dim)
        batch_size = trajs.shape[0]
        x_trajs = None
        if self.FK is not None:  # NOTE(an): only works with SE(3) FK for now
            x_trajs = self.FK(trajs.view(-1, self.dim)[:, :self.n_dof]).reshape(batch_size, self.traj_len, -1, 4, 4)
        costs = 0

        for cost in self.cost_list:
            costs += cost.eval(trajs, x_trajs=x_trajs, **observation)

        return costs

    def get_linear_system(self, trajs, **observation):
        trajs.requires_grad = True
        trajs = trajs.reshape(-1, self.traj_len, self.dim)
        batch_size = trajs.shape[0]
        x_trajs = None
        if self.FK is not None:  # NOTE(an): only works with SE(3) FK for now
            x_trajs = self.FK(trajs.view(-1, self.dim)[:, :self.n_dof]).reshape(batch_size, self.traj_len, -1, 4, 4)
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost in self.cost_list:
            A, b, K = cost.get_linear_system(trajs, x_trajs=x_trajs, **observation)
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())
        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset:offset+dim, offset:offset+dim] = Ks[i]
            offset += dim
        return A, b, K 


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

    def eval(self, trajs, x_trajs=None, **observation):
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
    
    def get_linear_system(self, trajs, x_trajs=None, **observation):
        batch_size = trajs.shape[0]
        A = torch.zeros(batch_size, self.dim * self.traj_len, self.dim * self.traj_len, **self.tensor_args)
        b = torch.zeros(batch_size, self.dim * self.traj_len, 1, **self.tensor_args)
        K = torch.zeros(batch_size, self.dim * self.traj_len, self.dim * self.traj_len, **self.tensor_args)

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(trajs[:, [0]])
        A[:, :self.dim, :self.dim] = H_p
        b[:, :self.dim] = err_p
        K[:, :self.dim, :self.dim] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajs)
        for i in range(self.traj_len - 1):
            A[:, (i+1)*self.dim:(i+2)*self.dim, i*self.dim:(i+1)*self.dim] = H1_gp[[i]]
            A[:, (i+1)*self.dim:(i+2)*self.dim, (i+1)*self.dim:(i+2)*self.dim] = H2_gp[[i]]
            b[:, (i+1)*self.dim:(i+2)*self.dim] = err_gp[:, i]
            K[:, (i+1)*self.dim:(i+2)*self.dim, (i+1)*self.dim:(i+2)*self.dim] = self.gp_prior.Q_inv[[i]]

        return A, b, K


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
            [1, self.traj_len]
        )

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.field is not None:
            err_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs
    
    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            A = torch.zeros(batch_size, (self.traj_len - 1), self.dim * self.traj_len, **self.tensor_args)
            err_obst, H_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=True,
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            for i in range(self.traj_len - 1):
                A[:, i, (i+1)*self.dim:(i+1)*self.dim + self.n_dof] = H_obst[:, i]
            b = err_obst.unsqueeze(-1)
            K = self.obst_factor.K * torch.eye((self.traj_len - 1), **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


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
            [self.traj_len - 1, self.traj_len]   # only take last point
        )

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.field is not None:
            err_obst = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
            )
            w_mat = self.goal_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            A = torch.zeros(batch_size, 1, self.dim * self.traj_len, **self.tensor_args)
            err_goal, H_goal = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=True,
            )
            A[:, :, -self.dim:(-self.dim + self.n_dof)] = H_goal
            b = err_goal.unsqueeze(-1)
            K = self.goal_factor.K * torch.eye(1, **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


class CostGoalPrior(Cost):

    def __init__(
        self,
        n_dof,
        traj_len,
        multi_goal_states=None,  # num_goal x n_dim (pos + vel)
        num_particles_per_goal=None,
        num_samples=None,
        sigma_goal_prior=None,
        tensor_args=None,
    ):
        super().__init__(n_dof, traj_len)
        self.multi_goal_states = multi_goal_states
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior
        self.tensor_args = tensor_args

        self.set_cost_factors()

    def set_cost_factors(self):

        self.multi_goal_prior = []
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal_prior,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.multi_goal_states is not None:
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal * self.num_samples, self.traj_len, self.dim)
            costs = torch.zeros(self.num_goals, self.num_particles_per_goal * self.num_samples, **self.tensor_args)
            for i in range(self.num_goals):
                err_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]], calc_jacobian=False)
                w_mat = self.multi_goal_prior[i].K
                goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
                goal_costs = goal_costs.squeeze()
                costs[i] += goal_costs
            costs = costs.flatten()
        return costs

    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.multi_goal_states is not None:
            npg = self.num_particles_per_goal
            batch_size = npg * self.num_goals
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal, self.traj_len, self.dim)
            A = torch.zeros(batch_size, self.dim, self.dim * self.traj_len, **self.tensor_args)
            b = torch.zeros(batch_size, self.dim, 1, **self.tensor_args)
            K = torch.zeros(batch_size, self.dim, self.dim, **self.tensor_args)
            for i in range(self.num_goals):
                err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
                A[i*npg: (i+1)*npg, :, -self.dim:] = H_g
                b[i*npg: (i+1)*npg] = err_g
                K[i*npg: (i+1)*npg] = self.multi_goal_prior[i].K
        return A, b, K
