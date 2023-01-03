import torch
from stoch_gpmp.costs.factors.mp_priors_multi import MultiMPPrior
from stoch_gpmp.costs.factors.gp_factor import GPFactor
from stoch_gpmp.costs.factors.unary_factor import UnaryFactor
from stoch_gpmp.costs.factors.field_factor import FieldFactor


# TODO(an):  add interface for cost factors!!
class StochGPMP:

    def __init__(
            self,
            num_particles_per_goal,
            num_samples,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            step_size=1.,
            temp=1.,
            start_state=None,
            multi_goal_states=None,
            initial_particle_means=None,
            cost=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            seed=None,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        if seed is not None:
            torch.manual_seed(seed)

        self.n_dof = n_dof
        self.d_state_opt = 2 * self.n_dof
        self.dt = dt

        self.traj_len = traj_len
        self.goal_directed = (multi_goal_states is not None)
        if not self.goal_directed:  # NOTE(an): if xere is no goal, we assume xere is at least one solution
            self.num_goals = 1
        else:
            assert multi_goal_states.dim() == 2
            self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.sigma_start_init = sigma_start_init
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal_init = sigma_goal_init
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp_init = sigma_gp_init
        self.sigma_gp_sample = sigma_gp_sample
        self.start_states = start_state # position + velocity
        self.multi_goal_states = multi_goal_states # position + velocity
        self.cost = cost

        self._mean = None
        self._weights = None
        self._sample_dist = None

        self.reset(start_state, multi_goal_states, initial_particle_means=initial_particle_means)

    def set_prior_factors(self):

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_init,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_init = []
        if self.goal_directed:
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_init,    # NOTE(sasha) Assume same goal Cov. for now
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_sample,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sigma_gp_sample,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_sample = []
        if self.goal_directed:
            for i in range(self.num_goals):
                self.multi_goal_prior_sample.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_sample,   # NOTE(sasha) Assume same goal Cov. for now
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

    def get_prior_dist(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
    ):

        return MultiMPPrior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # NOTE(sasha) Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
            initial_particle_means=None,
    ):

        if start_state is not None:
            self.start_state = start_state.detach().clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.detach().clone()

        self.set_prior_factors()

        if initial_particle_means is not None:
            self.particle_means = initial_particle_means
        else:
            # Initialization particles from prior distribution
            self._init_dist = self.get_prior_dist(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if self.goal_directed else None,
                self.start_state,
                goal_states=self.multi_goal_states,
            )
            self.particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
            del self._init_dist  # free memory
        self.particle_means = self.particle_means.flatten(0, 1)

        # Sampling distributions
        self._sample_dist = self.get_prior_dist(
            self.start_prior_sample.K,
            self.gp_prior_sample.Q_inv[0],
            self.multi_goal_prior_sample[0].K if self.goal_directed else None,
            self.start_state,
            particle_means=self.particle_means,
            goal_states=self.multi_goal_states
        )
        self.Sigma_inv = self._sample_dist.Sigma_inv
        self.state_samples = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

    def _get_costs(self, **observation):

        costs = self.cost.eval(self.state_samples, **observation).reshape(self.num_particles, self.num_samples)

        # Add cost from importance-sampling ratio
        V  = self.state_samples.view(-1, self.num_samples, self.traj_len * self.d_state_opt)  # flatten trajectories
        U = self.particle_means.view(-1, 1, self.traj_len * self.d_state_opt)
        costs += self.temp * (V @ self.Sigma_inv @ U.transpose(1, 2)).squeeze(2)
        return costs

    def sample_and_eval(self, **observation):
        # TODO: update prior covariance with new goal location

        # Sample state-trajectory particles
        self.state_samples = self._sample_dist.sample(self.num_samples).to(
            **self.tensor_args)

        # Evaluate costs
        costs = self._get_costs(**observation)

        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]

        position_seq_mean = self.particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self.particle_means[..., -self.n_dof:].clone()

        return (
            velocity_seq,
            position_seq,
            velocity_seq_mean,
            position_seq_mean,
            costs,
        )

    def _update_distribution(self, costs, traj_samples):

        self._weights = torch.softmax( -costs / self.temp, dim=1)
        self._weights = self._weights.reshape(-1, self.num_samples, 1, 1)

        self.particle_means.add_(
            self.step_size * (
                self._weights * (traj_samples - self.particle_means.unsqueeze(1))
            ).sum(1)
        )
        self._sample_dist.set_mean(self.particle_means.view(self.num_particles, -1))

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            with torch.no_grad():
                (control_samples,
                 state_trajectories,
                 control_particles,
                 state_particles,
                 costs,) = self.sample_and_eval(**observation)

                self._update_distribution(costs, self.state_samples)

        self._recent_control_samples = control_samples
        self._recent_control_particles = control_particles
        self._recent_state_trajectories = state_trajectories
        self._recent_state_particles = state_particles
        self._recent_weights = self._weights

        return (
            state_particles,
            control_particles,
            state_trajectories,
            control_samples,
            costs,
        )

    def _get_traj(self, mode='best'):
        if mode == 'best':
            #TODO: Fix for multi-particles
            particle_ind = self._weights.argmax()
            traj = self.state_samples[particle_ind].clone()
        elif mode == 'mean':
            traj = self._mean.clone()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return traj

    def get_recent_samples(self):
        return (
            self._recent_control_samples.detach().clone(),
            self._recent_control_particles.detach().clone(),
            self._recent_state_trajectories.detach().clone(),
            self._recent_state_particles.detach().clone(),
            self._recent_weights.detach().clone(),
        )

    def sample_trajectories(self, num_samples_per_particle):
        self._sample_dist.set_mean(self.particle_means.view(self.num_particles, -1))
        self.state_samples = self._sample_dist.sample(num_samples_per_particle).to(
            **self.tensor_args)
        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]
        return (
            position_seq,
            velocity_seq,
        )



class GPMP:

    def __init__(
            self,
            num_particles_per_goal,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            step_size=1.,
            temp=1.,
            start_state=None,
            multi_goal_states=None,
            cost=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            sigma_goal=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            seed=None,
            solver_params=None,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        if seed is not None:
            torch.manual_seed(seed)

        self.n_dof = n_dof
        self.d_state_opt = 2 * self.n_dof
        self.dt = dt

        self.traj_len = traj_len
        self.goal_directed = (multi_goal_states is not None)
        if not self.goal_directed:
            self.num_goals = 1
        else:
            assert multi_goal_states.dim() == 2
            self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.sigma_start_init = sigma_start_init
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal = sigma_goal
        self.sigma_goal_init = sigma_goal_init
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp_init = sigma_gp_init
        self.sigma_gp_sample = sigma_gp_sample
        self.start_states = start_state # position + velocity
        self.solver_params = solver_params
        self.multi_goal_states = multi_goal_states # position + velocity
        self.cost = cost
        self.N = self.d_state_opt * self.traj_len

        self._mean = None
        self._weights = None
        self._dist = None

        self.reset(start_state, multi_goal_states)

    def set_prior_factors(self):

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_init,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        if self.goal_directed:
            self.multi_goal_prior_init = []
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_init,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_sample,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sigma_gp_sample,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        if self.goal_directed:
            self.multi_goal_prior_sample = []
            for i in range(self.num_goals):
                self.multi_goal_prior_sample.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_sample,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

    def get_dist(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
    ):

        return MultiMPPrior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
    ):

        if start_state is not None:
            self.start_state = start_state.clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.clone()

        self.set_prior_factors()

        # Initialization particles from prior distribution
        self._init_dist = self.get_dist(
            self.start_prior_init.K,
            self.gp_prior_init.Q_inv[0],
            self.multi_goal_prior_init[0].K if self.goal_directed else None,
            self.start_state,
            goal_states=self.multi_goal_states,
        )
        self.particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
        self.particle_means = self.particle_means.flatten(0, 1)
        del self._init_dist  # freeing memory

        self._sample_dist = self.get_dist(
            self.start_prior_sample.K,
            self.gp_prior_sample.Q_inv[0],
            self.multi_goal_prior_sample[0].K if self.goal_directed else None,
            self.start_state,
            particle_means=self.particle_means,
        )

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            b, K = self._step(**observation)

        self.costs = self._get_costs(b, K)

        position_seq_mean = self.particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self.particle_means[..., -self.n_dof:].clone()
        costs = self.costs.clone()

        self._recent_control_particles = velocity_seq_mean
        self._recent_state_trajectories = position_seq_mean

        return (
            velocity_seq_mean,
            position_seq_mean,
            costs,
        )

    def _step(self, **observation):
        A, b, K = self.cost.get_linear_system(self.particle_means, **observation)

        J_t_J, g = self._get_grad_terms(
            A, b, K,
            delta=self.solver_params['delta'],
            trust_region=self.solver_params['trust_region'],
        )
        d_theta = self.get_torch_solve(
            J_t_J, g,
            method=self.solver_params['method'],
        )

        d_theta = d_theta.view(
                self.num_particles,
                self.traj_len,
                self.d_state_opt,
            )

        self.particle_means = self.particle_means + self.step_size * d_theta
        self.particle_means.detach_()

        return b, K

    def _get_grad_terms(
            self,
            A, b, K,
            delta=0.,
            trust_region=False,
    ):
        I = torch.eye(self.N, self.N, **self.tensor_args)
        A_t_K = A.transpose(1, 2) @ K
        A_t_A = A_t_K @ A
        if not trust_region:
            J_t_J = A_t_A + delta*I
        else:
            J_t_J = A_t_A + delta * I * torch.diagonal(A_t_A, dim1=1, dim2=2).unsqueeze(-1)
            # Since hessian will be averaged over particles, add diagonal matrix of the mean.
            diag_A_t_A = A_t_A.mean(0) * I
            J_t_J = A_t_A + delta * diag_A_t_A
        g = A_t_K @ b
        return J_t_J, g

    def get_torch_solve(
        self,
        A, b,
        method,
    ):
        if method == 'inverse':
            return torch.linalg.solve(A, b)
        elif method == 'cholesky':
            l = torch.linalg.cholesky(A)
            # z = torch.linalg.solve_triangular(l, b, upper=False)
            # return torch.linalg.solve_triangular(l.mT, z, upper=False)
            z = torch.triangular_solve(b, l, transpose=False, upper=False)[0]
            return torch.triangular_solve(z, l, transpose=True, upper=False)[0]
        else:
            raise NotImplementedError

    def _get_costs(self, errors, w_mat):
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(self.num_particles,)

    def sample_trajectories(self, num_samples_per_particle):
        self._sample_dist.set_mean(self.particle_means.view(self.num_particles, -1))
        self.state_samples = self._sample_dist.sample(num_samples_per_particle).to(
            **self.tensor_args)
        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]
        return (
            position_seq,
            velocity_seq,
        )
