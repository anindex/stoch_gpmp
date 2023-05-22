import torch
import time
import random
import matplotlib.pyplot as plt

# from dplan.motion_planners.utils import elapsed_time
# from experiment_launcher.utils import fix_random_seed
from robot_envs.base_envs.pointmass_env_base import PointMassEnvBase
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from stoch_gpmp.planner import StochGPMP
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite, CostGP, CostGoalPrior
from torch_planning_objectives.fields.primitive_distance_fields import SphereField


if __name__ == "__main__":
    seed = 0
    # fix_random_seed(seed)

    device = 'cuda'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    obst_primitives_l = [
        SphereField([[0., 0.]],
                    [1.0],
                    tensor_args=tensor_args
                    )
    ]

    env = PointMassEnvBase(
        q_min=(-10, -10),
        q_max=(10, 10),
        work_space_bounds=((-10., 10.), (-10., 10.), (-10., 10.)),
        obst_primitives_l=obst_primitives_l,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_dof = env.q_dim
    traj_len = 64
    dt = 0.02
    num_particles_per_goal = 5
    num_samples = 128
    seed = 11
    start_q = torch.Tensor([-9, -9]).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))

    multi_goal_states = torch.tensor([
        [9, 6, 0., 0.],
        [9, -3, 0., 0.],
        [-3, 9, 0., 0.],
    ]).to(**tensor_args)

    # -------------------------------- Cost func. ---------------------------------
    # Factored Cost params
    cost_sigmas = dict(
        sigma_start=0.001,
        sigma_gp=0.1,
    )
    sigma_coll = 1e-5
    sigma_goal_prior = 0.001

    # Construct cost function
    cost_prior = CostGP(
        n_dof, traj_len, start_state, dt,
        cost_sigmas, tensor_args
    )
    cost_goal_prior = CostGoalPrior(n_dof, traj_len, multi_goal_states=multi_goal_states, 
                                    num_particles_per_goal=num_particles_per_goal, 
                                    num_samples=num_samples, 
                                    sigma_goal_prior=sigma_goal_prior,
                                    tensor_args=tensor_args)

    def collision_cost_wrapper(samples, **kwargs):
        # sum over points in the trajectory
        return env.compute_collision_cost(samples, field_type='sdf').sum(-1)

    cost_obst_2D = collision_cost_wrapper
    cost_func_list = [cost_prior, cost_goal_prior, cost_obst_2D]
    cost_composite = CostComposite(n_dof, traj_len, cost_func_list)

    # -------------------------------- Planner ---------------------------------
    stochgpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        traj_len=traj_len,
        dt=dt,
        n_dof=n_dof,
        opt_iters=1,  # Keep this 1 for visualization
        temperature=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        cost=cost_composite,
        step_size=0.3,
        sigma_start_init=1e-3,
        sigma_goal_init=1e-3,
        sigma_gp_init=10.,
        sigma_start_sample=1e-3,
        sigma_goal_sample=1e-3,
        sigma_gp_sample=1.,
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = StochGPMP(**stochgpmp_params)
    obs = {}

    # ---------------------------------------------------------------------------
    # Optimize
    opt_iters = 1000

    traj_history = []
    start_time = time.time()
    for i in range(opt_iters + 1):
        time_start = time.time()
        planner.optimize(**obs)
        time_finish = time.time()
        if i == 1 or i % 50 == 0:
            print(i)
            print(f'Time(s) per iter: {time_finish - time_start:.4f} sec')
            trajectories, controls = planner.get_recent_samples()
            traj_history.append(trajectories)
    # print(f'Planning time: {elapsed_time(start_time)}')

    # ---------------------------------------------------------------------------
    # Plotting

    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    fig = plt.figure()
    for trajs in traj_history:
        plt.clf()
        ax = fig.gca()
        env.render(ax)

        trajs = trajs.cpu().numpy()
        mean_trajs = trajs.mean(1)
        for i in range(trajs.shape[0]):
            for j in range(trajs.shape[1]):
                ax.plot(trajs[i, j, :, 0], trajs[i, j, :, 1], 'r', alpha=0.15)
        for i in range(trajs.shape[0]):
            ax.plot(mean_trajs[i, :, 0], mean_trajs[i, :, 1], 'b')
        plt.draw()
        plt.pause(1e-2)
