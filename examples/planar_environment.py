import torch
import time
import random
import matplotlib.pyplot as plt

from stoch_gpmp.costs.obstacle_map.map_generator import generate_obstacle_map
from stoch_gpmp.planner import StochGPMP


if __name__ == "__main__":
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    start_q = torch.Tensor([-9, -9]).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))

    multi_goal_states = torch.tensor([
        [9, 6, 0., 0.],
        [9, -3, 0., 0.],
        [-3, 9, 0., 0.],
    ]).to(**tensor_args)

    num_particles = 16
    num_particles_per_goal = 5
    seed = 11

    ## Planner - 2D point particle dynamics
    stochgpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        num_samples=128,
        traj_len=64,
        dt=0.02,
        n_dof=2,
        opt_iters=1, # Keep this 1 for visualization
        temp=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        step_size=0.5,
        sigma_start=1e-3,
        sigma_goal=1e-3,
        sigma_gp=0.1,
        sigma_start_init=1e-3,
        sigma_goal_init=1e-3,
        sigma_gp_init=50.,
        sigma_start_sample=1e-3,
        sigma_goal_sample=1e-3,
        sigma_gp_sample=3,
        sigma_obst=1e-5,
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = StochGPMP(**stochgpmp_params)

    ## Obstacle map
    # obst_list = [(0, 0, 4, 6)]
    obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=10,
        rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
        rand_shape=[2, 2],
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    obst_map = generate_obstacle_map(**obst_params)[0]

    obs = {
        'obst_map_func': obst_map.get_collisions
    }

    #---------------------------------------------------------------------------
    # Optimize
    opt_iters = 500

    traj_history = []
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        print(f'Time(s) per iter: {time.time() - time_start} sec')
        if i == 1 or i % 50 == 0:
            controls, _, trajectories, _, weights = planner.get_recent_samples()
            traj_history.append(trajectories)

    #---------------------------------------------------------------------------
    # Plotting

    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    fig = plt.figure()
    for trajs in traj_history:
        plt.clf()
        ax = fig.gca()
        cs = ax.contourf(x, y, obst_map.map, 20)
        cbar = fig.colorbar(cs, ax=ax)

        trajs = trajs.cpu().numpy()
        mean_trajs = trajs.mean(1)
        for i in range(trajs.shape[0]):
            for j in range(trajs.shape[1]):
                ax.plot(trajs[i, j, :, 0], trajs[i, j, :, 1], 'r', alpha=0.15)
        for i in range(trajs.shape[0]):
            ax.plot(mean_trajs[i, :, 0], mean_trajs[i, :, 1], 'b')
        plt.draw()
        plt.pause(1e-10)
