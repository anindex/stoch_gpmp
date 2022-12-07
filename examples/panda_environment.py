import time
import matplotlib.pyplot as plt
import random
import numpy as np
import torch

from stoch_gpmp.planner import StochGPMP
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite, CostGP, CostGoal

from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_planning_objectives.fields.collision_bodies import PandaSphereDistanceField
from torch_planning_objectives.fields.distance_fields import SkeletonPointField
from torch_kinematics_tree.geometrics.frame import Frame
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model


def random_init_static_sphere(
    scale_min: float,
    scale_max: float,
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    base_offset: float,
) -> tuple:
    # Get scale
    alpha_scale = np.random.uniform()
    scale = alpha_scale * scale_min + (1 - alpha_scale) * scale_max

    # Get position
    idx = np.random.permutation([1, 0, 0])
    base_position = np.random.rand(3)
    alpha = np.random.rand(1)
    base_position[idx == 1] = (
        alpha * base_position_min[idx == 1] + (1 - alpha) * base_position_max[idx == 1]
    )
    base_position[:-1] *= np.random.randint(2, size=2) * 2 - 1

    # Guarantee no collision at the beginning
    base_position = np.sign(base_position) * np.clip(
        np.abs(base_position), a_min=base_offset, a_max=base_position_max
    )
    # Guarantee no collision with the box
    base_position = np.sign(base_position) * np.clip(
        np.abs(base_position), a_min=base_offset, a_max=base_position_max
    )
    return scale, base_position


if __name__ == '__main__':
    # params
    # device = 'cpu'
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float64}
    seed = int(time.time())
    num_particles_per_goal = 10
    num_samples = 1024
    num_obst = 5
    traj_len = 8
    dt = 0.01
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # world setup
    target_pos = np.array([.3, .3, .3])
    target_H = Frame(trans=torch.from_numpy(target_pos).to(**tensor_args), device=device).get_transform_matrix()  # set translation and orientation of target here

    # start & goal
    start_q = torch.tensor([0.012, -0.57, 0., -2.81, 0., 3.037, 0.741], **tensor_args)
    start_state = torch.cat((start_q, torch.zeros_like(start_q)))
    multi_goal_states = None   # NOTE(an): can also put IK solution here

    # FK
    panda_fk = DifferentiableFrankaPanda(gripper=False, device=device)
    n_dof = panda_fk._n_dofs

    ## Cost functions
    panda_collision = PandaSphereDistanceField(device=device)
    panda_collision.build_batch_features(batch_dim=[num_particles_per_goal * num_samples * (traj_len - 1), ], clone_objs=True)
    panda_goal = SkeletonPointField(via_H=target_H, link_list=['ee_link'], device=device)
    panda_goal.set_link_weights({
        'ee_link': 1.
    })
    # Factored Cost params
    cost_sigmas = dict(
        sigma_start=0.0001,
        sigma_gp=3.,
    )
    sigma_coll = 0.000001
    sigma_goal = 0.000001

    # Construct cost function
    cost_func_list = []

    cost_prior = CostGP(
        n_dof, traj_len, start_state, dt,
        cost_sigmas, tensor_args
    )
    cost_func_list += [cost_prior.eval]
    cost_coll = CostCollision(n_dof, traj_len, sigma_coll=sigma_coll)
    cost_func_list += [cost_coll.eval]
    cost_goal = CostGoal(n_dof, traj_len, sigma_goal=sigma_goal)
    cost_func_list += [cost_goal.eval]

    cost_composite = CostComposite(n_dof, traj_len, cost_func_list)
    cost_func = cost_composite.eval

    ## Planner - 2D point particle dynamics
    stochgpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        traj_len=traj_len,
        dt=dt,
        n_dof=n_dof,
        opt_iters=1, # Keep this 1 for visualization
        temp=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        cost_func=cost_func,
        step_size=0.5,
        sigma_start_init=0.0001,
        sigma_goal_init=1,
        sigma_gp_init=100.,
        sigma_start_sample=0.0001,
        sigma_goal_sample=1,
        sigma_gp_sample=25.,
        sigma_goal=1.,  # this is not used (sigma_goal joint space)
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = StochGPMP(**stochgpmp_params)

    # spawn obstacles
    obst_r = [0.1, 0.2]
    obst_range_lower = np.array([0.6, -0.2, 0.6])
    obst_range_upper = np.array([1., 0.2, 1])
    obstacle_spheres = np.zeros((1, num_obst, 4))
    for i in range(num_obst):
        r, pos = random_init_static_sphere(obst_r[0], obst_r[1], obst_range_lower, obst_range_upper, 0.01)
        obstacle_spheres[0, i, :3] = pos
        obstacle_spheres[0, i, 3] = r
    obstacle_spheres = torch.from_numpy(obstacle_spheres).to(**tensor_args)

    obs = {
        'collision_field': panda_collision.compute_cost,
        'goal_field': panda_goal.compute_cost,
        'FK': lambda q: panda_fk.compute_forward_kinematics_all_links(q, return_dict=True),
        'obstacle_spheres': obstacle_spheres
    }

    #---------------------------------------------------------------------------
    # Optimize
    opt_iters = 400

    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        print(f'Time(s) per iter: {time.time() - time_start} sec')
        controls, _, trajectories, trajectory_means, weights = planner.get_recent_samples()

    #---------------------------------------------------------------------------
    # Plotting
    start_q = start_state.detach().cpu().numpy()
    trajs = trajectory_means.detach()
    obstacle_spheres = obstacle_spheres.detach().cpu().numpy()

    for traj in trajs:
        plt.figure()
        ax = plt.axes(projection='3d')
        for t in range(traj.shape[0] - 1):
            skeleton = get_skeleton_from_model(panda_fk, traj[t], panda_fk.get_link_names())
            skeleton.draw_skeleton()
        skeleton = get_skeleton_from_model(panda_fk, traj[-1], panda_fk.get_link_names())
        skeleton.draw_skeleton(color='g')

        start_skeleton = get_skeleton_from_model(panda_fk, start_state[:n_dof], panda_fk.get_link_names())
        start_skeleton.draw_skeleton(color='b')
        ax.plot(target_pos[0], target_pos[1], target_pos[2], 'r*', markersize=7)
        ax.scatter(obstacle_spheres[0, :, 0], obstacle_spheres[0, :, 1], obstacle_spheres[0, :, 2], s=obstacle_spheres[0, :, 3]*1000, color='r')
        plt.show()
