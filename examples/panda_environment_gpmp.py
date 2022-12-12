import time
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import pybullet as p

from stoch_gpmp.planner import GPMP
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite, CostGP, CostGoal, CostGoalPrior

from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_planning_objectives.fields.distance_fields import EESE3DistanceField, LinkDistanceField, FloorDistanceField, LinkSelfDistanceField
from torch_kinematics_tree.geometrics.frame import Frame
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from robot_envs.pybullet.utils import random_init_static_sphere
from robot_envs.pybullet.panda import PandaEnv
from torch_kinematics_tree.geometrics.spatial_vector import (
    z_rot,
    y_rot,
    x_rot,
)


if __name__ == '__main__':
    # params
    # device = 'cpu'
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}
    seed = int(time.time())
    num_particles_per_goal = 5
    num_obst = 5
    traj_len = 64
    dt = 0.05
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # world setup (target_pos & target_rot can be randomized)
    target_pos = np.array([-.3, -.3, .3])
    target_rot = (z_rot(-torch.tensor(torch.pi)) @ y_rot(-torch.tensor(torch.pi))).to(**tensor_args)
    target_frame = Frame(rot=target_rot, trans=torch.from_numpy(target_pos).to(**tensor_args), device=device)
    target_quat = target_frame.get_quaternion().squeeze().cpu().numpy()  # [x, y, z, w]
    target_H = target_frame.get_transform_matrix()  # set translation and orientation of target here

    # FK
    panda_fk = DifferentiableFrankaPanda(gripper=False, device=device)
    # panda_fk.print_link_names()
    n_dof = panda_fk._n_dofs

    # start & goal
    start_q = torch.tensor([0.012, -0.57, 0., -2.81, 0., 3.037, 0.741], **tensor_args)
    start_state = torch.cat((start_q, torch.zeros_like(start_q)))
    # use IK solution from pybullet
    env = PandaEnv(
        render=False,
        realtime=False,
        horizon=100000
    )
    env.reset()
    q_goal = env.panda.solveInverseKinematics(target_pos, target_quat)[:n_dof]
    q_goal = torch.tensor(q_goal, **tensor_args)
    # multi_goal_states = None
    multi_goal_states = torch.cat([q_goal, torch.zeros_like(q_goal)]).unsqueeze(0)  # put IK solution

    ## Cost functions
    # panda_collision = PandaSphereDistanceField(device=device)
    # panda_collision.build_batch_features(batch_dim=[num_particles_per_goal * num_samples * (traj_len - 1), ], clone_objs=True)\
    # panda_floor = FloorDistanceField(margin=0.05, device=device)
    panda_self_link = LinkSelfDistanceField(margin=0.03, device=device)
    panda_collision_link = LinkDistanceField(device=device)
    panda_goal = EESE3DistanceField(target_H, device=device)

    # Factored Cost params
    prior_sigmas = dict(
        sigma_start=0.0001,
        sigma_gp=0.01,
    )
    # sigma_floor = 0.1
    sigma_self = 0.05
    sigma_coll = 0.0005
    sigma_goal = 0.0005
    sigma_goal_prior = 10.

    # Construct cost function
    cost_prior = CostGP(
        n_dof, traj_len, start_state, dt,
        prior_sigmas, tensor_args
    )
    # cost_floor = CostCollision(n_dof, traj_len, field=panda_floor, sigma_coll=sigma_floor)
    cost_self = CostCollision(n_dof, traj_len, field=panda_self_link, sigma_coll=sigma_self, tensor_args=tensor_args)
    cost_coll = CostCollision(n_dof, traj_len, field=panda_collision_link, sigma_coll=sigma_coll, tensor_args=tensor_args)
    cost_goal = CostGoal(n_dof, traj_len, field=panda_goal, sigma_goal=sigma_goal, tensor_args=tensor_args)
    cost_goal_prior = CostGoalPrior(n_dof, traj_len, multi_goal_states=multi_goal_states, 
                                    num_particles_per_goal=num_particles_per_goal,
                                    sigma_goal_prior=sigma_goal_prior,
                                    tensor_args=tensor_args)
    cost_func_list = [cost_prior, cost_goal_prior, cost_self, cost_coll, cost_goal]
    # cost_func_list = [cost_prior, cost_goal_prior, cost_goal]
    cost_composite = CostComposite(n_dof, traj_len, cost_func_list, FK=panda_fk.compute_forward_kinematics_all_links, tensor_args=tensor_args)

    ## Planner - 2D point particle dynamics
    gpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        traj_len=traj_len,
        dt=dt,
        n_dof=n_dof,
        opt_iters=1, # Keep this 1 for visualization
        temp=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        cost=cost_composite,
        step_size=0.003,
        sigma_start_init=0.0001,
        sigma_goal_init=0.05,
        sigma_gp_init=0.8,
        sigma_start_sample=0.001,
        sigma_goal_sample=0.07,
        sigma_gp_sample=0.2,
        seed=seed,
        solver_params={
            'delta': 0.,
            'trust_region': True,
            'method': 'cholesky',
        },
        tensor_args=tensor_args,
    )
    planner = GPMP(**gpmp_params)

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
        'obstacle_spheres': obstacle_spheres
    }

    #---------------------------------------------------------------------------
    # Optimize
    opt_iters = 40

    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        control_means, trajectory_means, costs = planner.optimize(obs)
        print(f'Time(s) per iter: {time.time() - time_start} sec')

    #---------------------------------------------------------------------------
    # Plotting
    start_q = start_state.detach().cpu().numpy()
    trajs = trajectory_means.detach()
    obstacle_spheres = obstacle_spheres.detach().cpu().numpy()

    for traj in trajs:
        plt.figure()
        ax = plt.axes(projection='3d')
        skeleton = get_skeleton_from_model(panda_fk, q_goal, panda_fk.get_link_names()) # visualize IK solution
        skeleton.draw_skeleton(color='r')
        for t in range(traj.shape[0] - 1):
            if t % 4 == 0:
                skeleton = get_skeleton_from_model(panda_fk, traj[t], panda_fk.get_link_names())
                skeleton.draw_skeleton()
        skeleton = get_skeleton_from_model(panda_fk, traj[-1], panda_fk.get_link_names())
        skeleton.draw_skeleton(color='g')

        start_skeleton = get_skeleton_from_model(panda_fk, start_state[:n_dof], panda_fk.get_link_names())
        start_skeleton.draw_skeleton(color='b')
        ax.plot(target_pos[0], target_pos[1], target_pos[2], 'r*', markersize=7)
        ax.scatter(obstacle_spheres[0, :, 0], obstacle_spheres[0, :, 1], obstacle_spheres[0, :, 2], s=obstacle_spheres[0, :, 3]*2000, color='r')
        plt.show()
