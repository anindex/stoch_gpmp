from abc import ABC, abstractmethod
import torch

from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance


class DistanceField(ABC):
    def __init__(self, tensor_args=None):
        self.tensor_args = tensor_args

    def distances(self):
        pass

    def compute_collision(self):
        pass

    @abstractmethod
    def compute_distance(self):
        pass

    @abstractmethod
    def compute_cost(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


class LinkDistanceField(DistanceField):

    def __init__(self, field_type='rbf', clamp_sdf=False, num_interpolate=0, link_interpolate_range=[5, 7],
                 **kwargs):
        super().__init__(**kwargs)
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range

    def distances(self, link_tensor, obstacle_spheres):
        link_pos = link_tensor[..., :3, -1]
        link_pos = link_pos.unsqueeze(-2)
        obstacle_spheres = obstacle_spheres.unsqueeze(0).unsqueeze(0)
        centers = obstacle_spheres[..., :3]
        radii = obstacle_spheres[..., 3]
        return torch.linalg.norm(link_pos - centers, dim=-1) - radii

    def compute_collision(self, link_tensor, obstacle_spheres=None, buffer=0.02):  # position tensor
        collisions = torch.zeros(link_tensor.shape[:2]).to(**self.tensor_args)  # batch, trajectory
        if obstacle_spheres is None:
            return collisions
        distances = self.distances(link_tensor, obstacle_spheres)
        collisions = torch.any(torch.any(distances < buffer, dim=-1), dim=-1)
        return collisions

    def compute_distance(self, link_tensor, obstacle_spheres=None, **kwargs):
        if obstacle_spheres is None:
            return 1e10
        link_tensor = link_tensor[..., :3, -1].unsqueeze(-2)
        obstacle_spheres = obstacle_spheres.unsqueeze(0)
        return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) - obstacle_spheres[..., 3]).sum((-1, -2)) 

    def compute_cost(self, link_tensor, obstacle_spheres=None, **kwargs):
        if obstacle_spheres is None:
            return 0
        link_tensor = link_tensor[..., :3, -1]
        link_dim = link_tensor.shape[:-1]
        if self.num_interpolate > 0:
            alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
            alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
            for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
                X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
                eval_sphere = X1 + (X2 - X1) * alpha
                link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
        link_tensor = link_tensor.unsqueeze(-2)
        obstacle_spheres = obstacle_spheres.unsqueeze(0)
        # signed distance field
        if self.field_type == 'rbf':
            return torch.exp(-0.5 * torch.square(link_tensor - obstacle_spheres[..., :3]).sum(-1) / torch.square(obstacle_spheres[..., 3])).sum((-1, -2))
        elif self.field_type == 'sdf':
            sdf = -torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) + obstacle_spheres[..., 3]
            if self.clamp_sdf:
                sdf = sdf.clamp(max=0.)
            return sdf.max(-1)[0].max(-1)[0]
        elif self.field_type == 'occupancy':
            return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) < obstacle_spheres[..., 3]).sum((-1, -2))

    def zero_grad(self):
        pass


class LinkSelfDistanceField(DistanceField):

    def __init__(self, margin=0.03, num_interpolate=0, link_interpolate_range=[5, 7], **kwargs):
        super().__init__(**kwargs)
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range
        self.margin = margin

    def distances(self, link_tensor):
        link_pos = link_tensor[..., :3, -1]
        return torch.linalg.norm(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3), dim=-1)

    def compute_collision(self, link_tensor, buffer=0.05):  # position tensor
        distances = self.distances(link_tensor)
        self_collisions = torch.tril(distances < buffer, diagonal=-2)
        any_self_collision = torch.any(torch.any(self_collisions, dim=-1), dim=-1)
        return any_self_collision

    def compute_distance(self, link_tensor):  # position tensor
        distances = self.distances(link_tensor)
        return distances.sum((-1, -2))

    def compute_cost(self, link_tensor, **kwargs):   # position tensor
        link_tensor = link_tensor[..., :3, -1]
        link_dim = link_tensor.shape[:-1]
        if self.num_interpolate > 0:
            alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
            alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
            for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
                X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
                eval_sphere = X1 + (X2 - X1) * alpha
                link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
        return torch.exp(torch.square(link_tensor.unsqueeze(-2) - link_tensor.unsqueeze(-3)).sum(-1) / (-self.margin**2 * 2)).sum((-1, -2))

    def zero_grad(self):
        pass


class EESE3DistanceField(DistanceField):

    def __init__(self, target_H, w_pos=1., w_rot=1., square=True, **kwargs):
        super().__init__(**kwargs)
        self.target_H = target_H
        self.square = square
        self.w_pos = w_pos
        self.w_rot = w_rot

    def update_target(self, target_H):
        self.target_H = target_H

    def compute_distance(self, link_tensor):  # position tensor
        return SE3_distance(link_tensor[..., -1, :, :], self.target_H, w_pos=self.w_pos,
                            w_rot=self.w_rot)  # get EE as last link

    def compute_cost(self, link_tensor, **kwargs):  # position tensor
        dist = self.compute_distance(link_tensor).squeeze()
        if self.square:
            dist = torch.square(dist)
        return dist

    def zero_grad(self):
        pass