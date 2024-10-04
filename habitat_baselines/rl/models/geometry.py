from typing import Dict, Tuple
from einops import rearrange
from multiprocess.pool import Pool
import torch.nn.functional as F
import torch_scatter
import torch
import numpy as np
import math
import cv2
from einops import asnumpy

"""
Code adapted from https://github.com/saimwani/multiON
"""
def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    return rot_grid, trans_grid

class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).floor()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).floor()
        return grid_x, grid_y
    
    def get_gps_coords(self, idx):
        # H, W indices to gps coordinates
        grid_x = idx[0].item()
        grid_y = idx[1].item()

        gps_x = self.coordinate_max - grid_x * self.grid_size
        gps_y = self.coordinate_min + grid_y * self.grid_size

        return gps_x, gps_y

class RotateTensor:
    def __init__(self, device):
        self.device = device

    def forward(self, x_gp, heading):
        sin_t = torch.sin(heading.squeeze(1))
        cos_t = torch.cos(heading.squeeze(1))
        A = torch.zeros(x_gp.size(0), 2, 3).to(self.device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = sin_t
        A[:, 1, 0] = -sin_t
        A[:, 1, 1] = cos_t

        grid = F.affine_grid(A, x_gp.size(), align_corners=True)
        rotated_x_gp = F.grid_sample(x_gp, grid, align_corners=True, mode='nearest')
        return rotated_x_gp

class ComputePointCloud():
    def __init__(self, egocentric_map_size, global_map_size, 
        device, coordinate_min, coordinate_max, height_min, height_max, is_occ=False
    ):
        self.device = device
        self.cx, self.cy = 256./2., 256./2.   
        self.fx = (256. / 2.) / np.tan(np.deg2rad(79 / 2.))
        self.fy = (256. / 2.) / np.tan(np.deg2rad(79 / 2.))
        # self.cx, self.cy = 640./2., 480./2.   
        # self.fx = (640. / 2.) / np.tan(np.deg2rad(90 / 2.))
        # self.fy = (480. / 2.) / np.tan(np.deg2rad(67.5 / 2.))
        
        self.egocentric_map_size = egocentric_map_size
        self.local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
        self.height_min = height_min
        self.height_max = height_max
        self.is_occ = is_occ
        
    def forward(self, depth) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = depth.permute(0, 3, 1, 2)
        _, _, imh, imw = depth.shape   # batchsize, 1, imh, imw
        x    = rearrange(torch.arange(0, imw), 'w -> () () () w').to(self.device)
        y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(self.device)
        xx   = (x - self.cx) / self.fx
        yy   = (y - self.cy) / self.fy
        # 3D real-world coordinates (in meters)
        Z = depth
        X = xx * Z
        Y = yy * Z
        
        if self.is_occ:
            valid_inputs = (depth != 0) & (depth <= 10.) & (Y < self.height_max)
        else:
            valid_inputs = (depth != 0) & (depth <= 10.) & (Y > self.height_min) & (Y < self.height_max)
        # X ground projection and Y ground projection
        x_gp = ( (X / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, 1, imh, imw)
        y_gp = (-(Z / self.local_scale) + (self.egocentric_map_size-1)/2).round().long() # (bs, 1, imh, imw)
        
        return torch.cat([x_gp, y_gp], dim=1), Y, valid_inputs

class ProjectToGroundPlane():
    def __init__(self, egocentric_map_size, device, 
            vaccant_bel, occupied_bel, 
            height_min, height_max
        ):
        self.egocentric_map_size = egocentric_map_size
        self.device = device
        self.vaccant_bel = vaccant_bel
        self.occupied_bel = occupied_bel
        self.height_min = height_min
        self.height_max = height_max

    def forward(self, img, spatial_locs, valid_map):
        (outh, outw) = (self.egocentric_map_size, self.egocentric_map_size)
        bs, f, HbyK, WbyK = img.shape
        K = 1
        eps=-1e16
        # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
        idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(self.device), \
                    (torch.arange(0, WbyK, 1)*K).long().to(self.device))

        spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
        valid_inputs_ss = valid_map[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
        valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
        invalid_inputs_ss = ~valid_inputs_ss
        
        # Filter out invalid spatial locations
        invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                            (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

        invalid_writes = invalid_spatial_locs | invalid_inputs_ss
        
        # Set the idxes for all invalid locations to (0, 0)
        spatial_locs_ss[:, 0][invalid_spatial_locs] = 0
        spatial_locs_ss[:, 1][invalid_spatial_locs] = 0
        # invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').bool()
        invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
        img_masked = img * (1 - invalid_writes_f) + eps * invalid_writes_f
        
        # Linearize ground-plane indices (linear idx = y * W + x)
        linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
        linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
        linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()
        img_target = rearrange(img_masked, 'b e h w -> b e (h w)')
        
        proj_feats, _ = torch_scatter.scatter_min(
            img_target,
            linear_locs_ss,
            dim=2,
            dim_size=outh*outw
        )
        proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)
        eps_mask = (proj_feats == eps).float()
        proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)
        # Valid inputs
        occupied_area = (proj_feats != 0) & ((proj_feats > self.height_min) & (proj_feats < self.height_max))
        vaccant_area = (proj_feats != 0) & (proj_feats < self.height_min)
        
        # The belief image for projection
        belief_map = torch.zeros_like(proj_feats)
        belief_map[occupied_area] = self.occupied_bel
        belief_map[vaccant_area] = self.vaccant_bel
        
        return belief_map
    
class ProjectSemanticToGroundPlane():
    def __init__(self, egocentric_map_size, num_classes, device):
        self.egocentric_map_size = egocentric_map_size
        self.num_classes = num_classes
        self.device = device

    def forward(self, img, spatial_locs, valid_map):
        ### img: B X C X H X W, C = num_classes, one-hot encoding vector
        ### torch_sum => get classes frequency histogram
        ### get argmax in dim=1 will be most-frequent class index
        (outh, outw) = (self.egocentric_map_size, self.egocentric_map_size)
        bs, f, HbyK, WbyK = img.shape
        K = 1
        eps=0
        # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
        idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(self.device), \
                    (torch.arange(0, WbyK, 1)*K).long().to(self.device))

        spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
        valid_inputs_ss = valid_map[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
        valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
        invalid_inputs_ss = ~valid_inputs_ss
        
        # Filter out invalid spatial locations
        invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                            (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

        invalid_writes = invalid_spatial_locs | invalid_inputs_ss
        
        # Set the idxes for all invalid locations to (0, 0)
        spatial_locs_ss[:, 0][invalid_spatial_locs] = 0
        spatial_locs_ss[:, 1][invalid_spatial_locs] = 0        
        invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').repeat(1, f, 1, 1).bool()
        img[invalid_writes_f] = 0 
          
        # Linearize ground-plane indices (linear idx = y * W + x)
        linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
        linear_locs_ss = rearrange(linear_locs_ss.contiguous(), 'b h w -> b () (h w)')
        linear_locs_ss = linear_locs_ss.repeat(1, f, 1) # .contiguous()
        img_target: torch.Tensor = rearrange(img.contiguous(), 'b e h w -> b e (h w)')
        
        # proj_feats = torch.zeros((bs, f, outh * outw), device=self.device)
        # proj_feats.scatter_add_(2, linear_locs_ss, img_target)
        proj_feats = torch_scatter.scatter_add(
            img_target,
            linear_locs_ss,
            dim=2,
            dim_size=outh*outw
        )
        proj_feats = rearrange(proj_feats.contiguous(), 'b e (h w) -> b e h w', h=outh)
        proj_feats[:, 0, ...] = 0
        proj_feats = torch.argmax(proj_feats, dim=1) # B X H X W
        proj_feats = F.one_hot(proj_feats.long(), num_classes=self.num_classes).long().permute(0, 3, 1, 2)
        return proj_feats # B X C X H X W

class OccupancyProjection:
    def __init__(self, 
            egocentric_map_size, global_map_size, device, 
            coordinate_min, coordinate_max, vaccant_bel, occupied_bel, 
            height_min, height_max
        ):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputePointCloud(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max, height_min, height_max, is_occ=True
        ) 
        self.project_to_ground_plane = ProjectToGroundPlane(egocentric_map_size, device, 
            vaccant_bel, occupied_bel,
            height_min, height_max
        )

    def forward(self, depth) -> torch.Tensor:
        spatial_locs, height_map, valid_maps = self.compute_spatial_locs.forward(depth)
        ego_local_map = self.project_to_ground_plane.forward(height_map, spatial_locs, valid_maps)
        return ego_local_map

class SemanticObsProjection:
    def __init__(self, 
            egocentric_map_size, global_map_size, device, 
            coordinate_min, coordinate_max, height_min, height_max, num_classes
        ):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.compute_spatial_locs = ComputePointCloud(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max, height_min, height_max
        )
        self.project_to_ground_plane = ProjectSemanticToGroundPlane(egocentric_map_size, num_classes, device)

    def forward(self, semantic, depth) -> torch.Tensor:
        spatial_locs, _, valid_maps = self.compute_spatial_locs.forward(depth)
        ego_local_map = self.project_to_ground_plane.forward(semantic, spatial_locs, valid_maps)
        return ego_local_map

class Registration():
    def __init__(self, 
            egocentric_map_size, 
            global_map_size, 
            coordinate_min, 
            coordinate_max, 
            num_process, 
            device, 
            global_map_depth = 1
        ):
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth
        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.num_obs = num_process
        self.device = device
        
    def forward(
            self, 
            observations: Dict[str, torch.Tensor], 
            global_allocentric_map: torch.Tensor, 
            egocentric_map: torch.Tensor,
            # trajectory_map: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Register egocentric_map to full_global_map

        Args:
            observations: Dictionary containing habitat observations
                - Position: observations['gps']
                - Heading: observations['compass']
            global_allocentric_map: torch.tensor containing global map, (num_obs, global_map_depth=1, global_map_size, global_map_size) 
            egocentric_map: torch.tensor containing egocentrc map, (num_obs, global_map_depth=1, egocentric_map_size, egocentric_map_size) 

        Returns:
            registered_map: torch.tensor containing registered map, (num_obs, global_map_depth=1, global_map_size, global_map_size)
            egocentric_global_map: torch.tensor containing registered map, (num_obs, global_map_depth=1, global_map_size, global_map_size)
        """
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        global_allocentric_map = global_allocentric_map.to(self.device)
        bs = egocentric_map.shape[0]
        
        if bs != self.num_obs:
            global_allocentric_map[bs:, ...] = global_allocentric_map[bs:, ...] * 0.
        global_allocentric_map = global_allocentric_map[:bs, ...].clone().detach().unsqueeze(dim=1)
        
        agent_view = torch.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)

        agent_view[:, :, 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ] = egocentric_map

        st_pose = torch.cat(
            [
                -(grid_y.unsqueeze(1) - (self.global_map_size//2))/(self.global_map_size//2), 
                -(grid_x.unsqueeze(1) - (self.global_map_size//2))/(self.global_map_size//2), 
                observations['compass']
            ], 
            dim=1
        ).to(self.device)
        
        # Generate warpping matrix for pytorch
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        
        # Warpping for global allocentric map
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True, mode='nearest')
        translated = F.grid_sample(rotated, trans_mat, align_corners=True, mode='nearest')
        registered_map = global_allocentric_map + translated # * 0.5 # 
        
        return registered_map.squeeze(dim=1) # , trajectory_map
    
    def forward_seg(self, 
        observations: Dict[str, torch.Tensor], 
        global_allocentric_map: torch.Tensor, 
        egocentric_map: torch.Tensor
    ):
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        global_allocentric_map = global_allocentric_map.clone().detach().permute(0, 3, 1, 2).to(self.device)
        bs, c, _, _ = global_allocentric_map.shape
        if bs != self.num_obs:
            global_allocentric_map[bs:, ...] = global_allocentric_map[bs:, ...] * 0.
        
        agent_view = torch.FloatTensor(bs, c, self.global_map_size, self.global_map_size).to(self.device).fill_(0)

        agent_view[:, :, 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
            self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
        ] = egocentric_map

        st_pose = torch.cat(
            [
                -(grid_y.unsqueeze(1) - (self.global_map_size//2))/(self.global_map_size//2), 
                -(grid_x.unsqueeze(1) - (self.global_map_size//2))/(self.global_map_size//2), 
                observations['compass']
            ], 
            dim=1
        ).to(self.device)
        
        # Generate warpping matrix for pytorch
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        
        # Warpping for global allocentric map
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        registered_map = (global_allocentric_map + translated) # B X C X H X W
        
        return registered_map.permute(0, 2, 3, 1)

class SemanticMap():
    def __init__(self,
            global_map_size, egocentric_map_size, num_process, num_classes,
            device, coordinate_min, coordinate_max, height_min=-0.7, height_max=0.8
        ):
        self.num_process = num_process
        self.global_map_size = global_map_size
        self.egocentric_map_size = egocentric_map_size
        self.num_classes = num_classes
        self.global_allocentric_semantic_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size,
            num_classes
        )
        self.device = device
        self.projection_sem = SemanticObsProjection(
            self.egocentric_map_size, self.global_map_size, device, 
            coordinate_min, coordinate_max, height_min, height_max, num_classes
        )
        self.registration = Registration(
            self.egocentric_map_size, self.global_map_size, coordinate_min, coordinate_max, 
            self.num_process, device
        )
    
    def reset_index(self, idx):
        self.global_allocentric_semantic_map[idx, ...] = torch.zeros(
            self.global_map_size,
            self.global_map_size,
            self.num_classes
        )
    
    def reset(self):
        self.global_allocentric_semantic_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size,
            self.num_classes
        )
    
    def get_current_global_maps(self) -> torch.Tensor:
        gasm = torch.argmax(self.global_allocentric_semantic_map, dim=-1)
        gasm = F.one_hot(gasm.long(), num_classes=self.num_classes).long()
        return gasm
    
    def update_map(self, observations: Dict[str, torch.Tensor]):
        depth = observations['depth']
        semantic = observations['semantic']
        sem_gt_torch = F.one_hot(semantic.long(), num_classes=self.num_classes).long().permute(0, 3, 1, 2)
        projection_res_sem = self.projection_sem.forward(sem_gt_torch, depth * 10.)
        self.global_allocentric_semantic_map = \
                self.registration.forward_seg(observations, self.global_allocentric_semantic_map, projection_res_sem)
        self.global_allocentric_semantic_map[..., 0] = 0
        
        
class OccupancyMap():
    
    def __init__(self, 
            global_map_size, egocentric_map_size, num_process,
            device, coordinate_min, coordinate_max, vaccant_bel, occupied_bel, 
            height_min=-0.8, height_max=1.5
        ):
        self.num_process = num_process
        self.global_map_size = global_map_size
        self.egocentric_map_size = egocentric_map_size
        self.BEL_VAC = vaccant_bel
        self.BEL_OCC = occupied_bel
        self.device = device
        ## global egocentric/allocentric map: (!!!) This is belief map, not the probability
        self.global_allocentric_occupancy_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size
        )
        self.occupancy_projection = OccupancyProjection(
            self.egocentric_map_size, self.global_map_size, device, 
            coordinate_min, coordinate_max, self.BEL_VAC, self.BEL_OCC, 
            height_min, height_max
        )
        self.registration = Registration(
            self.egocentric_map_size, self.global_map_size, coordinate_min, coordinate_max, 
            self.num_process, device
        )

    def reset_index(self, idx):
        self.global_allocentric_occupancy_map[idx, ...] = torch.zeros(
            self.global_map_size,
            self.global_map_size,
        )

    def reset(self):
        self.global_allocentric_occupancy_map = torch.zeros(
            self.num_process,
            self.global_map_size,
            self.global_map_size
        )

    def _belief_to_prob(self, x: torch.Tensor) -> torch.Tensor:
        return 1. - 1./(1 + torch.exp(x))

    def get_current_global_maps(self) -> torch.Tensor:
        '''
            Transform the belief map toward probability map
            (???????: Is it better to train the belief map?)
            
            Return:
                - Global Allocentric Map \in (batch size, channel=1, global map size, global map size)
                - Global Egocentric Map \in (batch size, channel=1, global map size, global map size)
            Map contents 
            M_{i,j} = 0: Vaccant
            M_{i,j} = 1: Occupied
            Otherwise: uncertainty between them.
        '''
        gaom = self._belief_to_prob(self.global_allocentric_occupancy_map)
        return gaom

    def update_map(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = observations["depth"].clone().detach()
        
        # Project to 2D map & Generate 2D local egocentric map
        projection = self.occupancy_projection.forward(depth * 10.)
        # Update global allocentric map & Get global egocentric map
        self.global_allocentric_occupancy_map = self.registration.forward(observations, self.global_allocentric_occupancy_map, projection)
        

def crop_global_map(global_map, gps_coord, compass, grid_t: to_grid, global_map_size, ego_map_size, device):
    """
    global semantic map: B X C X H X W
    global occupancy map: B X 1 X H X W
    """
    global_map_t = global_map.clone().detach().float()
    grid_x, grid_y = grid_t.get_grid_coords(gps_coord)
    st_pose = torch.cat(
        [
            (grid_y.unsqueeze(1) - (global_map_size//2))/(global_map_size//2),
            (grid_x.unsqueeze(1) - (global_map_size//2))/(global_map_size//2),
            -compass
        ], 
        dim=1
    ).to(device)
    rot_mat, trans_mat = get_grid(st_pose, global_map_t.size(), device)
        
    # Warpping for global allocentric map
    rotated = F.grid_sample(global_map_t, rot_mat, align_corners=True, mode='nearest')
    translated = F.grid_sample(rotated, trans_mat, align_corners=True, mode='nearest')
    
    c_h, c_w = global_map_size // 2, global_map_size // 2
    return translated[:, :,
        c_h - ego_map_size // 2: c_h + ego_map_size // 2, 
        c_w - ego_map_size // 2: c_w + ego_map_size // 2
    ].permute(0, 2, 3, 1)