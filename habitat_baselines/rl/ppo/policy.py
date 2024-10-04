#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN

from habitat_baselines.rl.models.mcd_qnet import QNet # , QNet512, Q_discrete

class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

class QAgent(nn.Module):
    def __init__(self, main_q, target_q, dim_actions, confidence, local_map_size):
        super(QAgent, self).__init__()
        self.q_net = main_q
        self.target = target_q
        self.dim_actions = dim_actions
        self.confidence = confidence
        self.w_occ = local_map_size
        self.h_occ = local_map_size
    
    ## 이미 confidence로 uncertainty를 고려한 exploration 조절이 가능함. Epsilion 필요 노
    ## We can already manage the trade-off between exploration and exploitation. There is no need for eps-greedy
    ## Action Sequence를 구성하는 방법 vs Point Goal Navigation w. DQN
    ### 우선 action sequence 구성이 먼저 진행
    ### Map anticipation은 나중에
    def _get_actions(self, q_map_mean, q_map_var):
        ucb_policy = q_map_mean + self.confidence * q_map_var # B X num_act
        # TODO: Does UCB is enough? Try Lower Confidence Bound & Other Methods
        ## Thompson Sampling can be the candidates
        # b, _ = ucb_policy.shape
        # import random
        # sample = random.random()
        action = torch.argmax(ucb_policy, dim=1)
        return action # B X 1
    
    def _get_map_point(self, q_map_mean, q_map_var):
        ucb_policy = q_map_mean + self.confidence * q_map_var
        tmp = torch.argmax(ucb_policy.view(1, -1)) # B X 1
        a_w = tmp - ((tmp / self.w_occ).long() * self.w_occ)
        a_h = ((tmp - a_w) / self.w_occ).long()
        map_point = torch.cat( [ a_h, a_w ], dim = 1)
        return map_point # B X 2
    
    def _get_action(self, map_point):
        bsz, _ = map_point.shape
        source = torch.Tensor([self.h_occ // 2, self.w_occ // 2]).unsqueeze(0).repeat(bsz, 1)
        
        action_list = []
        angle = torch.zeros((bsz, 1))
        if map_point[:, 1] > source[:, 1]:
            if map_point[:, 0] < source[:, 0]:
                angle = torch.arctan((map_point[:, 1] - source[:, 1]) / (source[:, 0] - map_point[:, 0]))
            else:
                angle = torch.arctan((map_point[:, 0] - source[:, 0]) / (map_point[:, 1] - source[:, 1]))
                angle += torch.pi / 2.
        elif map_point[0] < source[0]:
            angle = torch.arctan((source[:, 1] - map_point[:, 1]) / (source[:, 0] - map_point[:, 0]))
            angle = -angle
        else:
            angle = torch.arctan((map_point[:, 0] - source[:, 0]) / (source[:, 1] - map_point[:, 1]))
            angle = -np.pi/2 - angle
        action_list.append(-angle)
        action_list.append("MOVE_FORWARD")
        return action_list
    
    def act(self, observations, global_allocentric_map, global_semantic_map):
        q_map_mean, q_map_var = self.q_net(observations, global_allocentric_map, global_semantic_map)
        actions = self._get_actions(q_map_mean, q_map_var)
        return actions # only use stop action when agent gets close to the goal
        # , torch.gather(q_map_mean, 1, actions - 1)
        
    def get_q_main(self, observations, global_allocentric_map, global_semantic_map, actions):
        q_map_mean, q_map_var = self.q_net(observations, global_allocentric_map, global_semantic_map)
        actions = actions.unsqueeze(1)
        return torch.gather(q_map_mean, 1, actions), torch.gather(q_map_var, 1, actions)
    
    def get_target(self, observations, global_allocentric_map, global_semantic_map):
        q_map = self.target(observations, global_allocentric_map, global_semantic_map)
        q_max_t, _ = torch.max(q_map, dim=1)
        return q_max_t.unsqueeze(1)

    def update_target(self):
        for param_a, param_b in zip(self.target.Q_net.parameters(), self.q_net.Q_net.parameters()):
            param_b.data.copy_(param_a.data)


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
            ),
            action_space.n,
        )

class ObjectNavUncertainQAgent(QAgent):
    def __init__(self, 
            goal_sensor_uuid,
            num_local_steps,
            num_mc_drop,
            mc_drop_rate,
            device,
            num_classes,
            local_map_size,
            # global_map_size,
            # coord_min, coord_max,
            action_space, confidence
        ):
        super(ObjectNavUncertainQAgent, self).__init__(
            UncertainGoalQNet(
                goal_sensor_uuid,
                num_local_steps,
                num_mc_drop,
                mc_drop_rate,
                device,
                num_classes,
                action_space.n,
                False
            ),
            UncertainGoalQNet(
                goal_sensor_uuid,
                num_local_steps,
                num_mc_drop,
                mc_drop_rate,
                device,
                num_classes,
                action_space.n,
                True
            ), action_space.n, confidence, local_map_size
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class UncertainGoalQNet(nn.Module):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self,
            goal_sensor_uuid,
            num_local_steps,
            num_mc_drop,
            mc_drop_rate,
            device,
            num_classes,
            dim_actions,
            is_target,
            # global_map_size,
            # coord_min, coord_max
        ):
        super().__init__()
        self.num_mc_drop = num_mc_drop
        self.mc_drop_rate = mc_drop_rate
        self.num_local_steps = num_local_steps
        self.is_target = is_target
        
        self.goal_sensor_uuid = goal_sensor_uuid
        self.device = device
        self.Q_net = QNet(num_classes, True, dim_actions, False, self.mc_drop_rate, is_target)
        self.goal_embedding = nn.Embedding(21, 72)
        # self.transform_pos_grid = to_grid(global_map_size, coord_min, coord_max)

        ## (TODO): Things to consider
        ## 1. How much the map covers? ==> 40m coverage for now
        ## 2. Using whole global map or local map? => Local goal prediction or Global goal prediction
        
        ## Things to implement
        ## 1. RGB Segmentation model (MobileNetv3-small) 
        ##      -> Implemented in Trainer, Not in the policy network to reduce GPU utilization.
        ## 2. Ground projection for Semantic & Occupancy Map
        #### 2-1. (TODO) Auxiliary Task for map anticipation 
        #### 2-2. (TODO) Occupancy reward for navigation
        ## 3. MC-Dropout Policy => output is heatmap where goal is likely to be.
        ## 4. Rollout Storage for off-policy PPO or Q learning
        ## 5. Off-policy learning algorithm
        
        ## 9/27 결정 사항
        ## PPO 버림. Bayesian DQN으로 방향 전환.
        ## Bayesian DQN으로 바꾸는건 나중에. 현재는 MC dropout을 통하여 
        
        self.train()

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    # def count_ones_in_circle(self, img, radius):
    #     B, H, W, _ = img.shape
    #     y_center, x_center = H // 2, W // 2

    #     # 각 좌표의 (y, x) 인덱스 생성
    #     # print(B, H, W)
    #     y_indices = torch.arange(0, H).unsqueeze(1).float().to(self.device)
    #     x_indices = torch.arange(0, W).unsqueeze(0).float().to(self.device)

    #     # 중심점에서 각 좌표까지의 거리 계산
    #     distance_squared = (y_indices - y_center)**2 + (x_indices - x_center)**2
    #     mask = distance_squared <= radius**2
    #     mask = mask.float().unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).to(self.device)
        
    #     count = torch.sum(img.float() * mask, dim=(1, 2, 3))
    #     return count


    def forward(self, observations, local_allocentric_map, local_semantic_map):
        target_obj_id = self.get_target_encoding(observations).long()
        goal_emb = self.goal_embedding(target_obj_id).squeeze(1)
        # b, h, w, _ = local_semantic_map.shape
        # target_label_map = target_obj_id.view(b, 1, 1).expand(-1, h, w).clone().detach()
        # local_goal_index = torch.zeros_like(local_allocentric_map).to(self.device)
        # local_semantic_index = torch.argmax(local_semantic_map, dim=-1).long()
        # mask = (local_semantic_index == target_label_map).unsqueeze(-1)
        # local_goal_index += mask.long()
        local_map_final = torch.cat([
            local_allocentric_map.permute(0, 3, 1, 2),
            local_semantic_map.permute(0, 3, 1, 2),
            # local_goal_index.permute(0, 3, 1, 2)
        ], dim=1).float()
        
        if not self.is_target:
            # is_close = self.count_ones_in_circle(local_goal_index, radius=4.) < 3. # goal 근처가 아니면 1, goal 근처면 0
            q_map_list = []
            for _ in range(self.num_mc_drop):
                q_map_list.append(self.Q_net(local_map_final, goal_emb))
            q_map_stack = torch.stack(q_map_list, dim=1)
            q_map_mean = torch.mean(q_map_stack, dim=1) # B x num_act
            q_map_var = torch.var(q_map_stack, dim=1) # B x num_act
            
            return q_map_mean, q_map_var
        else:
            return self.Q_net(local_map_final, goal_emb)