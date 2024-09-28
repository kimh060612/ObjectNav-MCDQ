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
    def __init__(self, main_q, target_q, dim_actions, confidence):
        super(QAgent, self).__init__()
        self.q_net = main_q
        self.target = target_q
        self.dim_actions = dim_actions
        self.confidence = confidence
    
    ## 이미 confidence로 uncertainty를 고려한 exploration 조절이 가능함. Epsilion 필요 노
    ## We can already manage the trade-off between exploration and exploitation. There is no need for eps-greedy
    def _get_actions(self, q_map_mean, q_map_var):
        ucb_policy = q_map_mean + self.confidence * q_map_var # B X num_act
        # TODO: Does UCB is enough? Try Lower Confidence Bound & Other Methods
        ## Thompson Sampling can be the candidates
        action = torch.argmax(ucb_policy, dim=1)
        return action # B X 1
    
    def act(self, observations, global_allocentric_map, global_semantic_map):
        q_map_mean, q_map_var = self.q_net(observations, global_allocentric_map, global_semantic_map)
        return self._get_actions(q_map_mean, q_map_var)
    
    def get_q_main(self, observations, global_allocentric_map, global_semantic_map, actions):
        q_map_mean, q_map_var = self.q_net(observations, global_allocentric_map, global_semantic_map)
        actions = actions.unsqueeze(1)
        return torch.gather(q_map_mean, 1, actions), torch.gather(q_map_var, 1, actions)
    
    def get_target(self, observations, global_allocentric_map, global_semantic_map):
        q_map = self.target(observations, global_allocentric_map, global_semantic_map)
        q_max_t, _ = torch.max(q_map, dim=1)
        return q_max_t.unsqueeze(1)

    def update_target(self):
        self.target.Q_net.load_state_dict(self.q_net.Q_net.state_dict())


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
            ), action_space.n, confidence
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
        self.Q_net = QNet(num_classes, True, dim_actions, True, self.mc_drop_rate, is_target)
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

    def forward(self, observations, global_allocentric_map, global_semantic_map):
        target_obj_id = self.get_target_encoding(observations).long()
        
        b, _, _, _ = global_semantic_map.shape
        # print(global_semantic_map.shape)
        # print(target_obj_id, observations[self.goal_sensor_uuid].shape)
        # print(global_allocentric_map.shape)
        global_goal_index = torch.zeros_like(global_allocentric_map).to(self.device)
        global_semantic_index = torch.argmax(global_semantic_map, dim=-1).long()
        # print(global_goal_index.shape)
        global_goal_index[global_semantic_index == target_obj_id.view(b, 1, 1)] = 1.
        global_map_final = torch.cat([
            global_allocentric_map.permute(0, 3, 1, 2),
            global_semantic_map.permute(0, 3, 1, 2),
            global_goal_index.permute(0, 3, 1, 2)
        ], dim=1).float()
        
        if not self.is_target:
            q_map_list = []
            for _ in range(self.num_mc_drop):
                q_map_list.append(self.Q_net(global_map_final))
            q_map_stack = torch.stack(q_map_list, dim=1)
            q_map_mean = torch.mean(q_map_stack, dim=1) # B x num_act
            q_map_var = torch.var(q_map_stack, dim=1) # B x num_act
            
            return q_map_mean, q_map_var
        else:
            return self.Q_net(global_map_final)