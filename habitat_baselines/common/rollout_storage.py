#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
from random import sample
from typing import Dict, List

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class ReplayData():
    """
    Here we define the one-batch update data (s, a, r, s^{'})
    The elements are torch.Tensor for NUM_PROCESS batch size.
    
    s_t and action: observation['rgb', 'depth', 'semantic', 'gps', 'compass'], local_occupancy_map and local_semantic_map
    r: reward
    s^{'}: s_{t+1}
    
    Important: The Crop ratio need to be re-considered. Now it is same with egocentric map size
    """
    def __init__(self,
        observations_curr,
        local_occupancy_map_curr,
        local_semantic_map_curr,
        action, 
        reward,
        observations_next,
        local_occupancy_map_next,
        local_semantic_map_next,
        done_mask,
        device
    ):
        """
        observations['rgb']: B X C X H X W
        observations['depth']: B X 1 X H X W
        observations['semantic']: B X C X H X W
        observations['gps']: B X 2
        observations['compass']: B X 1
        local_occupancy_map: B X H X W
        local_semantic_map: B X C X H X W
        """
        self.observations_curr = observations_curr
        self.local_occupancy_map_curr = local_occupancy_map_curr
        self.local_semantic_map_curr = local_semantic_map_curr
        self.action = action
        self.reward = reward
        self.observations_next = observations_next
        self.local_occupancy_map_next = local_occupancy_map_next
        self.local_semantic_map_next = local_semantic_map_next
        self.done_mask = done_mask
        self.device = device
    
    def _to_obs(self, batch: Dict[str, torch.Tensor]):
        return { k: v.to(self.device) for k, v in batch.items() }
    
    @property
    def data(self):
        return (
            self._to_obs(self.observations_curr),
            self.local_occupancy_map_curr.to(self.device),
            self.local_semantic_map_curr.to(self.device),
            self.action.to(self.device),
            self.reward.to(self.device),
            self._to_obs(self.observations_next),
            self.local_occupancy_map_next.to(self.device),
            self.local_semantic_map_next.to(self.device),
            self.done_mask.to(self.device)
        )

class ReplayBuffer:
    """
    store the mini-batches for Deep Q Network
    """
    def __init__(self, 
        num_process,
        num_mini_batch,
        device
    ):
        self.num_process = num_process
        self.num_mini_batch = num_mini_batch
        self.replay = []
        self.device_cpu = torch.device('cpu')
        self.device_gpu = device
    
    def __len__(self,):
        return len(self.replay)
    
    def reset(self):
        self.replay.clear()
    
    def insert(self,
        observations_curr,
        local_occupancy_map_curr,
        local_semantic_map_curr,
        action,
        reward,
        observations_next,
        local_occupancy_map_next,
        local_semantic_map_next,
        done_mask
    ):
        data = ReplayData(
            self._to_obs(observations_curr),
            local_occupancy_map_curr.to(self.device_cpu),
            local_semantic_map_curr.to(self.device_cpu),
            action.to(self.device_cpu),
            reward.to(self.device_cpu),
            self._to_obs(observations_next),
            local_occupancy_map_next.to(self.device_cpu),
            local_semantic_map_next.to(self.device_cpu),
            done_mask.to(self.device_cpu),
            device=self.device_gpu
        )
        self.replay.append(data)

    def _to_obs(self, batch: Dict[str, torch.Tensor]):
        return { k: v.to(self.device_cpu) for k, v in batch.items() }

    def _make_batch(self, obs_list: List[Dict[str, torch.Tensor]]):
        obs_new_batch = { k: [] for k, _ in obs_list[0].items() }
        for obs in obs_list:
            for k, v in obs.items():
                obs_new_batch[k].append(v)
        return { k: torch.cat(v, dim=0) for k, v in obs_new_batch.items() }

    def sample_mini_batch(self):
        mini_batches = self.replay if len(self) < self.num_mini_batch \
                                    else sample(self.replay, self.num_mini_batch)
        
        list_obs_curr = []
        list_occ_map_curr = []
        list_sem_map_curr = []
        list_action_curr = []
        list_reward_curr = []
        list_obs_next = []
        list_occ_map_next = []
        list_sem_map_next = []
        list_done_mask = []
        for _, data in enumerate(mini_batches):
            (
                observations_curr,
                local_occupancy_map_curr,
                local_semantic_map_curr,
                actions, 
                reward,
                observations_next,
                local_occupancy_map_next,
                local_semantic_map_next,
                done_mask
            ) = data.data
            list_obs_curr.append(observations_curr)
            list_occ_map_curr.append(local_occupancy_map_curr)
            list_sem_map_curr.append(local_semantic_map_curr)
            list_action_curr.append(actions)
            list_reward_curr.append(reward)
            list_obs_next.append(observations_next)
            list_occ_map_next.append(local_occupancy_map_next)
            list_sem_map_next.append(local_semantic_map_next)
            list_done_mask.append(done_mask)

        obs_curr = self._make_batch(list_obs_curr)
        occ_curr = torch.cat(list_occ_map_curr, dim=0)
        sem_curr = torch.cat(list_sem_map_curr, dim=0)
        action_curr = torch.cat(list_action_curr, dim=0)
        reward_curr = torch.cat(list_reward_curr, dim=0)
        obs_next = self._make_batch(list_obs_next)
        occ_next = torch.cat(list_occ_map_next, dim=0)
        sem_next = torch.cat(list_sem_map_next, dim=0)
        mask = torch.cat(list_done_mask, dim=0)
        
        return (
            obs_curr,
            occ_curr,
            sem_curr,
            action_curr,
            reward_curr,
            obs_next,
            occ_next,
            sem_next,
            mask
        )
        