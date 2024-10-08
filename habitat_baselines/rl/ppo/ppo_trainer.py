#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage, ReplayBuffer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO, PointNavBaselinePolicy, ObjectNavUncertainQAgent
from habitat_baselines.rl.models.geometry import OccupancyMap, SemanticMap, to_grid, crop_global_map
from habitat_baselines.rl.models.mobilenet import get_mobilenet_v3_small_seg
from h5py import File as fs
from typing import Dict

@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

@baseline_registry.register_trainer(name="uncertain-q")
class QNetUncertainTrainer(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_dqn_agent(self, dqn_cfg: Config, map_cfg: Config, device, num_process) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        
        ### Debugging: env.py를 확인하면서 조사
        self.occ_map_generator = OccupancyMap(
            map_cfg.global_map_size, map_cfg.egocentric_map_size,
            num_process, device,
            map_cfg.coordinate_min, map_cfg.coordinate_max,
            map_cfg.vacant_belief, map_cfg.occupied_belief
        )
        self.sem_map_generator = SemanticMap(
            map_cfg.global_map_size, map_cfg.egocentric_map_size,
            num_process, dqn_cfg.num_classes, device,
            map_cfg.coordinate_min, map_cfg.coordinate_max,
            map_cfg.vacant_belief, map_cfg.occupied_belief
        )
        self.grid_transform = to_grid(map_cfg.global_map_size, map_cfg.coordinate_min, map_cfg.coordinate_max)
        
        ### Segmentation을 위하여 이미지 크기를 조정하는 것도 방법이라고 생각함.
        ### labeling이 얼마나 성능에 큰 파이를 차지하는지 조사
        ### Seraph로 Segmentation을 이전한 뒤에 우선 semantic GT를 사용하여 진행해봄.
        ### 나중에 이 라인의 주석을 풀고 segmentation module을 삽입해야함.
        # self.semantic_encoder = get_mobilenet_v3_small_seg(dqn_cfg.num_classes, dqn_cfg.semantic_pretrain, True).to(self.device)
        self.q_agent = ObjectNavUncertainQAgent(
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            num_local_steps=dqn_cfg.num_local_steps,
            num_mc_drop=dqn_cfg.num_mc_drop,
            mc_drop_rate=dqn_cfg.mc_drop_rate,
            device=device,
            num_classes=dqn_cfg.num_classes,
            action_space=self.envs.action_spaces[0],
            confidence=dqn_cfg.confidence_rate,
            local_map_size=map_cfg.egocentric_map_size
        )
        self.q_agent.to(self.device)
        self.loss_fn = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.q_agent.parameters())),
            lr=dqn_cfg.lr,
            eps=dqn_cfg.eps,
            weight_decay=dqn_cfg.weight_decay
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.q_agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _unpack_batch(self, batch, map_cfg):
        ### 1번 문제: Segmentation이 그지같은 성능을 보임
        ### 2번 문제: 또 또 Projection Module이 잘못됨
        batch['rgb'] = batch['rgb'].permute(0, 3, 1, 2)
        ### 우선 Feasiblity를 보기 위하여 ground truth semantic image를 사용해 보는 것도 방법이라고 생각함.
        ### habitat/core/env.py의 reset/step method를 수정함: 나중에 이를 지우고 segmentation module로써 진행함.
        # sem_img = self.semantic_encoder(batch['rgb'].to(self.device))
        # batch['semantic'] = torch.argmax(sem_img[0], dim=1)
        self.occ_map_generator.update_map(batch)
        self.sem_map_generator.update_map(batch)
        global_allocentric_map = self.occ_map_generator.get_current_global_maps()
        global_semantic_map = self.sem_map_generator.get_current_global_maps()
        
        # if not map_cfg.map_save_debug == '':
        #     with fs(f"{map_cfg.map_save_debug}/gdata_{len(self.replay_buf)}.h5", 'w') as f:
        #         f.create_dataset('occ_map', data=global_allocentric_map.cpu().numpy(), dtype=np.float32)
        #         f.create_dataset('sem_map', data=global_semantic_map.cpu().numpy(), dtype=np.float32)
        #         f.create_dataset('semantic', data=batch['semantic'].cpu().numpy(), dtype=np.float32)
        
        ### Cropping the map to egocentric
        local_occ_map = crop_global_map(
            global_allocentric_map.unsqueeze(1), # B X 1 X H x W
            batch['gps'], batch['compass'], self.grid_transform,
            map_cfg.global_map_size, map_cfg.egocentric_map_size, self.device
        )
        local_semantic_map = crop_global_map(
            global_semantic_map.permute(0, 3, 1, 2), # B X C X H x W
            batch['gps'], batch['compass'], self.grid_transform,
            map_cfg.global_map_size, map_cfg.egocentric_map_size, self.device
        ).long()
        return batch, local_occ_map, local_semantic_map

    def _entropy(self, x):
        x = x + 1e-10
        entropy = -torch.sum(x * torch.log(x), dim=(1, 2, 3))
        return entropy
    
    def _get_coverage(self, x: torch.Tensor):
        mask_occ = x > 0.7
        mask_free = x < 0.3
        mask = (mask_occ + mask_free).long()
        return torch.sum(mask, dim=(1, 2, 3)) # B X 1
    
    def _get_gain_reward(self, m_curr, m_next, map_size):
        return (self._get_coverage(m_next) - self._get_coverage(m_curr)) / (map_size * map_size)

    def _one_step_forward(self, 
        obs_curr, occ_map_curr, sem_map_curr, 
        current_episode_reward, running_episode_stats, map_cfg
    ):
        pth_time = 0.0
        env_time = 0.0
        t_sample_action = time.time()
        with torch.no_grad():
            actions = self.q_agent.act(obs_curr, occ_map_curr, sem_map_curr)
        
        pth_time += time.time() - t_sample_action
        t_step_env = time.time()

        outputs = self.envs.step([a.item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        
        env_time += time.time() - t_step_env
        
        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch, local_occ_map, local_semantic_map = self._unpack_batch(batch, map_cfg)
        for idx, done in enumerate(dones):
            if done:
                self.occ_map_generator.reset_index(idx)
                self.sem_map_generator.reset_index(idx)
                
        occ_reward = self._get_gain_reward(
            occ_map_curr, local_occ_map, map_cfg.egocentric_map_size
        ).unsqueeze(1).to(current_episode_reward.device)
        
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1) + map_cfg.entropy_coef * occ_reward
        
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        ) # done: 0.0 == Done, 1.0 == Not Done
        
        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        
        self.replay_buf.insert(
            obs_curr, occ_map_curr.clone().detach(), sem_map_curr.clone().detach(), actions, rewards.to(self.device),
            batch, local_occ_map.clone().detach(), local_semantic_map.clone().detach(), masks.to(self.device)
        )
        
        pth_time += time.time() - t_update_stats
        return (
            batch, local_occ_map, local_semantic_map, 
            pth_time, env_time, self.envs.num_envs, dones
        )

    def _before_step(self):
        nn.utils.clip_grad_norm_(
            self.q_agent.parameters(), 0.2
        )

    def _update_agent(self, ppo_cfg):
        t_update_model = time.time()
        (
            obs_curr,
            occ_curr,
            sem_curr,
            action_curr,
            reward_curr,
            obs_next,
            occ_next,
            sem_next,
            mask
        ) = self.replay_buf.sample_mini_batch()

        q_main_pred, q_main_pred_var = self.q_agent.get_q_main(obs_curr, occ_curr, sem_curr, action_curr)
        q_max_target = self.q_agent.get_target(obs_next, occ_next, sem_next)
        target = reward_curr + ppo_cfg.discount * q_max_target
        loss = self.loss_fn(q_main_pred * mask, target * mask)
        loss.backward()
        self._before_step()
        self.optimizer.step()
        
        self.q_agent.update_target()
        return (
            time.time() - t_update_model,
            loss,
            torch.mean(q_main_pred_var, dim=0)
        )

    def _copy_batch(self, batch_obs: Dict[str, torch.Tensor]):
        new_obs = { k: v.clone().detach() for k, v in batch_obs.items() }
        return new_obs

    def train(self) -> None:
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        dqn_cfg = self.config.RL.DQN
        map_cfg = self.config.RL.MAPS
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        import random
        torch.manual_seed(987654321)
        torch.cuda.manual_seed(987654321)
        random.seed(987654321)
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_dqn_agent(dqn_cfg, map_cfg, self.device, self.config.NUM_PROCESSES)
        self.replay_buf = ReplayBuffer(self.config.NUM_PROCESSES, dqn_cfg.num_mini_batch, self.device)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.q_agent.parameters())
            )
        )
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch, local_occ_map, local_semantic_map = self._unpack_batch(batch, map_cfg)
        
        batch_curr = self._copy_batch(batch)
        local_occ_map_curr = local_occ_map.clone().detach()
        local_sem_map_curr = local_semantic_map.clone().detach()
        
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=dqn_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if dqn_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                q_loss = 0
                action_uncertainty = 0
                
                num_epsiode_iter = dqn_cfg.num_steps // dqn_cfg.num_local_steps
                for n_s in range(num_epsiode_iter):
                    
                    for _t in range(dqn_cfg.num_local_steps):
                        
                        (
                            batch_next,
                            local_occ_map_next,
                            local_semantic_map_next,
                            delta_pth_time,
                            delta_env_time,
                            delta_steps,
                            dones
                        ) = self._one_step_forward(
                            batch_curr, local_occ_map_curr, local_sem_map_curr, 
                            current_episode_reward, running_episode_stats, map_cfg
                        )
                        global_allocentric_map = self.occ_map_generator.get_current_global_maps()
                        global_semantic_map = self.sem_map_generator.get_current_global_maps()
                        if not map_cfg.map_save_debug == '':
                            with fs(f"{map_cfg.map_save_debug}/gdata_update{update}_{n_s * num_epsiode_iter + _t}.h5", 'w') as f:
                                f.create_dataset('occ_map', data=global_allocentric_map.cpu().numpy(), dtype=np.float32)
                                f.create_dataset('sem_map', data=global_semantic_map.cpu().numpy(), dtype=np.float32)
                                f.create_dataset('semantic', data=batch_curr['semantic'].cpu().numpy(), dtype=np.float32)
                                f.create_dataset('rgb', data=batch_curr['rgb'].cpu().numpy(), dtype=np.float32)
                                f.create_dataset('gps', data=batch_curr['gps'].cpu().numpy(), dtype=np.float32)
                                f.create_dataset('compass', data=batch_curr['compass'].cpu().numpy(), dtype=np.float32)
                                f.create_dataset('depth', data=batch_curr['depth'].cpu().numpy(), dtype=np.float32)
                                f.create_dataset('dones', data=np.array(dones), dtype=np.bool)
                        batch_curr = self._copy_batch(batch_next)
                        local_occ_map_curr = local_occ_map_next.clone().detach()
                        local_sem_map_curr = local_semantic_map_next.clone().detach()
                        
                        pth_time += delta_pth_time
                        env_time += delta_env_time
                        count_steps += delta_steps

                    (
                        delta_pth_time,
                        q_loss_t,
                        action_uncertainty_t
                    ) = self._update_agent(dqn_cfg)
                    pth_time += delta_pth_time
                    q_loss += q_loss_t.item()
                    action_uncertainty += action_uncertainty_t.item()

                    q_loss /= dqn_cfg.num_local_steps
                    action_uncertainty /= dqn_cfg.num_local_steps

                    if len(self.replay_buf) >= dqn_cfg.replay_buffer_size:
                        self.replay_buf.reset()
                    # self.occ_map_generator.reset()
                    # self.sem_map_generator.reset()
                    
                    for k, v in running_episode_stats.items():
                        window_episode_stats[k].append(v.clone())

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "reward", deltas["reward"] / deltas["count"], count_steps
                    )

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        writer.add_scalar("metrics/episode_length", metrics["episode_length"], count_steps)
                        writer.add_scalar("metrics/success", metrics["success"], count_steps)
                        writer.add_scalar("metrics/distance_to_goal", metrics["distance_to_goal"], count_steps)
                        writer.add_scalar("metrics/spl", metrics["spl"], count_steps)
                        
                    losses = [q_loss, action_uncertainty]
                    writer.add_scalars(
                        "q-training",
                        {k: l for l, k in zip(losses, ["q_loss", "action_uncertainty"])},
                        count_steps,
                    )

                    # log stats
                    num_updates = num_epsiode_iter * update + n_s
                    if num_updates % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                num_updates, count_steps / (time.time() - t_start)
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                num_updates, env_time, pth_time, count_steps
                            )
                        )
                        
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )
                        
                    # checkpoint model
                    if num_updates % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                        )
                        count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        map_cfg = config.RL.MAPS

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_dqn_agent(ppo_cfg, map_cfg, self.device, self.config.NUM_PROCESSES)

        self.q_agent.load_state_dict(ckpt_dict["state_dict"])

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch, local_occ_map, local_semantic_map = self._unpack_batch(batch, map_cfg)
        
        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            256,
            self.config.NUM_PROCESSES,
            512,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.q_agent.eval()
        
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                actions = self.q_agent.act(observations, local_occ_map, local_semantic_map)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            batch, local_occ_map, local_semantic_map = self._unpack_batch(batch, map_cfg)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            for idx, done in enumerate(dones):
                if done:
                    self.occ_map_generator.reset_index(idx)
                    self.sem_map_generator.reset_index(idx)

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()