from statistics import mean
import math

import supersuit as ss
import numpy as np
import torch
from gym.spaces import Dict as GymDict, Box
from pettingzoo.magent import battle_v3_view7

from pretrained.magent import IDQN_Battle
from .multiagentenv import MultiAgentEnv
from .magent import PettingZooEnv

REGISTRY = {}
REGISTRY["battle_view7"] = battle_v3_view7.parallel_env

processed_channel_dim_dict = {"battle_view7": (9, 2, 2)}

MAPSIZE2N = {
    25: 20,
    35: 42,
    40: 64,
    45: 81,
    50: 100,
    55: 121,
    60: 144,
    70: 196,
    80: 256,
    90: 324,
    100: 400,
}


class _BattleWrapper(MultiAgentEnv):

    def __init__(self, **env_config):
        map = env_config.pop("map_name", None)
        # pretrained_ckpt = env_config.pop("pretrained_ckpt", None)
        self.seed = env_config.pop("seed", None)
        self.episode_limit = env_config["max_cycles"]

        env = REGISTRY[map](**env_config)
        # since init do not accept seed as an input, we need to pass it here
        env.seed(seed=self.seed)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        # wrap with parallel wrapper, but original version is parralel env wrapped with from_parallel wrapper, but no api ref https://github.com/Farama-Foundation/PettingZoo/blob/1.12.0/pettingzoo/magent/battle_v3.py
        # env = to_parallel_wrapper(env)
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = PettingZooEnv(env)

        self.processed_channel_dim = processed_channel_dim_dict[map][1]
        self.raw_channel_dim = processed_channel_dim_dict[map][0]  # before processed
        self.state_channel_dim = processed_channel_dim_dict[map][2]
        self.action_space = self.env.action_space

        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    dtype=self.env.observation_space.dtype,
                ),
                "state": Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    dtype=self.env.observation_space.dtype,
                ),
            }
        )
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obss = self.env.reset()
        obs = []
        positions = []
        for a in self.agents:
            obs.append(obss[a].flatten())
            positions.append(
                np.zeros(2)
            )  # the first step all hidden values are 0, so it does not matter whom to communicate
        self._obs = obs
        self._positions = positions
        self._episode_length = 0

        # very bad coding, find a way to fix later
        self.env.par_env.aec_env.env.env.env.env.env

        return self.get_obs(), self.get_state()

    def step(self, actions):
        rewards = []
        obs = []
        positions = []
        action_dict = {}
        # convert list of actions to dict
        for agent, action in zip(self.agents, actions):
            if isinstance(action, torch.Tensor):
                act = action.item()
            else:
                act = action
            action_dict[agent] = act
        obss, rews, dones, pos_infos = self.env.step(action_dict)
        self.red_team_alives = 0
        self.blue_team_alives = 0

        for agent in self.agents:
            rewards.append(rews[agent])
            if agent in obss:
                obs.append(obss[agent].flatten())
                positions.append(pos_infos[agent])
                if agent.startswith("red"):
                    self.red_team_alives += 1
                elif agent.startswith("blue"):
                    self.blue_team_alives += 1
            else:
                obs.append(np.zeros(math.prod(self.observation_space["obs"].shape)))
                positions.append(np.zeros(2))

        self._obs = obs
        self._positions = positions
        done = dones["__all__"]

        info = {}
        info["red_team_alives"] = self.red_team_alives
        info["blue_team_alives"] = self.blue_team_alives
        if done or self._episode_length >= self.episode_limit:
            info["red_team_win"] = self.red_team_alives > self.blue_team_alives
            info["blue_team_win"] = self.blue_team_alives > self.red_team_alives
            info["episode_length"] = self._episode_length
        else:
            self._episode_length += 1

        return rewards, done, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        obs_processed = np.array(self._obs).reshape(
            self.n_agents,
            self.observation_space["obs"].shape[0],
            self.observation_space["obs"].shape[1],
            self.raw_channel_dim,
        )
        my_team_hp = obs_processed[:, :, :, 2] - obs_processed[:, :, :, 0]
        other_team_hp = obs_processed[:, :, :, 5] - obs_processed[:, :, :, 0]
        obs_processed = np.concatenate((my_team_hp, other_team_hp), axis=-1)
        obs_processed = obs_processed.reshape(self.n_agents, -1)

        # return self._obs
        return obs_processed

    def get_positions(self):
        # (n_agents, 2)
        return np.array(self._positions).flatten()

    def get_state(self):
        obs = np.array(self._obs).reshape(
            self.n_agents,
            self.observation_space["obs"].shape[0],
            self.observation_space["obs"].shape[1],
            self.raw_channel_dim,
        )
        my_team_minimap = obs[:, :, :, 3]
        other_team_minimap = obs[:, :, :, 6]
        state = np.concatenate((my_team_minimap, other_team_minimap), axis=-1)
        state = state.reshape(self.n_agents, -1)

        # return state of the first agent(from the red team)
        return state[0]

    def get_avail_actions(self):
        avail_actions = []
        for _ in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions()
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self):
        valid = self.action_space.n * [1]
        return valid

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_env_info(self):
        env_info = {
            # "state_shape": math.prod(self.observation_space["state"].shape),  # flatten
            "state_shape": math.prod(self.observation_space["obs"].shape[:2])
            * self.state_channel_dim,  # flatten
            "obs_shape": math.prod(self.observation_space["obs"].shape[:2])
            * self.processed_channel_dim,  # flatten
            "n_actions": self.action_space.n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

        return env_info


class Battle_w_PretrainedOpp(MultiAgentEnv):
    def __init__(self, **env_config):
        map_size = env_config["map_size"]
        pretrained_ckpt = env_config.pop("pretrained_ckpt", None)
        self.global_reward = env_config.pop("global_reward", False)

        self._env = _BattleWrapper(**env_config)

        self.seed = self._env.seed
        self.episode_limit = self._env.episode_limit

        self.n_agents = MAPSIZE2N[map_size]
        self.input_shape = 98

        # there are two teams, red(first) and blue(second), the pretrained policy controls the blue team
        self.blue_agents_policy = IDQN_Battle(
            pretrained_ckpt=pretrained_ckpt, input_shape=self.input_shape
        )
        self.blue_team_obss = None

    def reset(self):
        obs, state = self._env.reset()
        self.blue_team_obss = obs[-self.n_agents :]
        return obs[: -self.n_agents], state

    def step(self, actions):
        blue_team_avali_actions = self._env.get_avail_actions()[-self.n_agents :]
        blue_team_actions = self.blue_agents_policy.step(
            obss=self.blue_team_obss, avail_actions=blue_team_avali_actions
        )
        # actions should be of shape (n_agents, ),
        actions = torch.tensor(actions, dtype=torch.long)
        blue_team_actions = blue_team_actions.to(actions.device)
        all_actions = torch.cat([actions, blue_team_actions], dim=0)
        rew, done, info = self._env.step(all_actions)
        obs = self._env.get_obs()
        self.blue_team_obss = obs[-self.n_agents :]
        obs = obs[: -self.n_agents]
        rew = rew[: -self.n_agents]
        if self.global_reward:
            rew = [mean(rew)] * len(rew)
        return rew, done, info

    def get_obs(self):
        return self._env.get_obs()[: -self.n_agents]

    def get_positions(self):
        all_possitions = self._env.get_positions().reshape(self.n_agents * 2, 2)
        return all_possitions[: -self.n_agents].flatten()

    def get_state(self):
        return self._env.get_state()

    def get_avail_actions(self):
        return self._env.get_avail_actions()[: -self.n_agents]

    def get_env_info(self):
        env_info = self._env.get_env_info()
        # overwrite n_agents
        env_info["n_agents"] = self.n_agents

        return env_info

    def get_stats(self):
        pass

    def render(self, mode="human"):
        return self._env.render(mode)

    def close(self):
        self._env.close()
