"""Replay buffer with an explicit checkpointable state."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, n_agents):
        self.mem_size = int(max_size)
        self.n_agents = int(n_agents)
        self.state_memory = np.zeros((self.mem_size, int(input_shape) * self.n_agents), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, int(n_actions) * self.n_agents), dtype=np.float32)
        self.reward_global_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_task1 = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.reward_task2 = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.new_state_memory = np.zeros_like(self.state_memory)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.mem_cntr = 0

    @property
    def size(self):
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, state, action, reward_g, reward_t1, reward_t2, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = np.asarray(state, dtype=np.float32)
        self.action_memory[index] = np.asarray(action, dtype=np.float32)
        self.reward_global_memory[index] = float(reward_g)
        self.reward_task1[index] = np.asarray(reward_t1, dtype=np.float32)
        self.reward_task2[index] = np.asarray(reward_t2, dtype=np.float32)
        self.new_state_memory[index] = np.asarray(state_, dtype=np.float32)
        self.terminal_memory[index] = bool(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = self.size
        if max_mem < int(batch_size):
            raise ValueError("not enough replay transitions")
        indices = np.random.choice(max_mem, int(batch_size), replace=False)
        return (
            self.state_memory[indices].copy(),
            self.action_memory[indices].copy(),
            self.reward_global_memory[indices].copy(),
            self.reward_task1[indices].copy(),
            self.reward_task2[indices].copy(),
            self.new_state_memory[indices].copy(),
            self.terminal_memory[indices].copy(),
        )

    def state_dict(self):
        return {
            "mem_size": self.mem_size,
            "n_agents": self.n_agents,
            "mem_cntr": self.mem_cntr,
            "state_memory": self.state_memory.copy(),
            "action_memory": self.action_memory.copy(),
            "reward_global_memory": self.reward_global_memory.copy(),
            "reward_task1": self.reward_task1.copy(),
            "reward_task2": self.reward_task2.copy(),
            "new_state_memory": self.new_state_memory.copy(),
            "terminal_memory": self.terminal_memory.copy(),
        }

    def load_state_dict(self, state):
        if int(state["mem_size"]) != self.mem_size or int(state["n_agents"]) != self.n_agents:
            raise ValueError("checkpoint replay shape does not match config")
        self.mem_cntr = int(state["mem_cntr"])
        for key in ("state_memory", "action_memory", "reward_global_memory", "reward_task1", "reward_task2", "new_state_memory", "terminal_memory"):
            setattr(self, key, np.asarray(state[key]).copy())

