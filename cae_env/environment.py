import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Dict, Any, Tuple, Optional
from cae_env.types import Message, HarmCategory, Action
from cae_env.messages import generate_message, shuffle_episode_categories, difficulty_for_step
from cae_env.users import build_users
from cae_env.reward import compute_reward
from cae_env.negotiation import NegotiationEngine

class OmniAlignEnv(gym.Env):
    """OmniAlign platform moderation simulation environment."""
    def __init__(self, num_users: int = 5, max_steps: int = 30, task: str = "basic"):
        super().__init__()
        self.num_users = num_users
        self.max_steps = max_steps
        self.task = task
        self.current_step = 0
        self.group_health = 0.5
        self.episode_log: List[Dict[str, Any]] = []
        self.rng = np.random.RandomState(42)
        self.users = build_users(num_users, self.rng)
        self.negotiation = NegotiationEngine()
        self.history = [np.zeros(128, dtype=np.float32) for _ in range(5)]
        
        obs_dim = 128 + num_users + 640 + 4 + 15 + 11 + 3 + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.pending_message: Optional[Message] = None
        self._category_schedule = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None: self.rng = np.random.RandomState(seed)
        self.current_step = 0
        self.group_health = 0.5
        self.episode_log = []
        self.history = [np.zeros(128, dtype=np.float32) for _ in range(5)]
        self._category_schedule = shuffle_episode_categories(self.task, self.max_steps, seed or 42)
        self._generate_next_message()
        return self._get_obs(), {"task": self.task}

    def _get_obs(self) -> np.ndarray:
        msg = self.pending_message
        if msg is None: return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        curr_emb = np.array(msg.embedding, dtype=np.float32)
        trusts = np.array([u.profile.trust_weight for u in self.users], dtype=np.float32)
        hist = np.concatenate(self.history, dtype=np.float32)
        media_vh = np.zeros(4, dtype=np.float32); media_vh[0] = 1.0
        lang_vh = np.zeros(15, dtype=np.float32); lang_vh[0] = 1.0
        harm_probs = np.zeros(11, dtype=np.float32)
        harm_idx = [c.value for c in HarmCategory].index(msg.ground_truth_type.value)
        harm_probs[harm_idx] = msg.primary_strength
        harm_probs = np.exp(harm_probs) / np.sum(np.exp(harm_probs))
        scalars = np.array([self.group_health, 0.1, self.current_step / self.max_steps], dtype=np.float32)
        obs = np.concatenate([curr_emb, trusts, hist, media_vh, lang_vh, harm_probs, scalars, trusts])
        if len(obs) < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)))
        return obs[:self.observation_space.shape[0]]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        msg = self.pending_message
        reward = compute_reward(action, msg.ground_truth_type, 0.0, False, False)
        self.group_health = np.clip(self.group_health + reward * 0.01, 0, 1)
        self.history.append(np.array(msg.embedding, dtype=np.float32))
        if len(self.history) > 5: self.history.pop(0)
        self.episode_log.append({"step": self.current_step, "action": action, "reward": reward, "ground_truth_type": msg.ground_truth_type.value})
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if not terminated: self._generate_next_message()
        return self._get_obs(), float(reward), terminated, False, {"group_health": self.group_health}

    def _generate_next_message(self):
        cat = self._category_schedule[self.current_step]
        self.pending_message = generate_message(self.current_step, 0, self.task, self.num_users, self.rng, category=cat)

    def state(self) -> Dict[str, Any]:
        return {"step": self.current_step, "health": self.group_health}
    def render(self, mode='human'): pass
    def seed(self, seed=None): self.rng = np.random.RandomState(seed)
