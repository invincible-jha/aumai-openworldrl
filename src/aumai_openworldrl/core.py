"""Core RL implementations: GridWorldEnv, QLearningAgent, Trainer, ExperienceReplay.

GridWorldEnv is a deterministic 2-D grid with walls, a start, and a goal.
QLearningAgent uses tabular Q-learning with epsilon-greedy exploration.
ExperienceReplay is a fixed-size circular buffer for experience storage.
Trainer orchestrates training loops and evaluation.
"""

from __future__ import annotations

import collections
import random
from contextlib import contextmanager
from typing import Iterator, Optional

from .models import Environment, Experience, RLAgent, TrainingResult


# ---------------------------------------------------------------------------
# GridWorldEnv
# ---------------------------------------------------------------------------


class GridWorldEnv:
    """Deterministic 2-D grid world environment.

    Layout: The agent starts at (0, 0) and must reach (width-1, height-1).
    Walls are randomly placed and excluded from start/goal positions.

    Actions:
        0 = up, 1 = right, 2 = down, 3 = left

    Example:
        >>> env_config = Environment(env_id="gw1", width=5, height=5)
        >>> env = GridWorldEnv(env_config, seed=42)
        >>> state = env.reset()
        >>> state, reward, done, info = env.step(1)
    """

    _ACTIONS: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    _ACTION_NAMES = ["up", "right", "down", "left"]

    def __init__(self, config: Environment, wall_density: float = 0.1, seed: Optional[int] = None) -> None:
        """Initialise the environment.

        Args:
            config: Environment configuration.
            wall_density: Fraction of cells that are walls.
            seed: Optional random seed.
        """
        self._config = config
        self._rng = random.Random(seed)
        self._walls: set[tuple[int, int]] = set()
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (config.height - 1, config.width - 1)
        self._steps_taken: int = 0
        self._wall_density = wall_density
        self._generate_walls()

    def reset(self) -> tuple[int, int]:
        """Reset the environment to the start state.

        Returns:
            The initial agent position as (row, col).
        """
        self._agent_pos = (0, 0)
        self._steps_taken = 0
        return self._agent_pos

    def step(
        self, action: int
    ) -> tuple[tuple[int, int], float, bool, dict[str, object]]:
        """Take one environment step.

        Args:
            action: Integer action index (0=up, 1=right, 2=down, 3=left).

        Returns:
            (next_state, reward, done, info) tuple.
        """
        self._steps_taken += 1
        dr, dc = self._ACTIONS[action % 4]
        row, col = self._agent_pos
        new_row = row + dr
        new_col = col + dc

        # Boundary and wall check
        if (
            0 <= new_row < self._config.height
            and 0 <= new_col < self._config.width
            and (new_row, new_col) not in self._walls
        ):
            self._agent_pos = (new_row, new_col)
            reward = self._config.step_penalty
        else:
            reward = self._config.wall_penalty

        done = False
        if self._agent_pos == self._goal_pos:
            reward += self._config.goal_reward
            done = True
        elif self._steps_taken >= self._config.max_steps:
            done = True

        info: dict[str, object] = {
            "steps": self._steps_taken,
            "reached_goal": self._agent_pos == self._goal_pos,
        }
        return self._agent_pos, reward, done, info

    def render(self) -> str:
        """Render the grid as ASCII text.

        Returns:
            Multi-line string representation of the grid.
        """
        rows: list[str] = []
        for r in range(self._config.height):
            row_chars: list[str] = []
            for c in range(self._config.width):
                pos = (r, c)
                if pos == self._agent_pos:
                    row_chars.append("A")
                elif pos == self._goal_pos:
                    row_chars.append("G")
                elif pos in self._walls:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        return "\n".join(rows)

    def _generate_walls(self) -> None:
        """Randomly populate walls, avoiding start and goal positions."""
        for r in range(self._config.height):
            for c in range(self._config.width):
                pos = (r, c)
                if pos in {(0, 0), self._goal_pos}:
                    continue
                if self._rng.random() < self._wall_density:
                    self._walls.add(pos)


# ---------------------------------------------------------------------------
# ExperienceReplay
# ---------------------------------------------------------------------------


class ExperienceReplay:
    """Fixed-capacity circular experience replay buffer.

    Example:
        >>> buffer = ExperienceReplay(capacity=1000)
        >>> buffer.push(Experience(state=(0,0), action=1, reward=0.1,
        ...                        next_state=(0,1), done=False))
        >>> batch = buffer.sample(32)
    """

    def __init__(self, capacity: int = 1000) -> None:
        """Initialise with fixed capacity.

        Args:
            capacity: Maximum number of experiences to store.
        """
        self._buffer: collections.deque[Experience] = collections.deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """Add one experience to the buffer.

        Args:
            experience: The transition to store.
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int, rng: Optional[random.Random] = None) -> list[Experience]:
        """Sample a random mini-batch.

        Args:
            batch_size: Number of samples to draw.
            rng: Optional seeded RNG.

        Returns:
            List of Experience objects (with replacement if buffer is small).
        """
        r = rng or random
        population = list(self._buffer)
        if len(population) <= batch_size:
            return population
        return r.sample(population, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    The Q-table is a dict mapping (state, action) pairs to float values.
    States are made hashable by converting tuples to tuples.

    Example:
        >>> agent_config = RLAgent(agent_id="q1", num_actions=4)
        >>> agent = QLearningAgent(agent_config, seed=42)
        >>> action = agent.select_action((0, 0))
        >>> agent.update((0, 0), action, 0.1, (0, 1), False)
    """

    def __init__(self, config: RLAgent, seed: Optional[int] = None) -> None:
        """Initialise the agent.

        Args:
            config: RLAgent configuration.
            seed: Optional random seed.
        """
        self._config = config
        self._rng = random.Random(seed)
        self._q_table: dict[tuple[object, int], float] = {}

    def select_action(self, state: object) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state (must be hashable).

        Returns:
            Integer action index.
        """
        if self._rng.random() < self._config.epsilon:
            return self._rng.randrange(self._config.num_actions)
        return self._greedy_action(state)

    def update(
        self,
        state: object,
        action: int,
        reward: float,
        next_state: object,
        done: bool,
    ) -> float:
        """Apply the Q-learning update rule.

        Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether the episode ended.

        Returns:
            The TD error (before update).
        """
        current_q = self._q_table.get((state, action), 0.0)
        if done:
            target = reward
        else:
            best_next = max(
                self._q_table.get((next_state, a), 0.0)
                for a in range(self._config.num_actions)
            )
            target = reward + self._config.discount_factor * best_next

        td_error = target - current_q
        self._q_table[(state, action)] = current_q + self._config.learning_rate * td_error
        return td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate by epsilon_decay, down to epsilon_min."""
        new_epsilon = max(
            self._config.epsilon_min,
            self._config.epsilon * self._config.epsilon_decay,
        )
        self._config = self._config.model_copy(update={"epsilon": new_epsilon})

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        """Context manager that sets epsilon to 0.0 for greedy evaluation.

        Saves the current epsilon value before entering and restores it
        unconditionally on exit via try/finally.
        """
        saved = self._config.epsilon
        self._config = self._config.model_copy(update={"epsilon": 0.0})
        try:
            yield
        finally:
            self._config = self._config.model_copy(update={"epsilon": saved})

    def _greedy_action(self, state: object) -> int:
        """Return the action with highest Q-value for *state*."""
        q_values = [
            self._q_table.get((state, a), 0.0) for a in range(self._config.num_actions)
        ]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return self._rng.choice(best_actions)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._config.epsilon

    @property
    def q_table_size(self) -> int:
        """Number of (state, action) entries in the Q-table."""
        return len(self._q_table)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrate Q-learning training and evaluation on GridWorldEnv.

    Example:
        >>> env_config = Environment(env_id="gw1", width=5, height=5, max_steps=100)
        >>> agent_config = RLAgent(agent_id="q1", num_actions=4)
        >>> trainer = Trainer(env_config, agent_config, seed=0)
        >>> result = trainer.train(episodes=500)
    """

    def __init__(
        self,
        env_config: Environment,
        agent_config: RLAgent,
        seed: Optional[int] = None,
        replay_capacity: int = 1000,
    ) -> None:
        """Initialise the trainer.

        Args:
            env_config: Environment configuration.
            agent_config: Agent configuration.
            seed: Optional random seed.
            replay_capacity: Size of the experience replay buffer.
        """
        self._env_config = env_config
        self._agent_config = agent_config
        self._seed = seed
        self._env = GridWorldEnv(env_config, seed=seed)
        self._agent = QLearningAgent(agent_config, seed=seed)
        self._replay = ExperienceReplay(capacity=replay_capacity)

    def train(self, episodes: int = 1000) -> TrainingResult:
        """Train the agent for a fixed number of episodes.

        Args:
            episodes: Number of training episodes.

        Returns:
            TrainingResult with summary statistics.
        """
        reward_history: list[float] = []
        total_steps = 0
        success_count = 0

        for episode in range(episodes):
            state = self._env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self._agent.select_action(state)
                next_state, reward, done, info = self._env.step(action)

                self._replay.push(
                    Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )
                )

                self._agent.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

            self._agent.decay_epsilon()
            reward_history.append(round(episode_reward, 4))

            if info.get("reached_goal"):
                success_count += 1

        last_100 = reward_history[-100:] if len(reward_history) >= 100 else reward_history
        mean_reward = sum(last_100) / len(last_100) if last_100 else 0.0
        best_reward = max(reward_history) if reward_history else 0.0

        return TrainingResult(
            agent_id=self._agent_config.agent_id,
            total_episodes=episodes,
            total_steps=total_steps,
            mean_reward=round(mean_reward, 4),
            best_reward=round(best_reward, 4),
            success_rate=round(success_count / episodes, 4),
            final_epsilon=round(self._agent.epsilon, 6),
            reward_history=reward_history,
        )

    def evaluate(self, episodes: int = 100, render_first: bool = False) -> TrainingResult:
        """Evaluate the trained agent greedily (epsilon=0).

        Args:
            episodes: Number of evaluation episodes.
            render_first: If True, print the grid for the first episode.

        Returns:
            TrainingResult with evaluation statistics.
        """
        reward_history: list[float] = []
        success_count = 0
        total_steps = 0

        with self._agent.eval_mode():
            for episode in range(episodes):
                state = self._env.reset()
                episode_reward = 0.0
                done = False
                show = render_first and episode == 0

                while not done:
                    if show:
                        print(self._env.render())
                        print()
                    action = self._agent.select_action(state)
                    next_state, reward, done, info = self._env.step(action)
                    state = next_state
                    episode_reward += reward
                    total_steps += 1

                reward_history.append(round(episode_reward, 4))
                if info.get("reached_goal"):
                    success_count += 1

        last_100 = reward_history[-100:]
        mean_reward = sum(last_100) / len(last_100) if last_100 else 0.0

        return TrainingResult(
            agent_id=self._agent_config.agent_id,
            total_episodes=episodes,
            total_steps=total_steps,
            mean_reward=round(mean_reward, 4),
            best_reward=max(reward_history) if reward_history else 0.0,
            success_rate=round(success_count / episodes, 4),
            final_epsilon=0.0,
            reward_history=reward_history,
        )
