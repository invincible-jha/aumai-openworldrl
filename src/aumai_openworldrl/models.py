"""Pydantic v2 models for open-world reinforcement learning."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Environment(BaseModel):
    """Description of a reinforcement learning environment.

    Attributes:
        env_id: Unique identifier.
        name: Human-readable name.
        width: Grid width (for grid-world environments).
        height: Grid height.
        num_actions: Number of discrete actions available.
        max_steps: Maximum steps per episode.
        goal_reward: Reward for reaching the goal.
        step_penalty: Negative reward applied at each non-terminal step.
        wall_penalty: Negative reward for bumping into walls.
    """

    env_id: str
    name: str = Field(default="GridWorld")
    width: int = Field(default=5, gt=0)
    height: int = Field(default=5, gt=0)
    num_actions: int = Field(default=4, gt=0)
    max_steps: int = Field(default=200, gt=0)
    goal_reward: float = Field(default=1.0)
    step_penalty: float = Field(default=-0.01)
    wall_penalty: float = Field(default=-0.05)


class Experience(BaseModel):
    """A single (state, action, reward, next_state, done) transition.

    Attributes:
        state: Current state representation.
        action: Action taken.
        reward: Reward received.
        next_state: Resulting state after action.
        done: Whether the episode terminated.
    """

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class RLAgent(BaseModel):
    """Configuration and identity of a reinforcement learning agent.

    Attributes:
        agent_id: Unique identifier.
        algorithm: RL algorithm name.
        learning_rate: Q-learning update rate.
        discount_factor: Gamma for future reward discounting.
        epsilon: Current exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        num_actions: Size of the action space.
    """

    agent_id: str
    algorithm: str = Field(default="q_learning")
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    num_actions: int = Field(default=4, gt=0)


class TrainingResult(BaseModel):
    """Summary of a completed training run.

    Attributes:
        agent_id: The trained agent.
        total_episodes: Episodes completed.
        total_steps: Total environment steps taken.
        mean_reward: Mean episode reward over the last 100 episodes.
        best_reward: Best single-episode reward seen.
        success_rate: Fraction of episodes where goal was reached.
        final_epsilon: Epsilon at end of training.
        reward_history: Per-episode rewards.
    """

    agent_id: str
    total_episodes: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    mean_reward: float = Field(default=0.0)
    best_reward: float = Field(default=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    final_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_history: list[float] = Field(default_factory=list)
