"""Comprehensive tests for aumai-openworldrl core module.

Covers: GridWorldEnv, ExperienceReplay, QLearningAgent, Trainer
and the Pydantic models Environment, Experience, RLAgent, TrainingResult.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from aumai_openworldrl.core import (
    ExperienceReplay,
    GridWorldEnv,
    QLearningAgent,
    Trainer,
)
from aumai_openworldrl.models import (
    Environment,
    Experience,
    RLAgent,
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env_config() -> Environment:
    return Environment(env_id="test_env", width=5, height=5, max_steps=50)


@pytest.fixture()
def agent_config() -> RLAgent:
    return RLAgent(agent_id="test_agent", num_actions=4)


@pytest.fixture()
def env(env_config: Environment) -> GridWorldEnv:
    return GridWorldEnv(env_config, wall_density=0.0, seed=42)


@pytest.fixture()
def agent(agent_config: RLAgent) -> QLearningAgent:
    return QLearningAgent(agent_config, seed=42)


@pytest.fixture()
def replay_buffer() -> ExperienceReplay:
    return ExperienceReplay(capacity=100)


# ---------------------------------------------------------------------------
# Model tests — Environment
# ---------------------------------------------------------------------------


class TestEnvironmentModel:
    def test_defaults(self) -> None:
        env = Environment(env_id="e1")
        assert env.width == 5
        assert env.height == 5
        assert env.max_steps == 200
        assert env.goal_reward == 1.0
        assert env.step_penalty == -0.01
        assert env.wall_penalty == -0.05

    def test_custom_fields(self) -> None:
        env = Environment(
            env_id="e2", width=10, height=8, max_steps=300,
            goal_reward=5.0, step_penalty=-0.1, wall_penalty=-0.5,
        )
        assert env.width == 10
        assert env.height == 8
        assert env.goal_reward == 5.0

    def test_width_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Environment(env_id="bad", width=0)

    def test_height_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Environment(env_id="bad", height=-1)

    def test_env_id_stored(self) -> None:
        env = Environment(env_id="unique_id_123")
        assert env.env_id == "unique_id_123"

    def test_num_actions_default(self) -> None:
        env = Environment(env_id="e3")
        assert env.num_actions == 4

    def test_serialisation_roundtrip(self) -> None:
        env = Environment(env_id="ser_test", width=7, height=7)
        data = env.model_dump()
        restored = Environment(**data)
        assert restored == env


# ---------------------------------------------------------------------------
# Model tests — Experience
# ---------------------------------------------------------------------------


class TestExperienceModel:
    def test_basic_creation(self) -> None:
        exp = Experience(state=(0, 0), action=1, reward=-0.01, next_state=(0, 1), done=False)
        assert exp.state == (0, 0)
        assert exp.action == 1
        assert exp.done is False

    def test_terminal_experience(self) -> None:
        exp = Experience(state=(3, 4), action=2, reward=1.0, next_state=(4, 4), done=True)
        assert exp.done is True
        assert exp.reward == 1.0

    def test_action_can_be_zero(self) -> None:
        exp = Experience(state=(0, 0), action=0, reward=0.0, next_state=(0, 0), done=False)
        assert exp.action == 0


# ---------------------------------------------------------------------------
# Model tests — RLAgent
# ---------------------------------------------------------------------------


class TestRLAgentModel:
    def test_defaults(self) -> None:
        agent = RLAgent(agent_id="a1")
        assert agent.epsilon == 1.0
        assert agent.epsilon_min == 0.01
        assert agent.epsilon_decay == 0.995
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.99
        assert agent.num_actions == 4

    def test_invalid_learning_rate_zero(self) -> None:
        with pytest.raises(ValidationError):
            RLAgent(agent_id="bad", learning_rate=0.0)

    def test_invalid_epsilon_above_one(self) -> None:
        with pytest.raises(ValidationError):
            RLAgent(agent_id="bad", epsilon=1.1)

    def test_invalid_num_actions_zero(self) -> None:
        with pytest.raises(ValidationError):
            RLAgent(agent_id="bad", num_actions=0)


# ---------------------------------------------------------------------------
# Model tests — TrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResultModel:
    def test_defaults(self) -> None:
        result = TrainingResult(agent_id="a1")
        assert result.total_episodes == 0
        assert result.success_rate == 0.0
        assert result.reward_history == []

    def test_invalid_success_rate_above_one(self) -> None:
        with pytest.raises(ValidationError):
            TrainingResult(agent_id="a1", success_rate=1.5)


# ---------------------------------------------------------------------------
# GridWorldEnv tests
# ---------------------------------------------------------------------------


class TestGridWorldEnv:
    def test_reset_returns_start(self, env: GridWorldEnv) -> None:
        state = env.reset()
        assert state == (0, 0)

    def test_step_returns_four_tuple(self, env: GridWorldEnv) -> None:
        env.reset()
        result = env.step(1)  # right
        assert len(result) == 4
        next_state, reward, done, info = result
        assert isinstance(next_state, tuple)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_info_has_steps(self, env: GridWorldEnv) -> None:
        env.reset()
        _, _, _, info = env.step(0)
        assert "steps" in info
        assert info["steps"] == 1

    def test_step_info_has_reached_goal(self, env: GridWorldEnv) -> None:
        env.reset()
        _, _, _, info = env.step(1)
        assert "reached_goal" in info

    def test_wall_penalty_on_boundary_hit(self, env: GridWorldEnv) -> None:
        env.reset()
        # Action 0 = up, from (0,0) should hit boundary
        _, reward, _, _ = env.step(0)
        assert reward == env._config.wall_penalty

    def test_step_penalty_on_valid_move(self, env: GridWorldEnv) -> None:
        env.reset()
        # Action 1 = right, should move to (0,1)
        _, reward, _, _ = env.step(1)
        assert reward == env._config.step_penalty

    def test_goal_reached(self) -> None:
        cfg = Environment(env_id="small", width=2, height=2, max_steps=100)
        small_env = GridWorldEnv(cfg, wall_density=0.0, seed=0)
        small_env.reset()
        # Grid 2x2: start (0,0), goal (1,1). Move right then down.
        small_env.step(1)  # -> (0,1)
        _, reward, done, info = small_env.step(2)  # down -> (1,1)
        assert done is True
        assert info["reached_goal"] is True
        assert reward == cfg.step_penalty + cfg.goal_reward

    def test_max_steps_terminates_episode(self) -> None:
        cfg = Environment(env_id="limit", width=5, height=5, max_steps=3)
        e = GridWorldEnv(cfg, wall_density=0.0, seed=0)
        e.reset()
        for _ in range(2):
            _, _, done, _ = e.step(0)  # keep hitting wall, no movement
            assert not done
        _, _, done, _ = e.step(0)
        assert done is True

    def test_reset_clears_steps(self, env: GridWorldEnv) -> None:
        env.reset()
        env.step(0)
        env.step(0)
        env.reset()
        _, _, _, info = env.step(0)
        assert info["steps"] == 1

    def test_agent_stays_in_bounds_x(self, env: GridWorldEnv) -> None:
        env.reset()
        for _ in range(10):
            env.step(3)  # left — should stay at col 0
        assert env._agent_pos[1] == 0

    def test_agent_stays_in_bounds_y(self, env: GridWorldEnv) -> None:
        env.reset()
        for _ in range(10):
            env.step(0)  # up — should stay at row 0
        assert env._agent_pos[0] == 0

    def test_render_contains_agent_marker(self, env: GridWorldEnv) -> None:
        env.reset()
        rendered = env.render()
        assert "A" in rendered

    def test_render_contains_goal_marker(self, env: GridWorldEnv) -> None:
        env.reset()
        rendered = env.render()
        assert "G" in rendered

    def test_render_dimensions(self, env: GridWorldEnv) -> None:
        env.reset()
        rendered = env.render()
        lines = rendered.split("\n")
        assert len(lines) == env._config.height

    def test_wall_density_zero_no_walls(self) -> None:
        cfg = Environment(env_id="clear", width=5, height=5)
        e = GridWorldEnv(cfg, wall_density=0.0, seed=0)
        assert len(e._walls) == 0

    def test_wall_density_full_excludes_start_goal(self) -> None:
        cfg = Environment(env_id="dense", width=5, height=5)
        e = GridWorldEnv(cfg, wall_density=1.0, seed=0)
        assert (0, 0) not in e._walls
        assert e._goal_pos not in e._walls

    def test_goal_position_is_bottom_right(self) -> None:
        cfg = Environment(env_id="g", width=5, height=5)
        e = GridWorldEnv(cfg, wall_density=0.0)
        assert e._goal_pos == (4, 4)

    def test_action_modulo_wraps(self, env: GridWorldEnv) -> None:
        """Actions beyond 3 should wrap via modulo 4."""
        env.reset()
        # Action 4 should behave like action 0 (up)
        state4, reward4, done4, _ = env.step(4)
        env.reset()
        state0, reward0, done0, _ = env.step(0)
        assert state4 == state0
        assert reward4 == reward0


# ---------------------------------------------------------------------------
# ExperienceReplay tests
# ---------------------------------------------------------------------------


class TestExperienceReplay:
    def test_initial_empty(self, replay_buffer: ExperienceReplay) -> None:
        assert len(replay_buffer) == 0

    def test_push_increments_length(self, replay_buffer: ExperienceReplay) -> None:
        exp = Experience(state=(0, 0), action=0, reward=0.0, next_state=(0, 1), done=False)
        replay_buffer.push(exp)
        assert len(replay_buffer) == 1

    def test_capacity_enforced(self) -> None:
        buf = ExperienceReplay(capacity=5)
        for i in range(10):
            buf.push(Experience(state=i, action=0, reward=0.0, next_state=i + 1, done=False))
        assert len(buf) == 5

    def test_sample_returns_list(self, replay_buffer: ExperienceReplay) -> None:
        for i in range(20):
            replay_buffer.push(Experience(state=i, action=0, reward=0.0, next_state=i + 1, done=False))
        batch = replay_buffer.sample(10)
        assert isinstance(batch, list)
        assert len(batch) == 10

    def test_sample_returns_all_when_small(self) -> None:
        buf = ExperienceReplay(capacity=100)
        for i in range(3):
            buf.push(Experience(state=i, action=0, reward=0.0, next_state=i + 1, done=False))
        batch = buf.sample(10)
        assert len(batch) == 3

    def test_sample_with_seeded_rng(self) -> None:
        buf = ExperienceReplay(capacity=100)
        for i in range(50):
            buf.push(Experience(state=i, action=0, reward=0.0, next_state=i + 1, done=False))
        rng = random.Random(42)
        batch1 = buf.sample(10, rng=random.Random(42))
        batch2 = buf.sample(10, rng=random.Random(42))
        assert [e.state for e in batch1] == [e.state for e in batch2]

    def test_push_experiences_are_retrievable(self) -> None:
        buf = ExperienceReplay(capacity=5)
        exp = Experience(state=(1, 2), action=3, reward=0.5, next_state=(1, 3), done=False)
        buf.push(exp)
        batch = buf.sample(1)
        assert batch[0] == exp

    def test_oldest_overwritten_at_capacity(self) -> None:
        buf = ExperienceReplay(capacity=3)
        for i in range(4):
            buf.push(Experience(state=i, action=0, reward=float(i), next_state=i + 1, done=False))
        states = [e.state for e in buf.sample(3)]
        assert 0 not in states  # state=0 was evicted


# ---------------------------------------------------------------------------
# QLearningAgent tests
# ---------------------------------------------------------------------------


class TestQLearningAgent:
    def test_select_action_in_range(self, agent: QLearningAgent) -> None:
        for _ in range(50):
            action = agent.select_action((0, 0))
            assert 0 <= action < agent._config.num_actions

    def test_update_returns_td_error(self, agent: QLearningAgent) -> None:
        td = agent.update((0, 0), 1, 0.5, (0, 1), False)
        assert isinstance(td, float)

    def test_update_changes_q_table(self, agent: QLearningAgent) -> None:
        initial_size = agent.q_table_size
        agent.update((5, 5), 2, 0.1, (5, 6), False)
        assert agent.q_table_size > initial_size

    def test_terminal_update_uses_reward_only(self, agent: QLearningAgent) -> None:
        # When done=True, target = reward (no bootstrap)
        agent.update((9, 9), 0, 10.0, (9, 9), True)
        q_val = agent._q_table.get(((9, 9), 0), 0.0)
        # Q(s,a) = 0 + lr * (10.0 - 0) = 1.0
        assert abs(q_val - 1.0) < 1e-6

    def test_epsilon_decay_reduces_epsilon(self, agent: QLearningAgent) -> None:
        initial_eps = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial_eps

    def test_epsilon_never_below_min(self) -> None:
        cfg = RLAgent(agent_id="a", epsilon=0.011, epsilon_min=0.01, epsilon_decay=0.1)
        a = QLearningAgent(cfg)
        for _ in range(100):
            a.decay_epsilon()
        assert a.epsilon >= cfg.epsilon_min

    def test_greedy_policy_when_epsilon_zero(self) -> None:
        cfg = RLAgent(agent_id="a", epsilon=0.0, num_actions=4)
        a = QLearningAgent(cfg, seed=7)
        # Set Q(s,2) = 10
        a._q_table[((0, 0), 2)] = 10.0
        for _ in range(20):
            assert a.select_action((0, 0)) == 2

    def test_q_table_size_property(self, agent: QLearningAgent) -> None:
        assert agent.q_table_size == 0
        agent.update((0, 0), 0, 0.0, (0, 1), False)
        assert agent.q_table_size == 1

    def test_epsilon_property(self, agent_config: RLAgent) -> None:
        a = QLearningAgent(agent_config, seed=0)
        assert a.epsilon == agent_config.epsilon

    def test_update_td_error_sign(self, agent: QLearningAgent) -> None:
        # If Q is 0 and reward is positive, TD error should be positive
        td = agent.update((7, 7), 3, 1.0, (7, 8), True)
        assert td > 0

    def test_update_bootstraps_from_next_state(self, agent: QLearningAgent) -> None:
        # Set a high Q value at next state
        agent._q_table[((1, 1), 0)] = 5.0
        td = agent.update((0, 0), 0, 0.0, (1, 1), False)
        # target = 0 + 0.99 * 5.0 = 4.95; td_error = 4.95 - 0 = 4.95
        assert abs(td - 4.95) < 1e-4


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_train_returns_training_result(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=10)
        assert isinstance(result, TrainingResult)

    def test_train_episode_count_matches(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=20)
        assert result.total_episodes == 20

    def test_train_reward_history_length(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=15)
        assert len(result.reward_history) == 15

    def test_train_total_steps_positive(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=5)
        assert result.total_steps > 0

    def test_train_success_rate_in_range(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=10)
        assert 0.0 <= result.success_rate <= 1.0

    def test_train_final_epsilon_less_than_initial(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=50)
        assert result.final_epsilon < agent_config.epsilon

    def test_evaluate_returns_training_result(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        trainer.train(episodes=50)
        result = trainer.evaluate(episodes=5)
        assert isinstance(result, TrainingResult)

    def test_evaluate_epsilon_is_zero(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        trainer.train(episodes=50)
        result = trainer.evaluate(episodes=5)
        assert result.final_epsilon == 0.0

    def test_evaluate_restores_epsilon(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        trainer.train(episodes=50)
        eps_before = trainer._agent.epsilon
        trainer.evaluate(episodes=5)
        assert abs(trainer._agent.epsilon - eps_before) < 1e-9

    def test_train_seeded_reproducibility(self, env_config: Environment, agent_config: RLAgent) -> None:
        t1 = Trainer(env_config, agent_config, seed=99)
        t2 = Trainer(env_config, agent_config, seed=99)
        r1 = t1.train(episodes=10)
        r2 = t2.train(episodes=10)
        assert r1.reward_history == r2.reward_history

    def test_agent_id_propagated(self, env_config: Environment) -> None:
        cfg = RLAgent(agent_id="custom_agent_id")
        trainer = Trainer(env_config, cfg, seed=0)
        result = trainer.train(episodes=5)
        assert result.agent_id == "custom_agent_id"

    def test_best_reward_ge_mean_reward(self, env_config: Environment, agent_config: RLAgent) -> None:
        trainer = Trainer(env_config, agent_config, seed=0)
        result = trainer.train(episodes=20)
        assert result.best_reward >= result.mean_reward


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


@given(
    width=st.integers(min_value=2, max_value=10),
    height=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=30, deadline=5000)
def test_gridworld_goal_always_set(width: int, height: int) -> None:
    cfg = Environment(env_id="prop", width=width, height=height)
    env = GridWorldEnv(cfg, wall_density=0.0, seed=0)
    assert env._goal_pos == (height - 1, width - 1)


@given(capacity=st.integers(min_value=1, max_value=50))
@settings(max_examples=20)
def test_replay_never_exceeds_capacity(capacity: int) -> None:
    buf = ExperienceReplay(capacity=capacity)
    for i in range(capacity * 3):
        buf.push(Experience(state=i, action=0, reward=0.0, next_state=i + 1, done=False))
    assert len(buf) <= capacity


@given(
    lr=st.floats(min_value=0.01, max_value=1.0),
    reward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=5000)
def test_qlearning_update_finite(lr: float, reward: float) -> None:
    cfg = RLAgent(agent_id="hyp", learning_rate=lr)
    agent = QLearningAgent(cfg, seed=0)
    td = agent.update((0, 0), 0, reward, (0, 1), True)
    assert not isinstance(td, float) or (td == td)  # not NaN
