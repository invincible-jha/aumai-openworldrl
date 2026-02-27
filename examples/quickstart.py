"""Quickstart examples for aumai-openworldrl.

Demonstrates the core API: environment setup, agent training, greedy evaluation,
experience replay inspection, and result serialization.

Run directly:

    python examples/quickstart.py

All demos use fixed seeds so output is reproducible.
"""

from __future__ import annotations

import json
import random

from aumai_openworldrl.core import (
    ExperienceReplay,
    GridWorldEnv,
    QLearningAgent,
    Trainer,
)
from aumai_openworldrl.models import Environment, Experience, RLAgent, TrainingResult


# ---------------------------------------------------------------------------
# Demo 1 — Basic training on a 5x5 grid
# ---------------------------------------------------------------------------


def demo_basic_training() -> TrainingResult:
    """Train a Q-learning agent on a small grid and print summary statistics."""
    print("=" * 60)
    print("Demo 1: Basic Training (5x5 grid, 1000 episodes)")
    print("=" * 60)

    env_config = Environment(
        env_id="basic_env",
        width=5,
        height=5,
        max_steps=200,
        goal_reward=1.0,
        step_penalty=-0.01,
        wall_penalty=-0.05,
    )
    agent_config = RLAgent(
        agent_id="basic_agent",
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        num_actions=4,
    )

    trainer = Trainer(env_config, agent_config, seed=42)
    result = trainer.train(episodes=1000)

    print(f"Total episodes  : {result.total_episodes}")
    print(f"Total steps     : {result.total_steps}")
    print(f"Mean reward     : {result.mean_reward:.4f}")
    print(f"Best reward     : {result.best_reward:.4f}")
    print(f"Success rate    : {result.success_rate:.1%}")
    print(f"Final epsilon   : {result.final_epsilon:.4f}")
    print(f"Q-table entries : {trainer._agent.q_table_size}")
    print()

    return result


# ---------------------------------------------------------------------------
# Demo 2 — Greedy evaluation with ASCII grid rendering
# ---------------------------------------------------------------------------


def demo_evaluation() -> None:
    """Pre-train for 800 episodes, then evaluate greedily and render first episode."""
    print("=" * 60)
    print("Demo 2: Greedy Evaluation with Grid Rendering")
    print("=" * 60)

    env_config = Environment(
        env_id="eval_env",
        width=5,
        height=5,
        max_steps=200,
    )
    agent_config = RLAgent(agent_id="eval_agent", num_actions=4)

    trainer = Trainer(env_config, agent_config, seed=7)

    print("Pre-training for 800 episodes...")
    trainer.train(episodes=800)

    print("Evaluating for 50 episodes (greedy policy, epsilon=0)...")
    print("First episode grid (A=agent, G=goal, #=wall, .=open):\n")
    result = trainer.evaluate(episodes=50, render_first=True)

    print(f"Evaluation success rate : {result.success_rate:.1%}")
    print(f"Evaluation mean reward  : {result.mean_reward:.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo 3 — eval_mode context manager
# ---------------------------------------------------------------------------


def demo_eval_mode() -> None:
    """Show that eval_mode temporarily fixes epsilon at 0 without side effects."""
    print("=" * 60)
    print("Demo 3: eval_mode Context Manager")
    print("=" * 60)

    agent_config = RLAgent(agent_id="ctx_agent", epsilon=0.5, num_actions=4)
    agent = QLearningAgent(agent_config, seed=0)

    print(f"Before eval_mode: epsilon = {agent.epsilon}")

    with agent.eval_mode():
        print(f"Inside eval_mode: epsilon = {agent.epsilon}")
        # Agent always acts greedily inside this block
        action = agent.select_action((0, 0))
        print(f"  Selected action (greedy): {action}")

    print(f"After eval_mode:  epsilon = {agent.epsilon}")
    print("Epsilon was correctly restored.\n")


# ---------------------------------------------------------------------------
# Demo 4 — Experience replay buffer
# ---------------------------------------------------------------------------


def demo_experience_replay() -> None:
    """Collect transitions from a random policy and inspect the replay buffer."""
    print("=" * 60)
    print("Demo 4: Experience Replay Buffer")
    print("=" * 60)

    env_config = Environment(env_id="replay_env", width=5, height=5)
    env = GridWorldEnv(env_config, seed=3)
    buffer = ExperienceReplay(capacity=200)

    rng = random.Random(3)
    state = env.reset()
    total_collected = 0

    for _ in range(150):
        action = rng.randrange(4)
        next_state, reward, done, info = env.step(action)
        buffer.push(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )
        total_collected += 1
        state = env.reset() if done else next_state

    print(f"Collected {total_collected} transitions")
    print(f"Buffer size     : {len(buffer)}")

    # Sample a mini-batch with a fixed seed for reproducibility
    sample_rng = random.Random(99)
    batch = buffer.sample(16, rng=sample_rng)
    print(f"Sampled {len(batch)} transitions")

    rewards = [e.reward for e in batch]
    goal_hits = sum(1 for e in batch if e.done and e.reward > 0)
    print(f"Avg batch reward: {sum(rewards) / len(rewards):.4f}")
    print(f"Goal reached    : {goal_hits} / {len(batch)} in batch")
    print()


# ---------------------------------------------------------------------------
# Demo 5 — JSON serialization and deserialization of results
# ---------------------------------------------------------------------------


def demo_serialization(result: TrainingResult) -> None:
    """Serialize a TrainingResult to JSON and reload it."""
    print("=" * 60)
    print("Demo 5: JSON Serialization")
    print("=" * 60)

    json_str = result.model_dump_json(indent=2)
    size_kb = len(json_str.encode()) / 1024
    print(f"Serialized size : {size_kb:.1f} KB")

    # Reload via Pydantic validation
    reloaded = TrainingResult.model_validate_json(json_str)

    assert reloaded.agent_id == result.agent_id
    assert reloaded.success_rate == result.success_rate
    assert len(reloaded.reward_history) == result.total_episodes

    print(f"Reloaded agent  : {reloaded.agent_id}")
    print(f"Episodes        : {reloaded.total_episodes}")
    print(f"Success rate    : {reloaded.success_rate:.1%}")
    print(f"History length  : {len(reloaded.reward_history)}")
    print("Serialization round-trip OK.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos."""
    print("\naumai-openworldrl Quickstart Examples")
    print("Python tabular Q-learning in open-ended grid environments\n")

    # Demo 1: basic training — returns result for use in demo 5
    result = demo_basic_training()

    # Demo 2: greedy evaluation with rendering
    demo_evaluation()

    # Demo 3: eval_mode context manager
    demo_eval_mode()

    # Demo 4: experience replay buffer
    demo_experience_replay()

    # Demo 5: JSON serialization using result from demo 1
    demo_serialization(result)

    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
