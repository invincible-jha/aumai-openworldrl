"""CLI entry point for aumai-openworldrl.

Commands:
    train     -- train a Q-learning agent on the grid world
    evaluate  -- evaluate a trained agent (greedy policy)
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .core import Trainer
from .models import Environment, RLAgent


@click.group()
@click.version_option()
def main() -> None:
    """AumAI OpenWorldRL -- RL in open-ended environments CLI."""


@main.command("train")
@click.option("--episodes", default=1000, show_default=True, type=int)
@click.option("--width", default=5, show_default=True, type=int, help="Grid width.")
@click.option("--height", default=5, show_default=True, type=int, help="Grid height.")
@click.option("--max-steps", default=200, show_default=True, type=int)
@click.option("--lr", default=0.1, show_default=True, type=float, help="Q-learning rate.")
@click.option("--gamma", default=0.99, show_default=True, type=float, help="Discount factor.")
@click.option("--epsilon-decay", default=0.995, show_default=True, type=float)
@click.option("--seed", default=None, type=int)
@click.option(
    "--output",
    "output_path",
    default="training_result.json",
    show_default=True,
    type=click.Path(path_type=Path),
)
def train_command(
    episodes: int,
    width: int,
    height: int,
    max_steps: int,
    lr: float,
    gamma: float,
    epsilon_decay: float,
    seed: int | None,
    output_path: Path,
) -> None:
    """Train a tabular Q-learning agent on a grid world environment.

    Example:

        aumai-openworldrl train --episodes 1000 --width 6 --height 6
    """
    env_config = Environment(
        env_id="gw_train",
        width=width,
        height=height,
        max_steps=max_steps,
    )
    agent_config = RLAgent(
        agent_id="q_agent",
        learning_rate=lr,
        discount_factor=gamma,
        epsilon_decay=epsilon_decay,
        num_actions=4,
    )

    click.echo(
        f"Training Q-learning agent: {episodes} episodes on "
        f"{width}x{height} grid (seed={seed})"
    )

    trainer = Trainer(env_config, agent_config, seed=seed)
    result = trainer.train(episodes=episodes)

    click.echo(f"\nTraining complete:")
    click.echo(f"  Total episodes  : {result.total_episodes}")
    click.echo(f"  Total steps     : {result.total_steps}")
    click.echo(f"  Mean reward     : {result.mean_reward:.4f}")
    click.echo(f"  Best reward     : {result.best_reward:.4f}")
    click.echo(f"  Success rate    : {result.success_rate:.1%}")
    click.echo(f"  Final epsilon   : {result.final_epsilon:.4f}")

    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"\nResult saved to {output_path}")


@main.command("evaluate")
@click.option("--episodes", default=100, show_default=True, type=int)
@click.option("--width", default=5, show_default=True, type=int)
@click.option("--height", default=5, show_default=True, type=int)
@click.option("--train-episodes", default=500, show_default=True, type=int, help="Pre-train for N episodes first.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--render", is_flag=True, default=False, help="Render first episode grid.")
def evaluate_command(
    episodes: int,
    width: int,
    height: int,
    train_episodes: int,
    seed: int,
    render: bool,
) -> None:
    """Pre-train then evaluate a Q-learning agent greedily.

    Example:

        aumai-openworldrl evaluate --train-episodes 500 --episodes 100 --render
    """
    env_config = Environment(env_id="gw_eval", width=width, height=height, max_steps=200)
    agent_config = RLAgent(agent_id="q_eval", num_actions=4)

    trainer = Trainer(env_config, agent_config, seed=seed)

    click.echo(f"Pre-training for {train_episodes} episodes...")
    trainer.train(episodes=train_episodes)

    click.echo(f"Evaluating for {episodes} episodes (greedy policy)...")
    result = trainer.evaluate(episodes=episodes, render_first=render)

    click.echo(f"\nEvaluation results:")
    click.echo(f"  Mean reward  : {result.mean_reward:.4f}")
    click.echo(f"  Best reward  : {result.best_reward:.4f}")
    click.echo(f"  Success rate : {result.success_rate:.1%}")


if __name__ == "__main__":
    main()
