# Getting Started with aumai-openworldrl

> **Experimental software.** This guide assumes Python 3.11 or later.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| Python | 3.11 | Required for `from __future__ import annotations` and match syntax |
| pip | 22.0 | For editable installs with pyproject.toml |
| pydantic | 2.0 | Installed automatically as a dependency |
| click | 8.0 | Installed automatically as a dependency |

No GPU, PyTorch, or any ML framework is required. The library runs entirely on the CPU using
the Python standard library plus pydantic and click.

---

## Installation

### From PyPI

```bash
pip install aumai-openworldrl
```

### From source (development install)

```bash
git clone https://github.com/aumai/aumai-openworldrl
cd aumai-openworldrl
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, hypothesis, mypy, and ruff.

### Verify the installation

```bash
aumai-openworldrl --version
python -c "import aumai_openworldrl; print(aumai_openworldrl.__version__)"
```

---

## Step-by-Step Tutorial

This tutorial walks through the full workflow: configuring an environment, training an agent,
evaluating it, and inspecting the results.

### Step 1 — Configure the environment

The `Environment` model describes the maze the agent will explore. All fields are validated
at construction time by Pydantic.

```python
from aumai_openworldrl.models import Environment

env_config = Environment(
    env_id="tutorial_maze",
    width=6,
    height=6,
    max_steps=200,
    goal_reward=1.0,
    step_penalty=-0.01,
    wall_penalty=-0.05,
)
print(f"Grid: {env_config.width}x{env_config.height}, max steps: {env_config.max_steps}")
```

The agent starts at `(0, 0)` (top-left) and must reach `(height-1, width-1)` (bottom-right).
Walls are randomly placed at construction time based on the seed you provide to `GridWorldEnv`.

### Step 2 — Configure the agent

```python
from aumai_openworldrl.models import RLAgent

agent_config = RLAgent(
    agent_id="tutorial_agent",
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,          # start fully exploratory
    epsilon_min=0.01,     # never drop below 1% exploration
    epsilon_decay=0.995,  # decay ~0.5% per episode
    num_actions=4,        # up, right, down, left
)
```

The relationship between `epsilon_decay` and convergence speed is important. With `decay=0.995`
and starting from 1.0, epsilon reaches 0.1 after approximately 460 episodes and 0.05 after 590
episodes. For larger grids with more exploration required, use a slower decay such as 0.998 or
0.999.

### Step 3 — Create a Trainer and train

```python
from aumai_openworldrl.core import Trainer

trainer = Trainer(
    env_config=env_config,
    agent_config=agent_config,
    seed=42,              # for reproducibility
    replay_capacity=2000, # how many transitions to store
)

result = trainer.train(episodes=1500)

print(f"Training complete after {result.total_steps} steps.")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Mean reward (last 100 ep): {result.mean_reward:.4f}")
print(f"Best reward seen: {result.best_reward:.4f}")
print(f"Final epsilon: {result.final_epsilon:.4f}")
```

The `TrainingResult` contains the full per-episode reward history in `result.reward_history`,
which you can plot with matplotlib to observe the learning curve.

### Step 4 — Evaluate the trained agent

Evaluation uses `eval_mode()` internally to fix epsilon at 0.0, so the agent acts purely
greedily based on its learned Q-table.

```python
eval_result = trainer.evaluate(
    episodes=200,
    render_first=True,   # prints ASCII grid for first episode
)

print(f"Evaluation success rate: {eval_result.success_rate:.1%}")
print(f"Evaluation mean reward: {eval_result.mean_reward:.4f}")
```

### Step 5 — Save and reload results

`TrainingResult` is a Pydantic model so it serializes directly to JSON:

```python
import json
from aumai_openworldrl.models import TrainingResult

# Save
json_str = result.model_dump_json(indent=2)
with open("training_run.json", "w", encoding="utf-8") as fh:
    fh.write(json_str)

# Reload
with open("training_run.json", encoding="utf-8") as fh:
    loaded = TrainingResult.model_validate_json(fh.read())

print(f"Reloaded: {len(loaded.reward_history)} episodes, "
      f"success_rate={loaded.success_rate:.1%}")
```

---

## Common Patterns and Recipes

### Recipe 1 — Sweep learning rates

```python
from aumai_openworldrl.core import Trainer
from aumai_openworldrl.models import Environment, RLAgent

env_config = Environment(env_id="sweep_env", width=5, height=5)

for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
    agent_config = RLAgent(agent_id=f"agent_lr{lr}", learning_rate=lr)
    trainer = Trainer(env_config, agent_config, seed=0)
    result = trainer.train(episodes=1000)
    print(f"lr={lr:.2f}  success_rate={result.success_rate:.1%}  "
          f"mean_reward={result.mean_reward:.4f}")
```

### Recipe 2 — Compare wall densities

The `GridWorldEnv` accepts a `wall_density` parameter controlling the fraction of cells that
become walls. Higher density makes navigation harder.

```python
from aumai_openworldrl.core import GridWorldEnv, QLearningAgent, Trainer
from aumai_openworldrl.models import Environment, RLAgent

env_base = Environment(env_id="wd_test", width=8, height=8, max_steps=300)
agent_config = RLAgent(agent_id="wd_agent", epsilon_decay=0.997)

for density in [0.05, 0.10, 0.20, 0.30]:
    # Manually construct with custom wall density
    env = GridWorldEnv(env_base, wall_density=density, seed=42)
    agent = QLearningAgent(agent_config, seed=42)
    from aumai_openworldrl.core import ExperienceReplay
    replay = ExperienceReplay(capacity=2000)
    # ... run episodes manually or adjust Trainer to accept pre-built env
    print(f"wall_density={density:.0%}: grid rendered below")
    print(env.render())
    print()
```

### Recipe 3 — Plot the learning curve

```python
import json
from aumai_openworldrl.core import Trainer
from aumai_openworldrl.models import Environment, RLAgent

result = Trainer(
    Environment(env_id="plot_env", width=6, height=6),
    RLAgent(agent_id="plot_agent"),
    seed=42,
).train(episodes=2000)

# Smooth with a rolling window
window = 50
smoothed = [
    sum(result.reward_history[max(0, i - window):i + 1]) /
    min(i + 1, window)
    for i in range(len(result.reward_history))
]

# Print a simple ASCII plot
print("Reward over episodes (smoothed):")
for i, v in enumerate(smoothed[::100]):
    bar = "#" * int((v + 0.1) * 30)
    print(f"  ep {i * 100:4d}: {bar} {v:.3f}")
```

### Recipe 4 — Use the replay buffer for offline analysis

```python
import random
from aumai_openworldrl.core import ExperienceReplay, GridWorldEnv, QLearningAgent
from aumai_openworldrl.models import Environment, Experience, RLAgent

buffer = ExperienceReplay(capacity=10000)
env    = GridWorldEnv(Environment(env_id="buf_env", width=5, height=5), seed=1)
agent  = QLearningAgent(RLAgent(agent_id="buf_agent"), seed=1)

# Collect random experiences (epsilon=1 effectively)
state = env.reset()
for _ in range(500):
    action = random.randrange(4)
    next_state, reward, done, _ = env.step(action)
    buffer.push(Experience(state=state, action=action, reward=reward,
                            next_state=next_state, done=done))
    state = env.reset() if done else next_state

print(f"Collected {len(buffer)} transitions")
batch = buffer.sample(32)
print(f"Sample batch: {len(batch)} transitions, "
      f"avg reward={sum(e.reward for e in batch)/len(batch):.4f}")
```

### Recipe 5 — Deterministic replay for debugging

Pass a seeded `random.Random` instance to `buffer.sample()` for reproducible sampling:

```python
import random
rng = random.Random(99)
batch = buffer.sample(16, rng=rng)
# Same seed → same batch every time, useful for unit tests
```

---

## Troubleshooting FAQ

**Q: The success rate stays at 0% even after 2000 episodes. What is wrong?**

A: The most common causes are:
1. `max_steps` is too low relative to the grid size. For an NxN grid, a rough lower bound is
   `max_steps >= 2*N*N`. Try `max_steps=300` for a 10x10 grid.
2. `epsilon_decay` is too aggressive. If epsilon reaches `epsilon_min` too quickly, the agent
   stops exploring before it has found the goal even once. Increase decay to 0.998 or 0.999.
3. The random seed produces an environment with the goal isolated by walls. Try different seeds.

---

**Q: My training results are not reproducible even with `seed=42`.**

A: Make sure you pass `seed=42` to `Trainer(...)` and that no other code calls `random.seed()`
or `random.random()` between construction and training. The `GridWorldEnv` and `QLearningAgent`
each maintain their own `random.Random` instances seeded from the provided value, so they are
isolated from the global RNG — but only if the global state is not externally modified.

---

**Q: How do I use a larger action space than 4?**

A: The `GridWorldEnv` is hardcoded to 4 actions (up/right/down/left). For a larger action space
you would need to subclass `GridWorldEnv` and override `step()`. Set `num_actions` on `RLAgent`
to match the new action space size.

---

**Q: The `q_table_size` grows very large. Is this a memory problem?**

A: For a 5x5 grid with 4 actions, the maximum Q-table size is 25 * 4 = 100 entries. For a 20x20
grid it is 1600 entries. Each entry is a Python float (about 28 bytes). At 100K entries the table
is roughly 2.8 MB, which is negligible. If you are training on very large grids (50x50+), consider
whether a function approximator (neural network) would be more appropriate.

---

**Q: Can I save and restore the Q-table between runs?**

A: The Q-table is stored in `agent._q_table` as a `dict[tuple, float]`. You can serialize it with
`json.dumps({str(k): v for k, v in agent._q_table.items()})`. Restoring requires
reconstructing the tuple keys from strings. A proper checkpoint API is on the roadmap.

---

**Q: The CLI train command writes JSON but the file is very large.**

A: The `reward_history` list stores one float per episode. For 100,000 episodes this is ~800 KB
as JSON. If storage is a concern, you can post-process the JSON to remove `reward_history` before
archiving.

---

**Q: Pydantic raises a `ValidationError` when I construct `RLAgent`.**

A: Common causes:
- `learning_rate` must be in `(0, 1]`. A value of 0 or negative raises a validation error.
- `discount_factor` must be in `[0, 1]`.
- `epsilon_decay` must be in `(0, 1]`. A value of 1.0 means no decay (epsilon stays constant).
- `num_actions` must be a positive integer.

Run `python -c "from aumai_openworldrl.models import RLAgent; print(RLAgent.model_fields)"` to
inspect all field constraints.
