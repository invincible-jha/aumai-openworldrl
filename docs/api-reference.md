# API Reference — aumai-openworldrl

> **Experimental software.** All public APIs are subject to change without notice.

This document covers every public class and function exported from
`aumai_openworldrl.core` and `aumai_openworldrl.models`.

---

## Module: `aumai_openworldrl.models`

All models use Pydantic v2. Fields are validated at construction time and are effectively
immutable (mutation requires `model_copy(update=...)`).

---

### `class Environment`

```python
class Environment(BaseModel):
    env_id: str
    name: str = "GridWorld"
    width: int = Field(default=5, gt=0)
    height: int = Field(default=5, gt=0)
    num_actions: int = Field(default=4, gt=0)
    max_steps: int = Field(default=200, gt=0)
    goal_reward: float = Field(default=1.0)
    step_penalty: float = Field(default=-0.01)
    wall_penalty: float = Field(default=-0.05)
```

Configuration model for a reinforcement learning environment.

**Fields:**

| Field | Type | Default | Constraint | Description |
|-------|------|---------|------------|-------------|
| `env_id` | `str` | required | non-empty | Unique identifier for this environment instance |
| `name` | `str` | `"GridWorld"` | — | Human-readable display name |
| `width` | `int` | 5 | `> 0` | Grid columns; agent starts at column 0 |
| `height` | `int` | 5 | `> 0` | Grid rows; agent starts at row 0 |
| `num_actions` | `int` | 4 | `> 0` | Discrete action space size |
| `max_steps` | `int` | 200 | `> 0` | Episode terminates after this many steps even if goal not reached |
| `goal_reward` | `float` | 1.0 | — | Reward added to the step reward when `agent_pos == goal_pos` |
| `step_penalty` | `float` | -0.01 | — | Reward applied at each non-terminal step |
| `wall_penalty` | `float` | -0.05 | — | Reward applied when the requested move would hit a wall or boundary |

**Example:**

```python
env = Environment(
    env_id="env_001",
    width=8,
    height=8,
    max_steps=300,
    goal_reward=2.0,
    step_penalty=-0.02,
    wall_penalty=-0.1,
)
```

---

### `class Experience`

```python
class Experience(BaseModel):
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
```

A single `(s, a, r, s', done)` transition tuple stored in the replay buffer.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `state` | `Any` | State before action (typically `tuple[int, int]` for grid world) |
| `action` | `int` | Integer action index taken |
| `reward` | `float` | Reward received from the environment |
| `next_state` | `Any` | State resulting from the action |
| `done` | `bool` | Whether this transition ended the episode |

**Example:**

```python
exp = Experience(
    state=(2, 3),
    action=1,          # right
    reward=-0.01,
    next_state=(2, 4),
    done=False,
)
```

---

### `class RLAgent`

```python
class RLAgent(BaseModel):
    agent_id: str
    algorithm: str = "q_learning"
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    num_actions: int = Field(default=4, gt=0)
```

Configuration and identity record for a reinforcement learning agent.

**Fields:**

| Field | Type | Default | Constraint | Description |
|-------|------|---------|------------|-------------|
| `agent_id` | `str` | required | — | Unique identifier |
| `algorithm` | `str` | `"q_learning"` | — | Algorithm tag (informational) |
| `learning_rate` | `float` | 0.1 | `(0, 1]` | Alpha in the Bellman update |
| `discount_factor` | `float` | 0.99 | `[0, 1]` | Gamma — weight given to future rewards |
| `epsilon` | `float` | 1.0 | `[0, 1]` | Current exploration probability |
| `epsilon_min` | `float` | 0.01 | `[0, 1]` | Minimum epsilon after decay |
| `epsilon_decay` | `float` | 0.995 | `(0, 1]` | Multiplied by epsilon at end of each episode |
| `num_actions` | `int` | 4 | `> 0` | Size of the discrete action space |

---

### `class TrainingResult`

```python
class TrainingResult(BaseModel):
    agent_id: str
    total_episodes: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    mean_reward: float = Field(default=0.0)
    best_reward: float = Field(default=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    final_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_history: list[float] = Field(default_factory=list)
```

Immutable summary of a completed training or evaluation run.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | ID of the agent that was trained |
| `total_episodes` | `int` | Total number of episodes run |
| `total_steps` | `int` | Total environment steps taken across all episodes |
| `mean_reward` | `float` | Mean episode reward over the last 100 episodes |
| `best_reward` | `float` | Best single-episode cumulative reward observed |
| `success_rate` | `float` | Fraction of episodes where the goal was reached, in `[0, 1]` |
| `final_epsilon` | `float` | Epsilon value at end of training (0.0 for evaluation results) |
| `reward_history` | `list[float]` | Rounded (4 dp) cumulative reward per episode in order |

---

## Module: `aumai_openworldrl.core`

---

### `class GridWorldEnv`

```python
class GridWorldEnv:
    _ACTIONS: list[tuple[int, int]]  # (dr, dc) deltas for [up, right, down, left]
    _ACTION_NAMES: list[str]

    def __init__(
        self,
        config: Environment,
        wall_density: float = 0.1,
        seed: Optional[int] = None,
    ) -> None: ...
```

Deterministic 2-D grid world environment. The agent starts at `(0, 0)` and must reach
`(height-1, width-1)`. Walls are randomly placed at construction time using the provided seed.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Environment` | required | Environment configuration |
| `wall_density` | `float` | 0.1 | Fraction of non-start, non-goal cells that become walls |
| `seed` | `Optional[int]` | None | Seed for wall generation and internal RNG |

**Actions:**

| Index | Direction | Delta (row, col) |
|-------|-----------|-----------------|
| 0 | up | (-1, 0) |
| 1 | right | (0, +1) |
| 2 | down | (+1, 0) |
| 3 | left | (0, -1) |

---

#### `GridWorldEnv.reset()`

```python
def reset(self) -> tuple[int, int]:
```

Reset the environment to the start state `(0, 0)`. Resets the step counter but does not
regenerate walls.

**Returns:** `tuple[int, int]` — Initial agent position `(row, col)`.

---

#### `GridWorldEnv.step()`

```python
def step(
    self,
    action: int,
) -> tuple[tuple[int, int], float, bool, dict[str, object]]:
```

Take one environment step.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `int` | Action index; values outside `[0, 3]` are reduced via `action % 4` |

**Returns:** `(next_state, reward, done, info)` where:
- `next_state: tuple[int, int]` — new agent position `(row, col)`
- `reward: float` — `step_penalty` for valid move, `wall_penalty` for blocked move, plus
  `goal_reward` if goal reached
- `done: bool` — True if goal reached or `max_steps` exceeded
- `info: dict` — `{"steps": int, "reached_goal": bool}`

**Reward logic:**

```
if move is valid (in bounds, not a wall):
    reward = step_penalty
    agent_pos = new_pos
else:
    reward = wall_penalty
    agent_pos unchanged

if agent_pos == goal_pos:
    reward += goal_reward
    done = True
elif steps >= max_steps:
    done = True
```

---

#### `GridWorldEnv.render()`

```python
def render(self) -> str:
```

Render the grid as a multi-line ASCII string.

**Returns:** String where each cell is one of: `A` (agent), `G` (goal), `#` (wall), `.` (open).
Cells are space-separated; rows are newline-separated.

**Example output:**

```
A . . . .
. # . # .
. . . . .
. # . # .
. . . . G
```

---

### `class ExperienceReplay`

```python
class ExperienceReplay:
    def __init__(self, capacity: int = 1000) -> None: ...
```

Fixed-capacity circular experience replay buffer backed by `collections.deque`.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | 1000 | Maximum number of transitions to store; oldest are evicted first |

---

#### `ExperienceReplay.push()`

```python
def push(self, experience: Experience) -> None:
```

Add one experience to the buffer. If at capacity, the oldest experience is silently evicted.

**Parameters:** `experience: Experience` — transition to store.

---

#### `ExperienceReplay.sample()`

```python
def sample(
    self,
    batch_size: int,
    rng: Optional[random.Random] = None,
) -> list[Experience]:
```

Sample a random mini-batch from the buffer.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | required | Number of samples to draw |
| `rng` | `Optional[random.Random]` | None | Seeded RNG for reproducible sampling; uses global `random` if None |

**Returns:** `list[Experience]` — if buffer size `<= batch_size`, returns all stored experiences.

---

#### `ExperienceReplay.__len__()`

```python
def __len__(self) -> int:
```

Returns the current number of experiences in the buffer.

---

### `class QLearningAgent`

```python
class QLearningAgent:
    def __init__(
        self,
        config: RLAgent,
        seed: Optional[int] = None,
    ) -> None: ...
```

Tabular Q-learning agent with epsilon-greedy exploration. The Q-table is a
`dict[tuple[object, int], float]` mapping `(state, action)` pairs to Q-values.
Unvisited pairs implicitly have Q-value 0.0.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `RLAgent` | required | Agent configuration |
| `seed` | `Optional[int]` | None | Seed for the internal `random.Random` used in exploration |

---

#### `QLearningAgent.select_action()`

```python
def select_action(self, state: object) -> int:
```

Select an action using epsilon-greedy policy.

With probability `epsilon`: return a uniformly random action index in `[0, num_actions)`.
With probability `1 - epsilon`: return the greedy action (highest Q-value; ties broken randomly).

**Parameters:** `state: object` — current state; must be hashable.

**Returns:** `int` — action index.

---

#### `QLearningAgent.update()`

```python
def update(
    self,
    state: object,
    action: int,
    reward: float,
    next_state: object,
    done: bool,
) -> float:
```

Apply the Q-learning (Bellman) update rule.

```
Q(s,a) <- Q(s,a) + lr * (target - Q(s,a))
target  = r                          if done
target  = r + gamma * max_a' Q(s',a') otherwise
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `object` | State before action |
| `action` | `int` | Action taken |
| `reward` | `float` | Reward received |
| `next_state` | `object` | State after action |
| `done` | `bool` | Whether the episode ended |

**Returns:** `float` — the TD error `(target - Q(s,a))` _before_ the update was applied.

---

#### `QLearningAgent.decay_epsilon()`

```python
def decay_epsilon(self) -> None:
```

Decay the exploration rate by one step: `epsilon = max(epsilon_min, epsilon * epsilon_decay)`.
Creates a new `RLAgent` config instance via `model_copy(update=...)`.

---

#### `QLearningAgent.eval_mode()`

```python
@contextmanager
def eval_mode(self) -> Iterator[None]:
```

Context manager that sets epsilon to 0.0 for the duration of the `with` block, then restores
the saved value unconditionally via `try/finally`.

**Usage:**

```python
with agent.eval_mode():
    action = agent.select_action(state)  # always greedy
# epsilon is restored here, even if an exception was raised inside the block
```

---

#### `QLearningAgent.epsilon` (property)

```python
@property
def epsilon(self) -> float:
```

Current exploration rate. Read-only.

---

#### `QLearningAgent.q_table_size` (property)

```python
@property
def q_table_size(self) -> int:
```

Number of `(state, action)` entries that have been visited and stored in the Q-table.

---

### `class Trainer`

```python
class Trainer:
    def __init__(
        self,
        env_config: Environment,
        agent_config: RLAgent,
        seed: Optional[int] = None,
        replay_capacity: int = 1000,
    ) -> None: ...
```

Orchestrates Q-learning training and evaluation loops on a `GridWorldEnv`.

Internally constructs a `GridWorldEnv`, a `QLearningAgent`, and an `ExperienceReplay` buffer.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_config` | `Environment` | required | Environment configuration |
| `agent_config` | `RLAgent` | required | Agent configuration |
| `seed` | `Optional[int]` | None | Passed to both `GridWorldEnv` and `QLearningAgent` |
| `replay_capacity` | `int` | 1000 | Capacity of the experience replay buffer |

---

#### `Trainer.train()`

```python
def train(self, episodes: int = 1000) -> TrainingResult:
```

Train the agent for a fixed number of episodes. Each episode:
1. Resets the environment.
2. Runs steps until `done` is True.
3. Pushes each transition to the replay buffer.
4. Updates the Q-table after each step.
5. Calls `decay_epsilon()` at episode end.

**Parameters:** `episodes: int` — number of episodes to run.

**Returns:** `TrainingResult` — summary statistics including the full per-episode reward history.

---

#### `Trainer.evaluate()`

```python
def evaluate(
    self,
    episodes: int = 100,
    render_first: bool = False,
) -> TrainingResult:
```

Evaluate the trained agent greedily (epsilon fixed at 0.0 via `eval_mode()`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `episodes` | `int` | 100 | Number of evaluation episodes |
| `render_first` | `bool` | False | If True, print the ASCII grid after each step of episode 0 |

**Returns:** `TrainingResult` with `final_epsilon=0.0` since evaluation is always greedy.

---

## Top-Level Exports (`aumai_openworldrl.__init__`)

```python
__version__ = "0.1.0"
```

The package currently exports only `__version__`. All classes must be imported directly from
`aumai_openworldrl.core` or `aumai_openworldrl.models`.

---

## Exceptions

No custom exception types are defined. The following standard exceptions may be raised:

| Exception | Source | Cause |
|-----------|--------|-------|
| `pydantic.ValidationError` | Model construction | Invalid field values (e.g., `learning_rate=0`) |
| `IndexError` | `ExperienceReplay.sample` | Cannot occur; returns all items if buffer is smaller than `batch_size` |
| `ValueError` | `random.sample` | Cannot occur in normal usage; guarded by the buffer size check |

---

## Thread Safety

No component in this library is thread-safe. The `QLearningAgent` Q-table and `RLAgent` config
are modified by `update()` and `decay_epsilon()` respectively. The `ExperienceReplay` buffer is
modified by `push()`. Do not share instances across threads without external synchronization.
