# 基于 DreamerV3 世界模型的多智能体方格交通任务

English: *DreamerV3-Based Multi-Agent Grid Traffic Task*

这个仓库正在从原始的多智能体方格运输作业，迁移为一个以个人研究兴趣为主导的世界模型项目。新的主线任务是二维多智能体方格交通，核心方法是 `DreamerV3` 风格的世界模型，同时保留基线强化学习算法用于对比。我想借这个项目系统探索世界模型在局部观测、多智能体交互和长期决策中的作用。

当前对比基线包括：

- `Tabular Q-Learning`
- `DQN`
- `MCTS` 局部规划器
- `PPO`

## 项目定位

这不是一个追求复杂工程堆砌的仓库，而是一个强调“任务建模 + 世界模型方法 + 基线对比 + 可扩展实验”的个人研究项目。

核心卖点在于：

- 从固定 `5x5` 扩展到任意 `N x N` 网格
- 从固定 `4` 个智能体扩展到可配置数量
- 将任务从“固定搬运”升级为“多智能体方格交通”
- 以 `DreamerV3` 作为主方法，同时保留基线强化学习算法进行对比
- 从单 notebook 形式升级为 `src/ + tests/ + CLI` 的结构化项目

## 当前结构

```text
.
├── README.md
├── pyproject.toml
├── notebooks
│   └── grid_traffic_baselines_exploration.ipynb
├── src
│   └── grid_traffic
│       ├── algorithms
│       │   ├── baselines
│       │   └── dreamer
│       ├── envs
│       ├── pipeline
│       ├── __init__.py
│       └── cli.py
└── tests
    ├── test_agents.py
    └── test_env.py
```

## 当前阶段

目前仓库已经具备：

- 可配置网格大小与智能体数量的环境骨架
- `Tabular Q`、`DQN`、`PPO`、`MCTS` 四个基线实现
- `uv` 管理、CLI 入口和基础测试

当前正在推进：

- 任务语义从运输切换到交通流
- `DreamerV3` 风格世界模型主线
- 持续完善 `PPO` 与 `DreamerV3` 的对比实验

## 方法说明

### DreamerV3

项目主方法采用 `DreamerV3` 风格世界模型，用局部观测学习潜在动态，并在 imagined rollout 中优化长期决策。

English: *The main method will be a DreamerV3-style world model trained for long-horizon decision making under local observations.*

### Tabular Q-Learning

使用离散状态编码学习局部决策策略，适合作为小规模交通场景的强基线。

English: *A compact baseline with explicit state-action values.*

### DQN

使用神经网络逼近 Q 值函数，对更大状态空间和后续扩展更友好。

English: *A function-approximation baseline for larger or more varied settings.*

### MCTS

这里的 `MCTS` 不是全局联合规划，而是“当前行动 agent 的局部搜索器”。其他 agent 被视为环境的一部分，并在 rollout 中由简单启发式策略推进。

这符合你当前项目里 `DQN` 和 `Tabular Q` 的建模视角，也更容易实现和讲清楚。

English: *A local planner under a single-agent decision view, not a globally optimal joint controller.*

### PPO

`PPO` 作为现代 model-free baseline 引入，用于和 `DreamerV3` 做更公平的策略学习对比。

English: *PPO will serve as the main model-free policy baseline alongside DreamerV3.*

## 任务设计

新任务不再是固定 `A -> B` 搬运，而是二维多智能体方格交通：

- 网格大小可配置，如 `5x5`、`7x7`、`10x10`
- 智能体数量可配置
- 智能体在路口、走廊和瓶颈区域中持续通行
- 环境强调局部观测、冲突规避、拥堵传播和长期通行效率
- 当前仍保留原有环境骨架，后续会逐步迁移为交通流语义

当前环境以“单个正在行动的 agent 的局部观察”为主，其他 agent 通过邻接占用与全局位置关系隐式体现在环境里。

## 为什么不是只保留 Notebook

单 notebook 对快速实验是够的，但不利于持续迭代世界模型研究。现在的仓库把 notebook 保留为实验记录，同时新增了：

- 可复用的环境模块
- 三类可对比 agent
- 可配置训练与评测入口
- 基础测试

这样更适合长期维护和做实验，因为任务、方法和训练入口都可以独立演化，而不是把所有逻辑都塞在一个 notebook 里。

## 使用 `uv`

1. 安装依赖：

```bash
uv sync
```

2. 运行测试：

```bash
uv run pytest
```

3. 训练或评测 `Tabular Q`：

```bash
uv run grid-traffic --agent tabular_q --grid-size 5 --num-agents 4 --episodes 200
```

4. 训练或评测 `DQN`：

```bash
uv run grid-traffic --agent dqn --grid-size 7 --num-agents 6 --episodes 300
```

5. 训练或评测 `PPO`：

```bash
uv run grid-traffic --agent ppo --grid-size 7 --num-agents 6 --episodes 300
```

6. 直接评测 `MCTS` 局部规划器：

```bash
uv run grid-traffic --agent mcts --grid-size 6 --num-agents 4 --eval-scenarios 30 --mcts-simulations 80 --mcts-depth 14
```

## 后续还可以继续补强的方向

- 实现 `DreamerV3-lite` 版本的世界模型训练流程
- 把运输语义彻底替换为交通流与路口通行语义
- 进一步调优 `PPO` 作为主要 model-free baseline
- 把训练结果自动保存为 `csv` 或 `json`
- 增加更适合交通任务的可视化与拥堵分析脚本
