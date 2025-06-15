from Astar import AstarBaseline
from DNQ import TrafficRoutingEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

GRID_ROWS = 8           # 网格行数
GRID_COLS = 8           # 网格列数
NUM_AGENTS = 4        # 智能体个数

def plot_astar_trajectories(trajectories, goals, grid_size):
    """
    在一个网格图上绘制所有智能体的 A* 轨迹，并标出起点和终点。

    参数：
      trajectories: {agent_id: [(r0, c0), (r1, c1), ...]} 字典
      goals: {agent_id: (goal_r, goal_c)} 字典
      grid_size: (H, W)
    """
    H, W = grid_size
    plt.figure(figsize=(H, W))

    # 先画出灰色网格线
    for x in range(H + 1):
        plt.plot([0, W], [x, x], color='gray', linewidth=0.5)
    for y in range(W + 1):
        plt.plot([y, y], [0, H], color='gray', linewidth=0.5)

    # 颜色列表，最多支持 6 个智能体
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # 绘制每个智能体的轨迹
    for i, traj in trajectories.items():
        # 将 (row, col) 转换为画图坐标 (x=col+0.5, y=(H-1-row)+0.5)
        xs = [pos[1] + 0.5 for pos in traj]
        ys = [(H - 1 - pos[0]) + 0.5 for pos in traj]
        plt.plot(xs, ys, marker='o', color=colors[i], label=f"Agent {i}")

        # 标记起点（方块）
        start = traj[0]
        sx, sy = start[1] + 0.5, (H - 1 - start[0]) + 0.5
        plt.scatter([sx], [sy], color=colors[i], marker='s', s=80)

        # 标记终点（星号）
        goal = goals[i]
        gx, gy = goal[1] + 0.5, (H - 1 - goal[0]) + 0.5
        plt.scatter([gx], [gy], color=colors[i], marker='*', s=120)

    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.gca().set_aspect('equal')
    plt.xticks([]);
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.title("A* Baseline Multi-Agent Trajectories")
    plt.tight_layout()
    plt.show()

def animate(trajectories, goals, grid_size):
    """
    在一个网格图上动画展示所有智能体的 A* 轨迹，逐步显示每一步移动。

    参数：
      trajectories: {agent_id: [(r0, c0), (r1, c1), ...]} 字典
      goals: {agent_id: (goal_r, goal_c)} 字典
      grid_size: (H, W)
    """
    H, W = grid_size
    fig, ax = plt.subplots(figsize=(H, W))

    # 画灰色网格线
    for x in range(H + 1):
        ax.plot([0, W], [x, x], color='gray', linewidth=0.5)
    for y in range(W + 1):
        ax.plot([y, y], [0, H], color='gray', linewidth=0.5)

    # 颜色列表，最多支持 6 个智能体
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # 绘制起点和终点（静态元素）
    for i, traj in trajectories.items():
        # 标记起点（方块）
        start = traj[0]
        sx, sy = start[1] + 0.5, (H - 1 - start[0]) + 0.5
        ax.scatter([sx], [sy], color=colors[i], marker='s', s=80)

        # 标记终点（星号）
        goal = goals[i]
        gx, gy = goal[1] + 0.5, (H - 1 - goal[0]) + 0.5
        ax.scatter([gx], [gy], color=colors[i], marker='*', s=120)

    # 初始化轨迹线
    lines = []
    for i in range(len(trajectories)):
        line, = ax.plot([], [], marker='o', color=colors[i], label=f"Agent {i}")
        lines.append(line)

    # 计算最大步数
    max_steps = max(len(traj) for traj in trajectories.values())

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            # 获取到当前帧为止的轨迹
            current_traj = trajectories[i][:frame+1]
            if len(current_traj) > 0:
                xs = [pos[1] + 0.5 for pos in current_traj]
                ys = [(H - 1 - pos[0]) + 0.5 for pos in current_traj]
                line.set_data(xs, ys)
        return lines

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.set_title("A* Baseline Multi-Agent Trajectories Animation")

    # 创建动画，每步间隔500毫秒
    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init,
                        interval=500, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    # 保存为GIF
    ani.save('animate_results/greedy_trajectories_BaselineAStar.gif', writer='pillow', fps=2, dpi=100)
    return ani  # 返回动画对象以便保存

if __name__ == "__main__":


    # 1. 初始化环境和基线算法
    env = TrafficRoutingEnv(grid_size=(GRID_ROWS, GRID_COLS), num_agents=NUM_AGENTS)
    baseline = AstarBaseline(env)

    # 2. 运行 A* 基线，获取轨迹和目标
    trajectories, goals = baseline.run_baseline()

    # 3. 打印每个智能体的轨迹
    print("A* Baseline Trajectories:")
    for i, traj in trajectories.items():
        print(f"Agent {i}: {traj}")

    # 4. 调用可视化函数，绘制所有智能体的路径
    plot_astar_trajectories(trajectories, goals, env.grid_size)
    #animate(trajectories, goals, env.grid_size)
