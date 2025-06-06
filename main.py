from Astar import AstarBaseline
from DNQ import TrafficRoutingEnv
import matplotlib.pyplot as plt

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
