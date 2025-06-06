# Astar.py

import heapq
import numpy as np
from collections import defaultdict

class AstarBaseline:
    """
    A* 基线算法的类封装，用于多智能体在 TrafficRoutingEnv 环境中规划路径并模拟移动。
    """

    def __init__(self, env):
        """
        初始化 A* 基线算法实例。

        参数：
          env: 一个 TrafficRoutingEnv 实例
        """
        self.env = env
        self.grid_size = env.grid_size
        self.num_agents = env.num_agents

    @staticmethod
    def _heuristic(a, b):
        """
        曼哈顿距离启发式函数。

        参数：
          a, b: (row, col) 坐标元组
        返回：
          两点的曼哈顿距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar_search(self, start, goal):
        """
        在空网格上执行 A* 搜索，返回一条从 start 到 goal 的最短路径（行列坐标列表）。

        参数：
          start: 起点坐标 (row, col)
          goal: 终点坐标 (row, col)
        返回：
          如果存在路径，返回 [(r0,c0), (r1,c1), ..., (rn,cn)]；否则返回 None。
        """
        H, W = self.grid_size
        open_set = []
        # 每个元素格式: (f_score, g_score, 当前节点, 父节点)
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, None))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            f_curr, g_curr, current, parent = heapq.heappop(open_set)
            if current in came_from:
                continue
            came_from[current] = parent

            if current == goal:
                # 重新构建路径
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return path[::-1]

            x, y = current
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    neighbors.append((nx, ny))

            for neighbor in neighbors:
                tentative_g = g_curr + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))

        return None

    def run_baseline(self, max_steps=50):
        """
        运行 A* 基线算法：为每个智能体独立计算一条空网格 A* 路径，然后模拟按路径移动，
        遇到碰撞时停留不动。返回每个智能体的轨迹。

        参数：
          max_steps: 最多模拟步数，默认 50

        返回：
          (agent_trajectories, goals)
            - agent_trajectories: {agent_id: [(r0,c0), (r1,c1), ...]} 的字典
            - goals: {agent_id: (goal_row, goal_col)} 的字典
        """
        # 1. 重置环境并获取初始位置、目标
        obs = self.env.reset()
        start_positions = {i: obs[i]['position'] for i in range(self.num_agents)}
        goals = {i: obs[i]['destination'] for i in range(self.num_agents)}

        # 2. 为每个智能体计算空网格 A* 路径
        paths = {}
        for i in range(self.num_agents):
            path = self._astar_search(start_positions[i], goals[i])
            if path is None:
                # 如果没有可行路径，则保持起始位置
                paths[i] = [start_positions[i]]
            else:
                paths[i] = path

        # 3. 按路径模拟移动，并收集每个智能体的轨迹
        agent_positions = dict(start_positions)
        agent_trajectories = {i: [agent_positions[i]] for i in range(self.num_agents)}

        for step in range(1, max_steps + 1):
            desired = {}
            # 确定每个智能体期望的位置
            for i in range(self.num_agents):
                current = agent_positions[i]
                path = paths[i]
                if current == goals[i]:
                    desired[i] = current
                else:
                    idx = path.index(current)
                    if idx + 1 < len(path):
                        desired[i] = path[idx + 1]
                    else:
                        desired[i] = current

            # 处理碰撞：如果多辆车想去同一格，则都停在原位
            new_positions = {}
            occupied = {}
            for i, pos in desired.items():
                if pos in occupied:
                    # 冲突：两个智能体都退回原位置
                    new_positions[i] = agent_positions[i]
                    other = occupied[pos]
                    new_positions[other] = agent_positions[other]
                else:
                    occupied[pos] = i
                    new_positions[i] = pos

            # 更新所有智能体位置，记录轨迹
            agent_positions = new_positions
            for i in range(self.num_agents):
                agent_trajectories[i].append(agent_positions[i])

            # 如果所有智能体都到达各自目标，提前结束
            if all(agent_positions[i] == goals[i] for i in range(self.num_agents)):
                break

        return agent_trajectories, goals
