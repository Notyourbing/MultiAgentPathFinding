import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import time

# 网络拓扑和参数设置
nodes = 6  # 网络中节点数
graph = {
    0: [1, 2],  # 节点0连接节点1,2
    1: [0, 3, 4],
    2: [0, 4],
    3: [1, 5],
    4: [1, 2, 5],
    5: [3, 4]
}

# 模拟的网络状态（带宽、延迟）
base_bandwidth = {  # 节点之间的带宽（单位：Mbps）
    (0, 1): 100, (0, 2): 50,
    (1, 3): 80, (1, 4): 60,
    (2, 4): 90, (3, 5): 100,
    (4, 5): 70
}
base_delay = {  # 节点之间的延迟（单位：ms）
    (0, 1): 20, (0, 2): 30,
    (1, 3): 15, (1, 4): 25,
    (2, 4): 40, (3, 5): 10,
    (4, 5): 35
}

# 参数设置
alpha = 1.0  # 费洛蒙的重要性
beta = 2.0   # 启发式信息的重要性
rho = 0.1    # 费洛蒙蒸发率
Q = 100       # 费洛蒙增量
iterations = 50  # 迭代次数
ants = 10      # 蚂蚁数量

# 初始化
pheromone = np.ones((nodes, nodes))  # 费洛蒙矩阵
heuristic = np.array([[1 / base_delay.get((i, j), float('inf')) if i != j else 0 for j in range(nodes)] for i in range(nodes)])  # 延迟的启发式信息矩阵

# 动态变化的带宽和延迟
def update_network_state():
    """模拟带宽和延迟的动态变化，随机改变带宽和延迟"""
    global base_bandwidth, base_delay
    for key in base_bandwidth:
        base_bandwidth[key] += random.randint(-10, 10)  # 带宽在一定范围内波动
        base_bandwidth[key] = max(base_bandwidth[key], 10)  # 保证带宽大于10
    for key in base_delay:
        base_delay[key] += random.randint(-5, 5)  # 延迟在一定范围内波动
        base_delay[key] = max(base_delay[key], 5)  # 保证延迟大于5ms

    # 更新启发式信息（基于延迟）
    global heuristic
    heuristic = np.array([[1 / base_delay.get((i, j), float('inf')) if i != j else 0 for j in range(nodes)] for i in range(nodes)])

# 蚂蚁路径选择
def select_next_node(node, visited, pheromone, heuristic):
    unvisited_nodes = [n for n in range(nodes) if n not in visited and (node, n) in base_bandwidth]
    probabilities = []
    for next_node in unvisited_nodes:
        pheromone_level = pheromone[node][next_node] ** alpha
        heuristic_value = heuristic[node][next_node] ** beta
        prob = pheromone_level * heuristic_value
        probabilities.append(prob)

    # 归一化
    total_prob = sum(probabilities)
    probabilities = [p / total_prob for p in probabilities]

    # 根据概率选择下一节点
    return random.choices(unvisited_nodes, probabilities)[0]

# 蚂蚁寻找路径
def ant_search():
    start, end = 0, 5  # 假设从节点0到节点5
    path = [start]
    visited = {start}
    while path[-1] != end:
        next_node = select_next_node(path[-1], visited, pheromone, heuristic)
        path.append(next_node)
        visited.add(next_node)
    return path

# 费洛蒙更新
def update_pheromone(paths, pheromone):
    pheromone *= (1 - rho)  # 费洛蒙蒸发
    for path in paths:
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += Q / len(path)  # 更新路径上的费洛蒙

# 可视化结果
def visualize(graph, pheromone, best_path, iteration):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

    # 绘制最优路径
    for i in range(len(best_path) - 1):
        plt.plot([pos[best_path[i]][0], pos[best_path[i + 1]][0]],
                 [pos[best_path[i]][1], pos[best_path[i + 1]][1]],
                 color='r', lw=2, alpha=0.7)

    plt.title(f"Best Path found by Ant Colony Optimization (Iteration {iteration})")
    plt.show()

# 主程序
def aco_routing():
    best_path = None
    best_length = float('inf')

    for iteration in range(iterations):
        paths = []
        for _ in range(ants):
            path = ant_search()
            paths.append(path)

        update_pheromone(paths, pheromone)
        update_network_state()  # 每一轮更新网络状态（带宽、延迟）

        # 找到当前最短路径
        for path in paths:
            path_length = sum(base_delay.get((path[i], path[i + 1]), float('inf')) for i in range(len(path) - 1))
            if path_length < best_length:
                best_length = path_length
                best_path = path

        # 可视化最优路径
        visualize(graph, pheromone, best_path, iteration)
        time.sleep(1)  # 用于模拟动态变化的效果，暂停一秒

    return best_path, best_length

# 运行ACO算法
best_path, best_length = aco_routing()

# 输出最佳路径和路径长度
print("Best Path:", best_path)
print("Best Path Length:", best_length)
