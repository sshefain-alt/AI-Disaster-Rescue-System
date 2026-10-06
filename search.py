"""
Search Algorithms Module
Contains all pathfinding algorithms:
- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- A* Search
- Risk-Aware A* Search
- Greedy Best-First Search
- Hill Climbing
"""

import heapq
from collections import deque


def bfs(environment, start, goal):
    """
    Breadth-First Search - Explores all neighbors level by level.
    Guarantees shortest path in unweighted grid.
    """
    if start == goal:
        return [start], 0

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (row, col), path = queue.popleft()

        for neighbor in environment.get_neighbors(row, col):
            if neighbor == goal:
                return path + [neighbor], len(path)

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, float('inf')


def dfs(environment, start, goal):
    """
    Depth-First Search - Explores as far as possible along each branch.
    Does NOT guarantee shortest path.
    """
    if start == goal:
        return [start], 0

    stack = [(start, [start])]
    visited = {start}

    while stack:
        (row, col), path = stack.pop()

        for neighbor in environment.get_neighbors(row, col):
            if neighbor == goal:
                return path + [neighbor], len(path)

            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

    return None, float('inf')


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(environment, start, goal):
    """
    A* Search - Uses f(n) = g(n) + h(n) where:
    - g(n) = cost from start to current node
    - h(n) = Manhattan distance heuristic to goal
    Guarantees optimal path.
    """
    if start == goal:
        return [start], 0

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, g_score[current]

        for neighbor in environment.get_neighbors(*current):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float('inf')


def risk_aware_astar(environment, start, goal, risk_weight=2.0):
    """
    Risk-Aware A* Search - Modified A* that considers risk zones.
    f(n) = g(n) + h(n) + risk_weight * risk(n)
    """
    if start == goal:
        return [start], 0

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, g_score[current]

        for neighbor in environment.get_neighbors(*current):
            risk = environment.get_risk_at(*neighbor)
            tentative_g = g_score[current] + 1 + risk_weight * risk

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float('inf')


def greedy_best_first(environment, start, goal):
    """
    Greedy Best-First Search - Uses only heuristic h(n) to guide search.
    Fast but does NOT guarantee optimal path.
    """
    if start == goal:
        return [start], 0

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))

    came_from = {}
    visited = {start}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, len(path) - 1

        for neighbor in environment.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    return None, float('inf')


def hill_climbing(environment, start, goal, max_iterations=1000):
    """
    Hill Climbing - Moves to the neighbor with lowest heuristic value.
    Can get stuck in local minima. Not guaranteed to find path.
    """
    if start == goal:
        return [start], 0

    current = start
    path = [current]
    visited = {current}

    for _ in range(max_iterations):
        if current == goal:
            return path, len(path) - 1

        neighbors = environment.get_neighbors(*current)
        # Filter out visited neighbors to avoid cycles
        unvisited = [n for n in neighbors if n not in visited]

        if not unvisited:
            # Stuck - no unvisited neighbors
            return None, float('inf')

        # Choose neighbor with lowest heuristic (greedy)
        next_node = min(unvisited, key=lambda n: heuristic(n, goal))

        # If we're moving away from goal (local minimum), stop
        if heuristic(next_node, goal) > heuristic(current, goal) and len(visited) > 1:
            # Allow some exploration but stop if clearly stuck
            pass

        current = next_node
        path.append(current)
        visited.add(current)

    return None, float('inf')


def _reconstruct_path(came_from, current):
    """Reconstruct path from came_from dictionary."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def compare_algorithms(environment, start, goal):
    """Run all algorithms and return comparison results."""
    results = {}

    print(f"\nComparing algorithms from {start} to {goal}...")
    print("-" * 50)

    # BFS
    path, cost = bfs(environment, start, goal)
    results['BFS'] = {'path': path, 'cost': cost}
    print(f"BFS:              Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    # DFS
    path, cost = dfs(environment, start, goal)
    results['DFS'] = {'path': path, 'cost': cost}
    print(f"DFS:              Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    # A*
    path, cost = astar(environment, start, goal)
    results['A*'] = {'path': path, 'cost': cost}
    print(f"A*:               Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    # Risk-Aware A*
    path, cost = risk_aware_astar(environment, start, goal)
    results['Risk-Aware A*'] = {'path': path, 'cost': cost}
    print(f"Risk-Aware A*:    Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    # Greedy Best-First
    path, cost = greedy_best_first(environment, start, goal)
    results['Greedy Best-First'] = {'path': path, 'cost': cost}
    print(f"Greedy Best-First: Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    # Hill Climbing
    path, cost = hill_climbing(environment, start, goal)
    results['Hill Climbing'] = {'path': path, 'cost': cost}
    print(f"Hill Climbing:    Cost={cost}, Path Length={len(path) if path else 'N/A'}")

    return results


if __name__ == "__main__":
    from environment import Environment

    env = Environment(seed=42)
    env.display()

    start = env.hospitals[0]
    goal = env.victims[0]['position']

    compare_algorithms(env, start, goal)
