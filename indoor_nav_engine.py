# indoor_nav_engine.py
# A* Pathfinding and Shopping List Route Optimization Engine

import heapq
import numpy as np

# Store Layout Grid: 0 = Walkable Aisle, 1 = Shelf/Obstacle
STORE_GRID = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # Aisle 1 (Dairy & Snacks)
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Main Center Crossway
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # Aisle 2 (Beverages & Staples)
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Checkout Counters
])

# Item Aisle Coordinates (x, y)
ITEM_LOCATIONS = {
    "Amul Taaza Milk 500ml": (0, 1),
    "Lays Salted Chips 50g": (3, 1),
    "Bisleri Mineral Water 1L": (0, 4),
    "Cadbury Dairy Milk 100g": (3, 4),
    "Maggi 2-Min Noodles 70g": (6, 4)
}

class StoreNavigationEngine:
    def __init__(self, grid=STORE_GRID):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar_path(self, start, goal):
        """Computes shortest walkable path on store grid using A*."""
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                return data[::-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    if self.grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return [start, goal]

    def optimize_shopping_route(self, current_pos, shopping_list):
        """Solves nearest-neighbor TSP to sequence item collection optimally."""
        unvisited = [item for item in shopping_list if item in ITEM_LOCATIONS]
        curr = current_pos
        route = [curr]

        while unvisited:
            next_item = min(
                unvisited, 
                key=lambda x: self.heuristic(curr, ITEM_LOCATIONS[x])
            )
            route.append(ITEM_LOCATIONS[next_item])
            curr = ITEM_LOCATIONS[next_item]
            unvisited.remove(next_item)

        return route