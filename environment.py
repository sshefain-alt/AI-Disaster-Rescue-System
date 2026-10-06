"""
Environment Module
Creates the disaster environment including:
- Grid
- Victims
- Hospitals
- Risk zones
- Blocked roads
"""

import random
import numpy as np


class Environment:
    """Disaster environment grid with victims, hospitals, risk zones, and blocked roads."""

    def __init__(self, rows=10, cols=10, num_victims=5, num_hospitals=2,
                 num_risk_zones=3, num_blocked=8, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)

        # Cell types
        self.EMPTY = 0
        self.VICTIM = 1
        self.HOSPITAL = 2
        self.RISK_ZONE = 3
        self.BLOCKED = 4
        self.AMBULANCE = 5

        self.victims = []
        self.hospitals = []
        self.risk_zones = []
        self.blocked_cells = []
        self.ambulance_positions = []

        self._generate_environment(num_victims, num_hospitals, num_risk_zones, num_blocked)

    def _generate_environment(self, num_victims, num_hospitals, num_risk_zones, num_blocked):
        """Generate the disaster environment with all elements."""
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        random.shuffle(all_cells)

        idx = 0

        # Place hospitals
        for _ in range(num_hospitals):
            if idx >= len(all_cells):
                break
            r, c = all_cells[idx]
            idx += 1
            self.grid[r][c] = self.HOSPITAL
            self.hospitals.append((r, c))

        # Place victims
        for _ in range(num_victims):
            if idx >= len(all_cells):
                break
            r, c = all_cells[idx]
            idx += 1
            self.grid[r][c] = self.VICTIM
            severity = random.randint(1, 10)
            self.victims.append({
                'id': len(self.victims),
                'position': (r, c),
                'severity': severity,
                'assigned': False,
                'rescued': False
            })

        # Place risk zones
        for _ in range(num_risk_zones):
            if idx >= len(all_cells):
                break
            r, c = all_cells[idx]
            idx += 1
            self.grid[r][c] = self.RISK_ZONE
            risk_level = round(random.uniform(0.3, 0.9), 2)
            self.risk_zones.append({
                'position': (r, c),
                'risk_level': risk_level
            })

        # Place blocked roads
        for _ in range(num_blocked):
            if idx >= len(all_cells):
                break
            r, c = all_cells[idx]
            idx += 1
            self.grid[r][c] = self.BLOCKED
            self.blocked_cells.append((r, c))

        # Place ambulances at hospitals
        for h in self.hospitals:
            self.ambulance_positions.append(h)

    def get_cell(self, row, col):
        """Get the value of a cell."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None

    def is_valid_move(self, row, col):
        """Check if a move to (row, col) is valid."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row][col] != self.BLOCKED

    def get_neighbors(self, row, col):
        """Get valid neighboring cells (up, down, left, right)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if self.is_valid_move(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def get_risk_at(self, row, col):
        """Get the risk level at a specific position."""
        for zone in self.risk_zones:
            if zone['position'] == (row, col):
                return zone['risk_level']
        return 0.0

    def add_blocked_road(self, row, col):
        """Dynamically add a blocked road."""
        if self.grid[row][col] == self.EMPTY:
            self.grid[row][col] = self.BLOCKED
            self.blocked_cells.append((row, col))
            return True
        return False

    def add_victim(self, row, col, severity=None):
        """Dynamically add a new victim."""
        if self.grid[row][col] == self.EMPTY:
            if severity is None:
                severity = random.randint(1, 10)
            self.grid[row][col] = self.VICTIM
            self.victims.append({
                'id': len(self.victims),
                'position': (row, col),
                'severity': severity,
                'assigned': False,
                'rescued': False
            })
            return True
        return False

    def increase_risk(self, row, col, amount=0.2):
        """Increase risk level at a position."""
        for zone in self.risk_zones:
            if zone['position'] == (row, col):
                zone['risk_level'] = min(1.0, zone['risk_level'] + amount)
                return True
        # Create new risk zone if none exists
        if self.grid[row][col] == self.EMPTY:
            self.grid[row][col] = self.RISK_ZONE
            self.risk_zones.append({
                'position': (row, col),
                'risk_level': amount
            })
            return True
        return False

    def remove_victim(self, victim_id):
        """Mark a victim as rescued."""
        for v in self.victims:
            if v['id'] == victim_id:
                v['rescued'] = True
                r, c = v['position']
                self.grid[r][c] = self.EMPTY
                return True
        return False

    def get_unrescued_victims(self):
        """Get all victims that haven't been rescued yet."""
        return [v for v in self.victims if not v['rescued']]

    def display(self):
        """Display the environment grid."""
        symbols = {
            self.EMPTY: '.',
            self.VICTIM: 'V',
            self.HOSPITAL: 'H',
            self.RISK_ZONE: 'R',
            self.BLOCKED: 'X',
            self.AMBULANCE: 'A'
        }
        print("\n" + "=" * (self.cols * 2 + 1))
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                row_str += symbols.get(self.grid[r][c], '?') + " "
            print(row_str + "|")
        print("=" * (self.cols * 2 + 1))
        print("Legend: V=Victim, H=Hospital, R=Risk Zone, X=Blocked, .=Empty")

    def display_with_path(self, path):
        """Display the environment with a path highlighted."""
        path_set = set(path) if path else set()
        symbols = {
            self.EMPTY: '.',
            self.VICTIM: 'V',
            self.HOSPITAL: 'H',
            self.RISK_ZONE: 'R',
            self.BLOCKED: 'X',
            self.AMBULANCE: 'A'
        }
        print("\n" + "=" * (self.cols * 2 + 1))
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                if (r, c) in path_set:
                    row_str += "* "
                else:
                    row_str += symbols.get(self.grid[r][c], '?') + " "
            print(row_str + "|")
        print("=" * (self.cols * 2 + 1))
        print("Legend: V=Victim, H=Hospital, R=Risk Zone, X=Blocked, *=Path, .=Empty")


if __name__ == "__main__":
    env = Environment(seed=42)
    env.display()
    print(f"\nVictims: {env.victims}")
    print(f"Hospitals: {env.hospitals}")
    print(f"Risk Zones: {env.risk_zones}")
    print(f"Blocked Cells: {env.blocked_cells}")
