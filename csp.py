"""
Constraint Satisfaction Problem (CSP) Module
Implements:
- CSP resource allocation
- Backtracking
- MRV-inspired heuristic (Minimum Remaining Values)
- Capacity constraints enforced
- Resource allocation optimization
"""

from itertools import product


class CSPAmbulanceAllocator:
    """
    CSP-based ambulance assignment system.
    Assigns victims to ambulances with constraints:
    - Each ambulance has a capacity limit
    - Each victim assigned to exactly one ambulance
    - Minimize total distance/cost
    """

    def __init__(self, victims, hospitals, ambulance_capacity=2):
        """
        Initialize CSP allocator.

        Args:
            victims: List of victim dicts with 'id', 'position', 'severity'
            hospitals: List of hospital positions (ambulance starting points)
            ambulance_capacity: Max victims per ambulance
        """
        self.victims = [v for v in victims if not v.get('rescued', False)]
        self.hospitals = hospitals
        self.ambulance_capacity = ambulance_capacity
        self.num_ambulances = len(hospitals)

        # Domains: each victim can be assigned to any ambulance
        self.domains = {}
        for v in self.victims:
            self.domains[v['id']] = list(range(self.num_ambulances))

        # Precompute distances
        self.distances = self._compute_distances()

    def _compute_distances(self):
        """Compute Manhattan distance from each hospital to each victim."""
        distances = {}
        for h_idx, h_pos in enumerate(self.hospitals):
            for v in self.victims:
                dist = abs(h_pos[0] - v['position'][0]) + abs(h_pos[1] - v['position'][1])
                distances[(v['id'], h_idx)] = dist
        return distances

    def is_consistent(self, assignment, victim_id, ambulance_id):
        """
        Check if assigning victim_id to ambulance_id is consistent with constraints.

        Constraints:
        1. Capacity: ambulance can't exceed its capacity
        2. Each victim assigned to exactly one ambulance
        """
        # Check capacity constraint
        count = sum(1 for a in assignment.values() if a == ambulance_id)
        if count >= self.ambulance_capacity:
            return False

        return True

    def backtracking_search(self, use_mrv=True):
        """
        Solve CSP using backtracking search.

        Args:
            use_mrv: If True, use MRV heuristic to select next variable

        Returns:
            assignment: Dict mapping victim_id to ambulance_id, or None if no solution
        """
        assignment = {}
        return self._backtrack(assignment, use_mrv)

    def _backtrack(self, assignment, use_mrv):
        """Recursive backtracking algorithm."""
        # Check if complete
        if len(assignment) == len(self.victims):
            return assignment.copy()

        # Select unassigned variable
        if use_mrv:
            var = self._select_unassigned_variable_mrv(assignment)
        else:
            var = self._select_unassigned_variable_simple(assignment)

        if var is None:
            return None

        # Try values in order
        for value in self._order_domain_values(var, assignment):
            if self.is_consistent(assignment, var, value):
                assignment[var] = value

                result = self._backtrack(assignment, use_mrv)
                if result is not None:
                    return result

                del assignment[var]

        return None

    def _select_unassigned_variable_simple(self, assignment):
        """Select first unassigned variable."""
        for v in self.victims:
            if v['id'] not in assignment:
                return v['id']
        return None

    def _select_unassigned_variable_mrv(self, assignment):
        """
        MRV (Minimum Remaining Values) Heuristic:
        Select the variable with the fewest legal values remaining.
        This helps fail early and prune the search tree.
        """
        unassigned = [v for v in self.victims if v['id'] not in assignment]
        if not unassigned:
            return None

        min_remaining = float('inf')
        best_var = None

        for v in unassigned:
            remaining = 0
            for ambulance_id in self.domains[v['id']]:
                if self.is_consistent(assignment, v['id'], ambulance_id):
                    remaining += 1

            if remaining < min_remaining:
                min_remaining = remaining
                best_var = v['id']

            # Early termination if any variable has 0 remaining values
            if remaining == 0:
                return v['id']

        return best_var

    def _order_domain_values(self, victim_id, assignment):
        """
        Order domain values by least constraining value heuristic.
        Prefer assignments that leave more options for other variables.
        """
        values = self.domains[victim_id]
        # Sort by distance (prefer closer ambulances)
        sorted_values = sorted(values, key=lambda a: self.distances.get((victim_id, a), float('inf')))
        return sorted_values

    def get_total_cost(self, assignment):
        """Calculate total cost of an assignment."""
        if assignment is None:
            return float('inf')
        return sum(self.distances.get((v_id, a), 0) for v_id, a in assignment.items())

    def get_assignment_details(self, assignment):
        """Get detailed assignment information."""
        if assignment is None:
            return None

        details = []
        for v in self.victims:
            if v['id'] in assignment:
                amb_id = assignment[v['id']]
                dist = self.distances.get((v['id'], amb_id), 0)
                details.append({
                    'victim_id': v['id'],
                    'victim_position': v['position'],
                    'severity': v['severity'],
                    'ambulance_id': amb_id,
                    'hospital_position': self.hospitals[amb_id],
                    'distance': dist
                })
        return details

    def solve_and_display(self):
        """Solve the CSP and display results."""
        print("\n" + "=" * 60)
        print("CSP AMBULANCE ALLOCATION")
        print("=" * 60)

        print(f"\nAmbulances: {self.num_ambulances} (capacity: {self.ambulance_capacity} each)")
        print(f"Victims to assign: {len(self.victims)}")

        # Solve without MRV
        print("\n--- Without MRV Heuristic ---")
        assignment_simple = self.backtracking_search(use_mrv=False)
        if assignment_simple:
            cost = self.get_total_cost(assignment_simple)
            print(f"Solution found! Total cost: {cost}")
            self._print_assignment(assignment_simple)
        else:
            print("No solution found!")

        # Solve with MRV
        print("\n--- With MRV Heuristic ---")
        assignment_mrv = self.backtracking_search(use_mrv=True)
        if assignment_mrv:
            cost = self.get_total_cost(assignment_mrv)
            print(f"Solution found! Total cost: {cost}")
            self._print_assignment(assignment_mrv)
        else:
            print("No solution found!")

        return assignment_mrv

    def _print_assignment(self, assignment):
        """Print assignment details."""
        details = self.get_assignment_details(assignment)
        if details:
            for d in details:
                print(f"  Victim {d['victim_id']} at {d['victim_position']} "
                      f"(severity: {d['severity']}) -> "
                      f"Ambulance {d['ambulance_id']} at {d['hospital_position']} "
                      f"[distance: {d['distance']}]")


if __name__ == "__main__":
    from environment import Environment

    env = Environment(seed=42)
    env.display()

    allocator = CSPAmbulanceAllocator(env.victims, env.hospitals, ambulance_capacity=2)
    allocator.solve_and_display()
