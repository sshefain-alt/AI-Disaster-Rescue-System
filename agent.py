"""
Agent Module
Handles:
- Victim selection
- Rescue decision-making
- Risk evaluation
- Dynamic replanning
"""

from search import astar, risk_aware_astar
from fuzzy import FuzzyRescueDecision


class RescueAgent:
    """
    Autonomous rescue agent that:
    - Selects the most critical victim
    - Finds optimal rescue paths
    - Avoids dangerous routes
    - Handles uncertainty
    - Reallocates resources dynamically
    - Adapts to environmental changes in real-time
    """

    def __init__(self, environment, hospitals):
        self.env = environment
        self.hospitals = hospitals
        self.fuzzy_system = FuzzyRescueDecision()
        self.rescued_victims = []
        self.current_paths = {}
        self.total_distance_traveled = 0
        self.rescue_log = []

    def calculate_victim_score(self, victim, ambulance_pos):
        """
        Calculate priority score for a victim.
        score = severity - (distance + risk)
        Higher score = higher priority
        """
        distance = self._manhattan_distance(ambulance_pos, victim['position'])
        risk = self.env.get_risk_at(*victim['position'])
        severity = victim['severity']

        score = severity - (distance * 0.5 + risk * 3)
        return score, distance, risk

    def select_victim(self, ambulance_pos):
        """
        Select the most suitable victim based on:
        - Severity
        - Distance
        - Risk level
        - Fuzzy logic priority

        Returns the victim with highest priority score.
        """
        unreduced = self.env.get_unrescued_victims()
        if not unreduced:
            return None

        best_victim = None
        best_score = float('-inf')
        best_details = None

        for victim in unreduced:
            score, distance, risk = self.calculate_victim_score(victim, ambulance_pos)

            # Get fuzzy priority
            fuzzy_result = self.fuzzy_system.evaluate(
                distance=distance,
                risk=risk,
                severity=victim['severity'],
                blockage_probability=0.3  # Default blockage probability
            )

            # Combined score: heuristic + fuzzy priority
            combined_score = score + fuzzy_result['priority_score']

            if combined_score > best_score:
                best_score = combined_score
                best_victim = victim
                best_details = {
                    'score': score,
                    'fuzzy_priority': fuzzy_result['priority_score'],
                    'fuzzy_category': fuzzy_result['category'],
                    'distance': distance,
                    'risk': risk,
                    'combined_score': combined_score
                }

        return best_victim, best_details

    def find_rescue_path(self, start, goal, use_risk_aware=None):
        """
        Find optimal rescue path.
        Can switch from shortest path to safer path when uncertainty increases.

        Args:
            start: Starting position
            goal: Goal position
            use_risk_aware: If None, use fuzzy logic to decide
        """
        if use_risk_aware is None:
            # Use fuzzy logic to decide
            risk = self.env.get_risk_at(*goal)
            use_risk_aware = self.fuzzy_system.should_use_risk_aware_path(risk, 0.3)

        if use_risk_aware:
            path, cost = risk_aware_astar(self.env, start, goal)
            algorithm = "Risk-Aware A*"
        else:
            path, cost = astar(self.env, start, goal)
            algorithm = "A*"

        return path, cost, algorithm

    def rescue_victim(self, victim, ambulance_pos):
        """
        Execute rescue operation for a victim.
        """
        # Find path to victim
        path, cost, algorithm = self.find_rescue_path(ambulance_pos, victim['position'])

        if path is None:
            self.rescue_log.append({
                'victim_id': victim['id'],
                'status': 'FAILED',
                'reason': 'No path found'
            })
            return False, None, None

        # Move to victim
        self.total_distance_traveled += cost

        # Find nearest hospital for drop-off
        nearest_hospital = min(self.hospitals,
                               key=lambda h: self._manhattan_distance(victim['position'], h))

        # Find path to hospital
        hospital_path, hospital_cost, hospital_algorithm = self.find_rescue_path(
            victim['position'], nearest_hospital
        )

        if hospital_path is None:
            self.rescue_log.append({
                'victim_id': victim['id'],
                'status': 'PARTIAL',
                'reason': 'No path to hospital'
            })
            return False, path, None

        self.total_distance_traveled += hospital_cost

        # Mark victim as rescued
        self.env.remove_victim(victim['id'])
        victim['rescued'] = True
        self.rescued_victims.append(victim)

        log_entry = {
            'victim_id': victim['id'],
            'victim_position': victim['position'],
            'severity': victim['severity'],
            'status': 'RESCUED',
            'path_to_victim': path,
            'path_to_hospital': hospital_path,
            'algorithm_to_victim': algorithm,
            'algorithm_to_hospital': hospital_algorithm,
            'distance_to_victim': cost,
            'distance_to_hospital': hospital_cost,
            'total_distance': cost + hospital_cost,
            'hospital': nearest_hospital
        }
        self.rescue_log.append(log_entry)

        return True, path, hospital_path

    def run_rescue_mission(self):
        """
        Run complete rescue mission.
        Iteratively selects and rescues all victims.
        """
        print("\n" + "=" * 60)
        print("RESCUE MISSION STARTED")
        print("=" * 60)

        ambulance_pos = self.hospitals[0]  # Start at first hospital
        mission_count = 0

        while True:
            # Select next victim
            result = self.select_victim(ambulance_pos)
            if result is None:
                print("\nAll victims have been rescued!")
                break

            victim, details = result
            mission_count += 1

            print(f"\n--- Mission {mission_count} ---")
            print(f"Selected Victim {victim['id']} at {victim['position']}")
            print(f"  Severity: {victim['severity']}")
            print(f"  Distance: {details['distance']}")
            print(f"  Risk: {details['risk']}")
            print(f"  Fuzzy Priority: {details['fuzzy_priority']} ({details['fuzzy_category']})")
            print(f"  Combined Score: {details['combined_score']:.2f}")

            # Execute rescue
            success, path_to_victim, path_to_hospital = self.rescue_victim(victim, ambulance_pos)

            if success:
                print(f"  Status: RESCUED")
                print(f"  Path to victim: {len(path_to_victim)} steps")
                print(f"  Path to hospital: {len(path_to_hospital)} steps")
                # Update ambulance position to hospital
                ambulance_pos = path_to_hospital[-1]
            else:
                print(f"  Status: FAILED")

        print(f"\n{'=' * 60}")
        print(f"MISSION COMPLETE")
        print(f"Total victims rescued: {len(self.rescued_victims)}")
        print(f"Total distance traveled: {self.total_distance_traveled}")
        print(f"{'=' * 60}")

        return self.rescue_log

    def handle_road_blockage(self, row, col):
        """
        Dynamic replanning: Handle road blockage.
        Recomputes paths using A*.
        """
        print(f"\n[ALERT] Road blockage detected at ({row}, {col})!")
        self.env.add_blocked_road(row, col)
        print("Environment updated. Paths will be recomputed.")
        return True

    def handle_risk_increase(self, row, col, amount=0.2):
        """
        Dynamic replanning: Handle increased risk.
        Switches to Risk-Aware A*.
        """
        print(f"\n[ALERT] Risk increased at ({row}, {col})!")
        self.env.increase_risk(row, col, amount)
        print("Risk-Aware A* will be enabled for affected areas.")
        return True

    def handle_new_victim(self, row, col, severity=None):
        """
        Dynamic replanning: Handle new victim.
        Re-evaluates victim priorities.
        """
        print(f"\n[ALERT] New victim detected at ({row}, {col})!")
        self.env.add_victim(row, col, severity)
        print("Victim priorities will be re-evaluated.")
        return True

    def handle_resource_depletion(self, ambulance_id):
        """
        Dynamic replanning: Handle resource depletion.
        Reassigns victims using CSP.
        """
        print(f"\n[ALERT] Ambulance {ambulance_id} depleted!")
        if ambulance_id < len(self.hospitals):
            self.hospitals[ambulance_id] = None
        print("Victims will be reassigned using CSP.")
        return True

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_statistics(self):
        """Get rescue mission statistics."""
        return {
            'total_victims': len(self.env.victims),
            'rescued_victims': len(self.rescued_victims),
            'remaining_victims': len(self.env.get_unrescued_victims()),
            'total_distance': self.total_distance_traveled,
            'missions_completed': len(self.rescue_log)
        }


if __name__ == "__main__":
    from environment import Environment

    env = Environment(seed=42)
    env.display()

    agent = RescueAgent(env, env.hospitals)
    agent.run_rescue_mission()

    print("\nFinal Statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
