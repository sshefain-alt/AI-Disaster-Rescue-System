"""
Main Integration File
Connects all AI components together:
- Environment
- Search Algorithms
- CSP
- Machine Learning
- Fuzzy Logic
- Agent
- Dynamic Replanning
"""

from environment import Environment
from search import compare_algorithms
from csp import CSPAmbulanceAllocator
from ml_model import RescueMLModel
from fuzzy import FuzzyRescueDecision
from agent import RescueAgent


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Main function demonstrating the complete AI Disaster Rescue System."""
    print_header("AI DISASTER RESCUE SYSTEM")
    print("Intelligent Hybrid Disaster Management and Rescue Simulation")
    print("Combining: Search | CSP | ML | Fuzzy Logic | Dynamic Replanning")

    # ============================================================
    # STEP 1: Create Environment
    # ============================================================
    print_header("STEP 1: ENVIRONMENT SETUP")
    env = Environment(rows=10, cols=10, num_victims=5, num_hospitals=2,
                      num_risk_zones=3, num_blocked=8, seed=42)
    env.display()

    print(f"\nEnvironment Statistics:")
    print(f"  Grid Size: {env.rows} x {env.cols}")
    print(f"  Victims: {len(env.victims)}")
    print(f"  Hospitals: {len(env.hospitals)}")
    print(f"  Risk Zones: {len(env.risk_zones)}")
    print(f"  Blocked Roads: {len(env.blocked_cells)}")

    # ============================================================
    # STEP 2: Pathfinding Algorithms Comparison
    # ============================================================
    print_header("STEP 2: PATHFINDING ALGORITHMS")
    start = env.hospitals[0]
    goal = env.victims[0]['position']
    compare_algorithms(env, start, goal)

    # ============================================================
    # STEP 3: CSP Ambulance Allocation
    # ============================================================
    print_header("STEP 3: CSP AMBULANCE ALLOCATION")
    allocator = CSPAmbulanceAllocator(env.victims, env.hospitals, ambulance_capacity=3)
    allocator.solve_and_display()

    # ============================================================
    # STEP 4: Machine Learning Models
    # ============================================================
    print_header("STEP 4: MACHINE LEARNING MODELS")
    ml = RescueMLModel(n_samples=500)
    ml.generate_dataset()
    ml.train_models()
    ml.evaluate_models()

    # Sample predictions
    print("\n--- Sample Priority Predictions ---")
    test_cases = [
        (9, 2, 0.8, 1, "Critical victim, close, high risk"),
        (5, 10, 0.4, 2, "Medium priority case"),
        (2, 15, 0.1, 4, "Low priority, far away"),
    ]
    for severity, distance, risk, blockage, desc in test_cases:
        priority, votes = ml.predict_priority(severity, distance, risk, blockage)
        print(f"\n  {desc}")
        print(f"    Input: severity={severity}, distance={distance}, risk={risk}")
        print(f"    Votes: {votes}")
        print(f"    Final: {priority}")

    # ============================================================
    # STEP 5: Fuzzy Logic System
    # ============================================================
    print_header("STEP 5: FUZZY LOGIC DECISION SYSTEM")
    fuzzy = FuzzyRescueDecision()

    fuzzy_cases = [
        (2, 0.8, 9, 0.1, "Close, High Risk, Severe"),
        (15, 0.2, 3, 0.8, "Far, Low Risk, Minor"),
        (8, 0.5, 5, 0.4, "Medium everything"),
    ]
    for distance, risk, severity, blockage, desc in fuzzy_cases:
        result = fuzzy.evaluate(distance, risk, severity, blockage)
        use_risk_path = fuzzy.should_use_risk_aware_path(risk, blockage)
        print(f"\n  {desc}")
        print(f"    Priority Score: {result['priority_score']} ({result['category']})")
        print(f"    Use Risk-Aware Path: {use_risk_path}")

    # ============================================================
    # STEP 6: Rescue Agent - Full Mission
    # ============================================================
    print_header("STEP 6: RESCUE AGENT MISSION")
    agent = RescueAgent(env, env.hospitals)
    agent.run_rescue_mission()

    # ============================================================
    # STEP 7: Dynamic Replanning Scenarios
    # ============================================================
    print_header("STEP 7: DYNAMIC REPLANNING SCENARIOS")

    # Scenario 1: Road Blockage
    print("\n--- Scenario 1: Road Blockage ---")
    agent.handle_road_blockage(3, 3)
    env.display()

    # Scenario 2: Risk Increase
    print("\n--- Scenario 2: Risk Increase ---")
    agent.handle_risk_increase(5, 5, 0.3)
    env.display()

    # Scenario 3: New Victim
    print("\n--- Scenario 3: New Victim ---")
    agent.handle_new_victim(7, 7, severity=8)
    env.display()

    # Scenario 4: Resource Depletion
    print("\n--- Scenario 4: Resource Depletion ---")
    agent.handle_resource_depletion(0)

    # ============================================================
    # Final Statistics
    # ============================================================
    print_header("FINAL STATISTICS")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print_header("SYSTEM DEMONSTRATION COMPLETE")
    print("All AI components have been successfully integrated and demonstrated.")
    print("Run 'python app.py' to launch the interactive web interface.")


if __name__ == "__main__":
    main()
