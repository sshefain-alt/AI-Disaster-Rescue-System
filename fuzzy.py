"""
Fuzzy Logic Module
Implements:
- Fuzzification
- Fuzzy inference rules
- Uncertainty handling
- Defuzzification

Handles uncertainty using fuzzy inference based on:
- Distance
- Risk
- Severity
- Blockage probability
"""

import numpy as np


class FuzzyRescueDecision:
    """
    Fuzzy logic system for rescue decision-making under uncertainty.
    Uses fuzzy sets and rules to determine rescue priority and approach.
    """

    def __init__(self):
        # Define fuzzy set ranges
        self.distance_range = np.linspace(0, 20, 100)    # 0-20 cells
        self.risk_range = np.linspace(0, 1, 100)          # 0.0-1.0
        self.severity_range = np.linspace(0, 10, 100)     # 0-10
        self.blockage_range = np.linspace(0, 1, 100)      # 0.0-1.0
        self.output_range = np.linspace(0, 10, 100)       # 0-10 priority score

    # ==================== MEMBERSHIP FUNCTIONS ====================

    def trimf(self, x, a, b, c):
        """Triangular membership function."""
        left = 0.0 if b == a else (x - a) / (b - a)
        right = 0.0 if c == b else (c - x) / (c - b)
        return np.maximum(0, np.minimum(left, right))

    def trapmf(self, x, a, b, c, d):
        """Trapezoidal membership function."""
        return np.maximum(0, np.minimum(
            np.minimum((x - a) / (b - a), 1),
            (d - x) / (d - c)
        ))

    # Distance membership functions
    def distance_near(self, x):
        return self.trimf(x, 0, 2, 6)

    def distance_medium(self, x):
        return self.trimf(x, 4, 8, 14)

    def distance_far(self, x):
        return self.trimf(x, 12, 16, 20)

    # Risk membership functions
    def risk_low(self, x):
        return self.trimf(x, 0, 0.1, 0.4)

    def risk_medium(self, x):
        return self.trimf(x, 0.3, 0.5, 0.7)

    def risk_high(self, x):
        return self.trimf(x, 0.6, 0.8, 1.0)

    # Severity membership functions
    def severity_low(self, x):
        return self.trimf(x, 0, 1, 4)

    def severity_medium(self, x):
        return self.trimf(x, 3, 5, 7)

    def severity_high(self, x):
        return self.trimf(x, 6, 8, 10)

    # Blockage membership functions
    def blockage_low(self, x):
        return self.trimf(x, 0, 0.1, 0.4)

    def blockage_high(self, x):
        return self.trimf(x, 0.3, 0.6, 1.0)

    # Output membership functions
    def priority_very_low(self, x):
        return self.trimf(x, 0, 0, 2)

    def priority_low(self, x):
        return self.trimf(x, 1, 2.5, 4)

    def priority_medium(self, x):
        return self.trimf(x, 3, 5, 7)

    def priority_high(self, x):
        return self.trimf(x, 6, 7.5, 9)

    def priority_very_high(self, x):
        return self.trimf(x, 8, 9, 10)

    # ==================== FUZZIFICATION ====================

    def fuzzify(self, distance, risk, severity, blockage):
        """
        Convert crisp inputs to fuzzy membership values.
        """
        fuzzy_inputs = {
            'distance': {
                'near': float(self.distance_near(distance)),
                'medium': float(self.distance_medium(distance)),
                'far': float(self.distance_far(distance))
            },
            'risk': {
                'low': float(self.risk_low(risk)),
                'medium': float(self.risk_medium(risk)),
                'high': float(self.risk_high(risk))
            },
            'severity': {
                'low': float(self.severity_low(severity)),
                'medium': float(self.severity_medium(severity)),
                'high': float(self.severity_high(severity))
            },
            'blockage': {
                'low': float(self.blockage_low(blockage)),
                'high': float(self.blockage_high(blockage))
            }
        }
        return fuzzy_inputs

    # ==================== FUZZY INFERENCE ====================

    def apply_rules(self, fuzzy_inputs):
        """
        Apply fuzzy inference rules.
        Returns aggregated output membership function.
        """
        # Initialize output aggregation
        aggregated = np.zeros_like(self.output_range)

        # Rule base
        # Format: (distance, risk, severity, blockage) -> output
        rules = [
            # High priority rules
            ('near', 'high', 'high', 'low', 'very_high'),
            ('near', 'high', 'high', 'high', 'high'),
            ('near', 'medium', 'high', 'low', 'high'),
            ('medium', 'high', 'high', 'low', 'high'),
            ('near', 'high', 'medium', 'low', 'high'),

            # Medium priority rules
            ('near', 'medium', 'medium', 'low', 'medium'),
            ('medium', 'medium', 'high', 'low', 'medium'),
            ('medium', 'high', 'medium', 'low', 'medium'),
            ('far', 'high', 'high', 'low', 'medium'),
            ('near', 'low', 'high', 'low', 'medium'),

            # Low priority rules
            ('far', 'low', 'low', 'low', 'very_low'),
            ('far', 'medium', 'low', 'low', 'low'),
            ('medium', 'low', 'low', 'low', 'low'),
            ('far', 'high', 'low', 'low', 'low'),
            ('far', 'low', 'medium', 'low', 'low'),

            # Blockage impact rules
            ('near', 'high', 'high', 'high', 'medium'),
            ('far', 'low', 'low', 'high', 'very_low'),
            ('medium', 'medium', 'medium', 'high', 'low'),
        ]

        rule_activations = []

        for dist, risk, sev, block, output in rules:
            # Get membership values
            mu_dist = fuzzy_inputs['distance'][dist]
            mu_risk = fuzzy_inputs['risk'][risk]
            mu_sev = fuzzy_inputs['severity'][sev]
            mu_block = fuzzy_inputs['blockage'][block]

            # Firing strength (AND = min)
            activation = min(mu_dist, mu_risk, mu_sev, mu_block)

            if activation > 0:
                rule_activations.append({
                    'rule': f"IF dist={dist} AND risk={risk} AND sev={sev} AND block={block} THEN {output}",
                    'activation': activation,
                    'output': output
                })

                # Get output membership function
                if output == 'very_low':
                    output_mf = self.priority_very_low(self.output_range)
                elif output == 'low':
                    output_mf = self.priority_low(self.output_range)
                elif output == 'medium':
                    output_mf = self.priority_medium(self.output_range)
                elif output == 'high':
                    output_mf = self.priority_high(self.output_range)
                else:  # very_high
                    output_mf = self.priority_very_high(self.output_range)

                # Aggregate (OR = max)
                aggregated = np.maximum(aggregated, np.minimum(activation, output_mf))

        return aggregated, rule_activations

    # ==================== DEFUZZIFICATION ====================

    def defuzzify_centroid(self, aggregated):
        """Defuzzify using centroid method."""
        if np.sum(aggregated) == 0:
            return 0.0
        return np.sum(self.output_range * aggregated) / np.sum(aggregated)

    def defuzzify_mom(self, aggregated):
        """Defuzzify using Mean of Maximum method."""
        max_val = np.max(aggregated)
        if max_val == 0:
            return 0.0
        mom_indices = np.where(aggregated == max_val)[0]
        return np.mean(self.output_range[mom_indices])

    # ==================== MAIN INTERFACE ====================

    def evaluate(self, distance, risk, severity, blockage_probability):
        """
        Main fuzzy logic evaluation.

        Args:
            distance: Distance to victim (0-20)
            risk: Risk level (0.0-1.0)
            severity: Victim severity (0-10)
            blockage_probability: Blockage probability (0.0-1.0)

        Returns:
            result: Dict with priority score, category, and details
        """
        # Fuzzify inputs
        fuzzy_inputs = self.fuzzify(distance, risk, severity, blockage_probability)

        # Apply rules
        aggregated, rule_activations = self.apply_rules(fuzzy_inputs)

        # Defuzzify
        priority_score = self.defuzzify_centroid(aggregated)

        # Categorize
        if priority_score >= 8:
            category = 'VERY HIGH'
        elif priority_score >= 6:
            category = 'HIGH'
        elif priority_score >= 4:
            category = 'MEDIUM'
        elif priority_score >= 2:
            category = 'LOW'
        else:
            category = 'VERY LOW'

        return {
            'priority_score': round(priority_score, 2),
            'category': category,
            'fuzzy_inputs': fuzzy_inputs,
            'rule_activations': rule_activations,
            'aggregated_output': aggregated
        }

    def should_use_risk_aware_path(self, risk, blockage_probability):
        """
        Decide whether to use risk-aware path planning based on fuzzy logic.

        Returns:
            bool: True if risk-aware path should be used
        """
        # Simple fuzzy decision
        risk_membership_high = float(self.risk_high(risk))
        blockage_membership_high = float(self.blockage_high(blockage_probability))

        # If either risk or blockage is high, use risk-aware path
        decision = max(risk_membership_high, blockage_membership_high)
        return decision > 0.5


if __name__ == "__main__":
    fuzzy = FuzzyRescueDecision()

    print("=" * 60)
    print("FUZZY LOGIC RESCUE DECISION SYSTEM")
    print("=" * 60)

    test_cases = [
        (2, 0.8, 9, 0.1, "Close, High Risk, Severe, Low Blockage"),
        (15, 0.2, 3, 0.8, "Far, Low Risk, Minor, High Blockage"),
        (8, 0.5, 5, 0.4, "Medium everything"),
        (1, 0.9, 10, 0.0, "Very Close, Very High Risk, Critical"),
    ]

    for distance, risk, severity, blockage, description in test_cases:
        result = fuzzy.evaluate(distance, risk, severity, blockage)
        use_risk_path = fuzzy.should_use_risk_aware_path(risk, blockage)

        print(f"\n--- {description} ---")
        print(f"  Inputs: distance={distance}, risk={risk}, severity={severity}, blockage={blockage}")
        print(f"  Priority Score: {result['priority_score']}")
        print(f"  Category: {result['category']}")
        print(f"  Use Risk-Aware Path: {use_risk_path}")
        print(f"  Active Rules: {len(result['rule_activations'])}")
