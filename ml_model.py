"""
Machine Learning Module
Contains:
- ML dataset generation
- Model training (kNN, Naive Bayes, Decision Tree)
- Evaluation metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- Priority prediction using majority voting
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


class RescueMLModel:
    """
    ML-based decision support system for rescue priority prediction.
    Uses three models with majority voting:
    - k-Nearest Neighbors (kNN)
    - Naive Bayes
    - Decision Tree
    """

    def __init__(self, n_samples=500, test_size=0.3, random_state=42):
        self.n_samples = n_samples
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.knn_model = None
        self.nb_model = None
        self.dt_model = None

        self.metrics = {}

        self.classes = ['LOW', 'MEDIUM', 'HIGH']

    def generate_dataset(self):
        """
        Generate synthetic rescue dataset.
        Features: [severity, distance, risk_level, blockage_proximity]
        Target: priority (LOW=0, MEDIUM=1, HIGH=2)
        """
        np.random.seed(self.random_state)

        # Generate features
        severity = np.random.randint(1, 11, self.n_samples)  # 1-10
        distance = np.random.randint(1, 20, self.n_samples)   # 1-19
        risk_level = np.random.uniform(0, 1, self.n_samples)   # 0.0-1.0
        blockage_proximity = np.random.randint(0, 5, self.n_samples)  # 0-4

        X = np.column_stack([severity, distance, risk_level, blockage_proximity])

        # Generate target based on weighted scoring
        # Higher severity + closer distance + higher risk = higher priority
        priority_score = (
            severity * 0.4 +
            (20 - distance) * 0.2 +
            risk_level * 10 * 0.3 +
            (5 - blockage_proximity) * 0.1
        )

        # Convert to classes
        y = np.where(priority_score > 7, 2, np.where(priority_score > 4, 1, 0))

        self.X = X
        self.y = y

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"Dataset generated: {self.n_samples} samples")
        print(f"  Training: {len(self.X_train)}, Testing: {len(self.X_test)}")
        print(f"  Class distribution: {dict(Counter(self.y))}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Train all three ML models."""
        if self.X_train is None:
            self.generate_dataset()

        print("\nTraining ML Models...")
        print("-" * 40)

        # kNN
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.knn_model.fit(self.X_train, self.y_train)
        print("kNN model trained.")

        # Naive Bayes
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.X_train, self.y_train)
        print("Naive Bayes model trained.")

        # Decision Tree
        self.dt_model = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
        self.dt_model.fit(self.X_train, self.y_train)
        print("Decision Tree model trained.")

    def evaluate_models(self):
        """Evaluate all models and compute metrics."""
        if self.knn_model is None:
            self.train_models()

        print("\n" + "=" * 60)
        print("ML MODEL EVALUATION")
        print("=" * 60)

        models = {
            'kNN': self.knn_model,
            'Naive Bayes': self.nb_model,
            'Decision Tree': self.dt_model
        }

        for name, model in models.items():
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            self.metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm
            }

            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {cm}")

        # Ensemble (majority voting) evaluation
        self._evaluate_ensemble()

        return self.metrics

    def _evaluate_ensemble(self):
        """Evaluate ensemble majority voting."""
        knn_pred = self.knn_model.predict(self.X_test)
        nb_pred = self.nb_model.predict(self.X_test)
        dt_pred = self.dt_model.predict(self.X_test)

        # Majority voting
        ensemble_pred = []
        for i in range(len(self.y_test)):
            votes = [knn_pred[i], nb_pred[i], dt_pred[i]]
            majority = Counter(votes).most_common(1)[0][0]
            ensemble_pred.append(majority)

        ensemble_pred = np.array(ensemble_pred)

        accuracy = accuracy_score(self.y_test, ensemble_pred)
        precision = precision_score(self.y_test, ensemble_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, ensemble_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, ensemble_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(self.y_test, ensemble_pred)

        self.metrics['Ensemble (Majority Vote)'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

        print(f"\nEnsemble (Majority Vote):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm}")

    def predict_priority(self, severity, distance, risk_level, blockage_proximity=0):
        """
        Predict rescue priority using majority voting.

        Args:
            severity: Victim severity (1-10)
            distance: Distance to victim
            risk_level: Risk level (0.0-1.0)
            blockage_proximity: Proximity to blocked roads (0-4)

        Returns:
            priority: 'HIGH', 'MEDIUM', or 'LOW'
            votes: Dict of individual model predictions
        """
        if self.knn_model is None:
            self.train_models()

        features = np.array([[severity, distance, risk_level, blockage_proximity]])

        knn_pred = self.knn_model.predict(features)[0]
        nb_pred = self.nb_model.predict(features)[0]
        dt_pred = self.dt_model.predict(features)[0]

        # Majority voting
        votes_list = [knn_pred, nb_pred, dt_pred]
        majority = Counter(votes_list).most_common(1)[0][0]

        priority_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

        votes = {
            'kNN': priority_map[knn_pred],
            'Naive Bayes': priority_map[nb_pred],
            'Decision Tree': priority_map[dt_pred]
        }

        return priority_map[majority], votes

    def get_classification_report(self):
        """Get detailed classification report for all models."""
        if self.knn_model is None:
            self.train_models()

        models = {
            'kNN': self.knn_model,
            'Naive Bayes': self.nb_model,
            'Decision Tree': self.dt_model
        }

        reports = {}
        for name, model in models.items():
            y_pred = model.predict(self.X_test)
            report = classification_report(
                self.y_test, y_pred,
                target_names=self.classes,
                zero_division=0
            )
            reports[name] = report

        return reports


if __name__ == "__main__":
    ml = RescueMLModel(n_samples=500)
    ml.generate_dataset()
    ml.train_models()
    ml.evaluate_models()

    # Test prediction
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    test_cases = [
        (9, 2, 0.8, 1),   # High severity, close, high risk
        (5, 10, 0.4, 2),  # Medium everything
        (2, 15, 0.1, 4),  # Low severity, far, low risk
    ]

    for severity, distance, risk, blockage in test_cases:
        priority, votes = ml.predict_priority(severity, distance, risk, blockage)
        print(f"\nInput: severity={severity}, distance={distance}, risk={risk}, blockage={blockage}")
        print(f"  Votes: {votes}")
        print(f"  Final Priority: {priority}")
