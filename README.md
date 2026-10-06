# AI Disaster Rescue System

An intelligent hybrid disaster management and rescue simulation platform designed to support emergency response operations in dynamic and uncertain environments.

## Overview

The AI Disaster Rescue System combines multiple Artificial Intelligence techniques including:

- **Search Algorithms** (BFS, DFS, A*, Risk-Aware A*, Greedy Best-First, Hill Climbing)
- **Constraint Satisfaction Problems (CSP)** with Backtracking and MRV heuristic
- **Machine Learning** (kNN, Naive Bayes, Decision Tree)
- **Fuzzy Logic** for uncertainty handling
- **Dynamic Replanning** for real-time adaptation

## Features

### Intelligent Victim Selection
Selects the most suitable victim based on:
- Severity
- Distance
- Risk level
- Fuzzy logic priority

### Pathfinding Algorithms
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search
- Risk-Aware A* Search
- Greedy Best-First Search
- Hill Climbing

### CSP Resource Allocation
- Ambulance assignment using Backtracking
- MRV-inspired heuristic
- Capacity constraints enforced
- Resource allocation optimization

### Machine Learning Integration
Three ML models for rescue priority prediction:
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Decision Tree

### Fuzzy Logic for Uncertainty Handling
- Fuzzification
- Fuzzy inference rules
- Uncertainty handling based on distance, risk, severity, and blockage probability

### Dynamic Replanning
- Road blockage handling
- Risk level adaptation
- New victim integration
- Resource depletion management

## Project Structure

```
AI-Disaster-Rescue-System/
│
├── agent.py           # Rescue agent with victim selection and decision-making
├── app.py             # Streamlit web interface
├── csp.py             # CSP resource allocation with backtracking
├── environment.py     # Disaster environment (grid, victims, hospitals, risks)
├── fuzzy.py           # Fuzzy logic inference system
├── logo.png           # Project logo
├── main.py            # Main integration file
├── ml_model.py        # ML models (kNN, Naive Bayes, Decision Tree)
├── requirements.txt   # Python dependencies
├── search.py          # Pathfinding algorithms
└── README.md          # This file
```

## Installation

### Clone Repository
```bash
git clone https://github.com/SanaAli17/AI-Disaster-Rescue-System.git
```

### Move Into Project Directory
```bash
cd AI-Disaster-Rescue-System
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

### Run Main Simulation
```bash
python main.py
```

### Run Web Interface
```bash
streamlit run app.py
```

## Machine Learning Evaluation

The system evaluates ML models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Dynamic Scenarios

The project supports multiple real-time dynamic scenarios:

| Scenario | Description | Response |
|----------|-------------|----------|
| Road Blockage | Road becomes blocked | Recomputes path using A* |
| Risk Increase | Risk level increases | Switches to Risk-Aware A* |
| New Victim | New victim appears | Re-evaluates victim priorities |
| Resource Depletion | Ambulance unavailable | Reassigns victims using CSP |

## Technologies Used

- **Python** - Programming Language
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Streamlit** - Web interface
- **Plotly** - Visualization
- **Pandas** - Data manipulation

## Future Improvements

- Real-time map integration
- Multi-agent coordination
- Reinforcement Learning
- Deep Learning-based prediction
- Live disaster sensor integration
- GPS integration
- Real-time traffic data

## License

This project is developed for University AI Lab purposes.
