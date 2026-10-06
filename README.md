# AI-Disaster-Rescue-System
An AI-powered disaster management system leveraging deep learning and computer vision to streamline emergency response. It uses real-time computer vision (YOLOv5) to detect victims in drone or satellite footage, processes live sensor data to predict risk zones, and applies AI routing algorithms to guide rescue teams efficiently.


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
#NAME?
#NAME?
- Risk level
- Fuzzy logic priority

### Pathfinding Algorithms
#NAME?
#NAME?
#NAME?
#NAME?
- Greedy Best-First Search
- Hill Climbing

### CSP Resource Allocation
- Ambulance assignment using Backtracking
#NAME?
- Capacity constraints enforced
- Resource allocation optimization

### Machine Learning Integration
Three ML models for rescue priority prediction:
#NAME?
- Naive Bayes
- Decision Tree

### Fuzzy Logic for Uncertainty Handling
#NAME?
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
#NAME?
#NAME?
#NAME?
#NAME?
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

#NAME?
#NAME?
- Reinforcement Learning
- Deep Learning-based prediction
- Live disaster sensor integration
- GPS integration
#NAME?

## License

This project is developed for University AI Lab purposes.
<img width="91" height="3433" alt="image" src="https://github.com/user-attachments/assets/b83c613d-f0fa-44b9-b915-1a3d3a9a0ece" />
