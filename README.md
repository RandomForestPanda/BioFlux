# NDVI-Based Predator-Prey Simulation

## Overview
This project integrates NDVI (Normalized Difference Vegetation Index) data to simulate the complex interactions between vegetation, herbivores, and carnivores. The simulation employs an underlying reinforcement learning agent to model population dynamics and ecological balance, with the goal of visualizing these interactions as a heat map.

## Features
- **NDVI-Based Environment:** The simulation uses NDVI data to determine vegetation density and its impact on herbivore movement and reproduction.
- **Herbivore and Carnivore Populations:** Herbivores consume vegetation, while carnivores hunt herbivores, creating a dynamic ecosystem.
- **Reinforcement Learning Integration:** A reinforcement learning model governs agent behaviors, optimizing survival strategies.
- **Heat Map Visualization:** The population densities of vegetation, herbivores, and carnivores are rendered as heat maps for intuitive analysis.
- **Seasonal NDVI Variations:** The environment periodically updates NDVI values to simulate seasonal vegetation changes.

## Simulation Process
1. **NDVI Data Initialization:** The environment is initialized with a grid representing NDVI values.
2. **Agent Placement:** Herbivores and carnivores are randomly distributed across the grid.
3. **Agent Movement:** Herbivores move towards higher NDVI regions, consuming vegetation, while carnivores pursue herbivores.
4. **Energy and Reproduction:** Herbivores gain energy from vegetation and reproduce when above a threshold. Carnivores consume herbivores to maintain energy and reproduce accordingly.
5. **Heat Map Visualization:** The population densities are displayed dynamically, allowing observation of ecosystem trends over time.

## Code Implementation
The main simulation logic is implemented in `main.py`. The key components include:
- **NDVI Generation:** Uses a sinusoidal base pattern with noise and Gaussian smoothing.
- **Agent Movement:** Herbivores seek high NDVI regions, while carnivores target herbivore populations.
- **Energy Dynamics:** Agents gain or lose energy based on consumption and movement.
- **Population Updates:** Agents reproduce or die based on energy levels.
- **Visualization:** Matplotlib is used to render heat maps representing NDVI, herbivore, and carnivore distributions.



## Usage
To run the simulation, execute:
```bash
python main.py
```
This will initialize the simulation and display the evolving ecosystem as a heat map.

