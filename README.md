# Spiral of Silence in Hybrid Human–AI Social Networks

This project implements an **agent-based model (ABM)** to study the dynamics of the *spiral of silence* in online social networks where **human users and large language model (LLM) agents coexist**.We investigate how **perceived majority opinion** shapes individuals’ willingness to express or withhold their views, and how **media aggregation mechanisms** can either amplify or mitigate silence dynamics.


## Research Overview

The model simulates a social environment with two types of agents:
- **Human agents**
  - Sensitive to perceived majority opinion
  - May choose to remain silent when holding minority views
  - Decision to keet silence or not is determined by perceived public opinion, local neighbour opinion, social isolation fear and individual confidence level

- **LLM agents (bots)**
  - Act as **stubborn opinion holders**
  - Always express their opinions regardless of social pressure

## Media Mechanisms
We compare two types of mass media interventions:
### 1. Neutral Media (No Gatekeeping)
- Publishes all collected messages without filtering
- Reflects the raw distribution of expressed opinions

### 2. Bias-Aware Media (Minority Amplification)
- Applies additional weight to minority opinions before publishing
- Aims to counteract dominance of majority voices

## Research Questions
- How do stubborn LLM agents affect silence dynamics?
- Can media interventions reduce the spiral-of-silence effect?
- What are the system-level consequences of different media paradigms?

## Repository Structure
### File Descriptions
- **`agent.py`**  
  Defines agent classes (human and LLM) and their state update rules at each timestep.
- **`mass_media.py`**  
  Implements the two media paradigms and the overall simulation environment.
- **`runfile.py`**  
  Entry point for running simulations.  
  Configure model parameters here and generate output plots.
- **`visualization.py`**  
  Contains functions for plotting simulation results.
- **`parameter_sensitivity_analysis.py`**  
  Performs sensitivity analysis on key parameters:
  - `alpha`: media bias weight (minority amplification)
  - `beta`: weight of global opinion influence
- **`output_plots/`**  
  Stores all generated figures.
  
## How to Run the Model
Run **`runfile.py`** for simulation outputs, and run **`parameter_sensitivity_analysis.py`** for sensitivity analysis output

## Key Insight

## Potential Future Work
- Set the model in a network structure and compare effects of different network typologies (e.g., scale-free vs. random network)
- LLM agents will be designed to be more adpative and strategic
- Media algorithms can be more personalized (e.g., personal recommender system)
- Calibrate the model with empirical data

## Author
Xuening Tang<br>
MSc Computational Science, University of Amsterdam <br>
Contact: xuening.tang@student.uva.nl
  

