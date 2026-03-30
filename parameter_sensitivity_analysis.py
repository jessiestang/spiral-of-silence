# conduct parameter sensitivity analysis for alpha and beta
import numpy as np
import matplotlib.pyplot as plt
from mass_media import SocialMedia
from agent import HumanAgent, LLMAgent
from visualization import Visualization

def run_sensitivity_analysis_alpha(num_agents, width, height, beta):
    alpha_values = np.arange(0.1, 1.0, 0.1) # range of alpha parameter
    silence_ratio_0 = {}
    silence_ratio_1 = {}

    for alpha in alpha_values:
        proportion_0 = []
        proportion_1 = []
        print(f"Running sensitivity analysis for alpha={alpha:.1f}")
        for _ in range(30):  # Number of iterations
            model = SocialMedia(num_agents=num_agents, width=width, height=height, alpha=alpha, beta=beta)
            for _ in range(100):  # Number of time steps
                model.step_weighted_media()
            
            # Collect the proportion of silent agents
            silence_count_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 0)
            silence_count_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 1)
            total_agent_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 0)
            total_agent_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 1)

            silence_proportion_0 = silence_count_0 / total_agent_0 if total_agent_0 > 0 else 0
            silence_proportion_1 = silence_count_1 / total_agent_1 if total_agent_1 > 0 else 0
            proportion_0.append(silence_proportion_0)
            proportion_1.append(silence_proportion_1)
        
        silence_ratio_0[alpha] = np.mean(proportion_0)
        silence_ratio_1[alpha] = np.mean(proportion_1)


    return silence_ratio_0, silence_ratio_1

def run_sensitivity_analysis_beta(num_agents, width, height, alpha):
    beta_values = np.arange(0.1, 1.0, 0.1) # range of beta parameter
    silence_ratio_0 = {}
    silence_ratio_1 = {}

    for beta in beta_values:
        proportion_0 = []
        proportion_1 = []
        print(f"Running sensitivity analysis for beta={beta:.1f}")
        for _ in range(30):  # Number of iterations
            model = SocialMedia(num_agents=num_agents, width=width, height=height, alpha=alpha, beta=beta)
            for _ in range(100):  # Number of time steps
                model.step_weighted_media()
            
            # Collect the proportion of silent agents
            silence_count_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 0)
            silence_count_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 1)
            total_agent_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 0)
            total_agent_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 1)

            silence_proportion_0 = silence_count_0 / total_agent_0 if total_agent_0 > 0 else 0
            silence_proportion_1 = silence_count_1 / total_agent_1 if total_agent_1 > 0 else 0
            proportion_0.append(silence_proportion_0)
            proportion_1.append(silence_proportion_1)
        
        silence_ratio_0[beta] = np.mean(proportion_0)
        silence_ratio_1[beta] = np.mean(proportion_1)


    return silence_ratio_0, silence_ratio_1

def run_sensitivity_analysis_alpha_beta_grid(num_agents, width, height, alpha_values, beta_values):
    alpha_values = np.asarray(alpha_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)
    silent_0_grid = np.zeros((len(beta_values), len(alpha_values)))
    silent_1_grid = np.zeros((len(beta_values), len(alpha_values)))

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            proportion_0 = []
            proportion_1 = []
            print(f"Running 3D sensitivity for alpha={alpha:.1f}, beta={beta:.1f}")

            for _ in range(30):
                model = SocialMedia(num_agents=num_agents, width=width, height=height, alpha=alpha, beta=beta)
                for _ in range(100):
                    model.step_weighted_media()

                silence_count_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 0)
                silence_count_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and not agent.is_speaking and agent.opinion == 1)
                total_agent_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 0)
                total_agent_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 1)

                proportion_0.append(silence_count_0 / total_agent_0 if total_agent_0 > 0 else 0)
                proportion_1.append(silence_count_1 / total_agent_1 if total_agent_1 > 0 else 0)

            silent_0_grid[i, j] = np.mean(proportion_0)
            silent_1_grid[i, j] = np.mean(proportion_1)

    return silent_0_grid, silent_1_grid

if __name__ == "__main__":

    # parameters
    NUM_AGENTS = 100
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    ALPHA = 0.4
    BETA = 0.6
    alpha_values = np.arange(0.1, 1.0, 0.1)
    beta_values = np.arange(0.1, 1.0, 0.1)

    # Run sensitivity analysis for alpha
    """silence_ratio_0_alpha, silence_ratio_1_alpha = run_sensitivity_analysis_alpha(num_agents = NUM_AGENTS, width = GRID_WIDTH, height = GRID_HEIGHT, beta = BETA)
    Visualization.plot_sensitivity(silence_ratio_0_alpha, silence_ratio_1_alpha, parameter_name="alpha")

    # Run sensitivity analysis for beta
    silence_ratio_0_beta, silence_ratio_1_beta = run_sensitivity_analysis_beta(num_agents = NUM_AGENTS, width = GRID_WIDTH, height = GRID_HEIGHT, alpha = ALPHA)
    Visualization.plot_sensitivity(silence_ratio_0_beta, silence_ratio_1_beta, parameter_name="beta")"""

    # Run joint alpha-beta sweep and plot 3D sensitivity surfaces
    silent_0_grid, silent_1_grid = run_sensitivity_analysis_alpha_beta_grid(
        num_agents=NUM_AGENTS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        alpha_values=alpha_values,
        beta_values=beta_values
    )
    Visualization.plot_sensitivity_3d(alpha_values, beta_values, silent_0_grid, silent_1_grid)
