import matplotlib.pyplot as plt
import numpy as np
from agent import HumanAgent, LLMAgent
from matplotlib.patches import Patch

class Visualization:
    @staticmethod
    def plot_silent_ratios(model):
        plt.figure(figsize=(12, 6))
        plt.plot(model.silent_ratios['opinion_0'], label='Silent Ratio Opinion 0', color='blue')
        plt.plot(model.silent_ratios['opinion_1'], label='Silent Ratio Opinion 1', color='orange')
        plt.title('Silent Ratios Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Silent Ratio')
        plt.legend()
        plt.grid()
        plt.savefig("output_plots/silent_ratios.png")
        plt.show()

    @staticmethod
    def plot_media_gap(model):
        plt.figure(figsize=(12, 6))
        plt.plot(model.media_0, label='Media Opinion 0 Count', color='blue')
        plt.plot(model.media_1, label='Media Opinion 1 Count', color='red')
        plt.plot(model.media_gap_series, label='Media Gap', color='green')
        plt.title('Media Gap Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Media Gap')
        plt.legend()
        plt.grid()
        plt.savefig("output_plots/media_gap.png")
        plt.show()
    
    @staticmethod
    def plot_grid(model, title="Grid State"):
        """Create a grid visualization showing agent type and opinion with different colors.
        - Dark Blue: Human with opinion 0
        - Dark Red: Human with opinion 1
        - Light Blue: LLM with opinion 0
        - Light Red/Pink: LLM with opinion 1
        - Gray: Empty cells
        """
        grid_data = np.zeros((model.grid.height, model.grid.width, 3))
        # Initialize with gray for vacant cells
        grid_data[:, :] = [0.8, 0.8, 0.8]  # Gray
        
        for agent in model.schedule.agents:
            x, y = agent.pos
            if isinstance(agent, HumanAgent):
                if agent.opinion == 0:
                    grid_data[y, x] = [0, 0, 0.8]  # Dark Blue for Human opinion 0
                else:
                    grid_data[y, x] = [0.8, 0, 0]  # Dark Red for Human opinion 1
            elif isinstance(agent, LLMAgent):
                if agent.opinion == 0:
                    grid_data[y, x] = [0.5, 0.7, 1]  # Light Blue for LLM opinion 0
                else:
                    grid_data[y, x] = [1, 0.5, 0.5]  # Pink/Light Red for LLM opinion 1
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_data)
        plt.title(title)
        plt.axis('off')
        
        # Add legend using custom patches
        legend_elements = [
            Patch(facecolor=[0, 0, 0.8], label='Human - Opinion 0'),
            Patch(facecolor=[0.8, 0, 0], label='Human - Opinion 1'),
            Patch(facecolor=[0.5, 0.7, 1], label='LLM - Opinion 0'),
            Patch(facecolor=[1, 0.5, 0.5], label='LLM - Opinion 1'),
            Patch(facecolor=[0.8, 0.8, 0.8], label='Empty')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_agent_grid(model, title="Agent Distribution on Grid"):
        plt.figure(figsize=(12, 10))
        
        # Separate agents by type and opinion
        human_opinion_0 = []
        human_opinion_1 = []
        llm_opinion_0 = []
        llm_opinion_1 = []
        
        for agent in model.schedule.agents:
            x, y = agent.pos
            if isinstance(agent, HumanAgent):
                if agent.opinion == 0:
                    human_opinion_0.append((x, y))
                else:
                    human_opinion_1.append((x, y))
            elif isinstance(agent, LLMAgent):
                if agent.opinion == 0:
                    llm_opinion_0.append((x, y))
                else:
                    llm_opinion_1.append((x, y))
        
        # Plot each group with different colors and markers
        if human_opinion_0:
            x_coords, y_coords = zip(*human_opinion_0)
            plt.scatter(x_coords, y_coords, c='blue', marker='o', s=100, 
                       label='Human - Opinion 0', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if human_opinion_1:
            x_coords, y_coords = zip(*human_opinion_1)
            plt.scatter(x_coords, y_coords, c='red', marker='o', s=100, 
                       label='Human - Opinion 1', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if llm_opinion_0:
            x_coords, y_coords = zip(*llm_opinion_0)
            plt.scatter(x_coords, y_coords, c='blue', marker='s', s=100, 
                       label='LLM - Opinion 0', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        if llm_opinion_1:
            x_coords, y_coords = zip(*llm_opinion_1)
            plt.scatter(x_coords, y_coords, c='red', marker='s', s=100, 
                       label='LLM - Opinion 1', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        plt.xlim(-1, model.grid.width)
        plt.ylim(-1, model.grid.height)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output_plots/agent_grid_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def difference_expressed_real_opinion(model):
        """
        This function compares the real opinion distribution int he population with the spoken opinions at the end of simualtion
        """
        real_opinion_0 = sum(1 for agent in model.schedule.agents if agent.opinion == 0)
        real_opinion_1 = sum(1 for agent in model.schedule.agents if agent.opinion == 1)
        spoken_opinion_0 = sum(1 for agent in model.schedule.agents if (agent.opinion == 0) and agent.is_speaking) 
        spoken_opinion_1 = sum(1 for agent in model.schedule.agents if (agent.opinion == 1) and agent.is_speaking) 

        plt.bar(['Real Opinion 0', 'Spoken Opinion 0', 'Real Opinion 1', 'Spoken Opinion 1'], 
                [real_opinion_0, spoken_opinion_0, real_opinion_1, spoken_opinion_1],
                color=['blue', 'lightblue', 'red', 'lightcoral'])
        plt.title('Comparison of Real Opinion Distribution and Spoken Opinions')
        plt.ylabel('Number of Agents')
        plt.grid()
        plt.savefig("output_plots/difference_expressed_real_opinion.png", dpi=150, bbox_inches='tight')
        plt.show()
