import matplotlib.pyplot as plt
import numpy as np
from agent import HumanAgent, LLMAgent
from matplotlib.patches import Patch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Visualization:
    @staticmethod
    def plot_average_silent_ratios_with_ci(opinion_0_runs, opinion_1_runs, confidence=0.95):
        """Plot per-step mean silent ratio with confidence intervals across runs."""
        data_0 = np.asarray(opinion_0_runs, dtype=float)
        data_1 = np.asarray(opinion_1_runs, dtype=float)

        if data_0.ndim != 2 or data_1.ndim != 2:
            raise ValueError("Run data must be a 2D array-like with shape (num_runs, num_steps).")

        if data_0.shape != data_1.shape:
            raise ValueError("Opinion 0 and opinion 1 run data must have the same shape.")

        num_runs, num_steps = data_0.shape
        z_value = 1.96 if np.isclose(confidence, 0.95) else 1.96

        mean_0 = data_0.mean(axis=0)
        mean_1 = data_1.mean(axis=0)

        if num_runs > 1:
            sem_0 = data_0.std(axis=0, ddof=1) / np.sqrt(num_runs)
            sem_1 = data_1.std(axis=0, ddof=1) / np.sqrt(num_runs)
            ci_0 = z_value * sem_0
            ci_1 = z_value * sem_1
        else:
            ci_0 = np.zeros(num_steps)
            ci_1 = np.zeros(num_steps)

        steps = np.arange(num_steps)
        plt.figure(figsize=(12, 6))

        plt.plot(steps, mean_0, label='Mean Silent Ratio Opinion 0', color='blue')
        plt.fill_between(steps, mean_0 - ci_0, mean_0 + ci_0, color='blue', alpha=0.2,
                         label='95% CI Opinion 0')

        plt.plot(steps, mean_1, label='Mean Silent Ratio Opinion 1', color='orange')
        plt.fill_between(steps, mean_1 - ci_1, mean_1 + ci_1, color='orange', alpha=0.2,
                         label='95% CI Opinion 1')
        #plt.axvline(x=25, linestyle="--", color="gray", linewidth=1.5, label="transition of media scheme")

        plt.title(f'Average Silent Ratios Over Time ({num_runs} runs)')
        plt.xlabel('Time Steps')
        plt.ylabel('Silent Ratio')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("output_plots/average_silent_ratios_ci.png", dpi=150, bbox_inches='tight')
        plt.show()

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

    @staticmethod
    def plot_expression_difference_with_ci(spoken_0_runs, spoken_1_runs, real_0_runs, real_1_runs, confidence=0.95):
        from scipy import stats
        
        data = {
            'Real Opinion 0':   np.asarray(real_0_runs,   dtype=float),
            'Spoken Opinion 0': np.asarray(spoken_0_runs, dtype=float),
            'Real Opinion 1':   np.asarray(real_1_runs,   dtype=float),
            'Spoken Opinion 1': np.asarray(spoken_1_runs, dtype=float),
        }
        
        labels = list(data.keys())
        means = [np.mean(v) for v in data.values()]
        n = len(spoken_0_runs)  # number of runs
        
        # 95% CI using t-distribution (robust for small n_runs)
        cis = [stats.t.interval(confidence, df=n-1, loc=np.mean(v), scale=stats.sem(v)) for v in data.values()]
        lower_err = [m - ci[0] for m, ci in zip(means, cis)]
        upper_err = [ci[1] - m for m, ci in zip(means, cis)]
        
        colors = ['#3266AD', '#85B7EB', '#E2A640', '#F5C4B3']
        
        fig, ax = plt.subplots()
        # bars with asymmetric CI
        bars = ax.bar(labels, means, color=colors, width=0.5,
                    yerr=[lower_err, upper_err], capsize=5,
                    error_kw={'elinewidth': 1.5, 'ecolor': '#444441'})
        
        # annotate mean values on top of bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + upper_err[i] + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=9, color='#444441')
        
        ax.set_ylabel('Number of agents')
        ax.set_title(f'Real vs. spoken opinion distribution (mean ± {int(confidence*100)}% CI, n={n} runs)')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("output_plots/difference_expressed_real_opinion_with_ci.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confidence_distribution(confidence_start, confidence_end):
        sns.kdeplot(confidence_start, label='Initial Confidence', fill=True, color='brown', alpha=0.5, clip = (0,1), bw_adjust = 0.5)
        sns.kdeplot(confidence_end, label='Final Confidence', fill=True, color='yellow', alpha=0.5, clip = (0,1), bw_adjust = 0.5)
        plt.title('Distribution of Total Agent Confidence Levels at Start and End of Simulation')
        plt.xlabel('Confidence Level')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("output_plots/confidence_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sensitivity(silence_ratio_0, silence_ratio_1, parameter_name):
        plt.plot(silence_ratio_0.keys(), silence_ratio_0.values(), label = "opinion_0", marker = "o", color = "blue")
        plt.plot(silence_ratio_1.keys(), silence_ratio_1.values(), label = "opinion_1", marker = "o", color = "orange")
        plt.title(f'Sensitivity Analysis of {parameter_name} on Silent Ratios')
        plt.xlabel(parameter_name)
        plt.ylabel('Average Silent Ratio (30 runs each condition)')
        plt.legend()
        plt.grid()
        plt.savefig(f"output_plots/sensitivity_{parameter_name}.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sensitivity_3d(alpha_values, beta_values, silent_0_grid, silent_1_grid):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        alpha_values = np.asarray(alpha_values, dtype=float)
        beta_values = np.asarray(beta_values, dtype=float)
        silent_0_grid = np.asarray(silent_0_grid, dtype=float)
        silent_1_grid = np.asarray(silent_1_grid, dtype=float)
        
        # create the meshgrid for plotting
        Alpha, Beta = np.meshgrid(alpha_values, beta_values)
        
        # Plot surface for opinion 0
        ax.plot_surface(Alpha, Beta, silent_0_grid, color='blue', alpha=0.5, label='Opinion 0')
        # Plot surface for opinion 1
        ax.plot_surface(Alpha, Beta, silent_1_grid, color='orange', alpha=0.5, label='Opinion 1')
        
        ax.set_xlabel('Alpha (media bias strength)')
        ax.set_ylabel('Beta (global vs local weight)')
        ax.set_zlabel('Average Silent Ratio')
        ax.set_title('Sensitivity Analysis of Alpha and Beta on Silent Ratios')
        
        # Create custom legend
        blue_patch = Patch(color='blue', label='Opinion 0')
        orange_patch = Patch(color='orange', label='Opinion 1')
        plt.legend()
        plt.savefig("output_plots/sensitivity_alpha_beta_3d.png", dpi=150, bbox_inches='tight')
        plt.show()


