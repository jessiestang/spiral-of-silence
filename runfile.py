import numpy as np
import os
from mass_media import SocialMedia
from agent import HumanAgent, LLMAgent
from visualization import Visualization


def run_multiple_simulations(num_runs, num_steps, num_agents, width, height, alpha, beta):
    """Run repeated simulations and return per-step silent ratios for each run."""
    opinion_0_runs = []
    opinion_1_runs = []
    real_1_runs = []
    real_0_runs = []
    spoken_1_runs = []
    spoken_0_runs = []

    for run_idx in range(num_runs):
        model = SocialMedia(num_agents=num_agents, width=width, height=height, alpha=alpha, beta=beta)
        for _ in range(num_steps):
            model.step_weighted_media()

        opinion_0_series = np.asarray(model.silent_ratios['opinion_0'], dtype=float)
        opinion_1_series = np.asarray(model.silent_ratios['opinion_1'], dtype=float)

        if len(opinion_0_series) != num_steps or len(opinion_1_series) != num_steps:
            raise ValueError(
                f"Run {run_idx} generated incomplete series: "
                f"opinion_0={len(opinion_0_series)}, opinion_1={len(opinion_1_series)}, expected={num_steps}."
            )

        opinion_0_runs.append(opinion_0_series)
        opinion_1_runs.append(opinion_1_series)

        agents = model.schedule.agents
        real_0_runs.append(sum(1 for a in agents if a.opinion == 0))
        real_1_runs.append(sum(1 for a in agents if a.opinion == 1))
        spoken_0_runs.append(sum(1 for a in agents if a.opinion == 0 and a.is_speaking))
        spoken_1_runs.append(sum(1 for a in agents if a.opinion == 1 and a.is_speaking))

        print(f"Completed run {run_idx + 1}/{num_runs}")

    return np.asarray(opinion_0_runs), np.asarray(opinion_1_runs), np.asarray(real_1_runs), np.asarray(real_0_runs), np.asarray(spoken_0_runs), np.asarray(spoken_1_runs)


def summarize_final_state(num_agents, width, height, num_steps, alpha, beta):
    """Run one simulation and return model plus start/end human confidence arrays."""
    model = SocialMedia(num_agents=num_agents, width=width, height=height, alpha=alpha, beta=beta)
    initial_confidence = model.get_human_confidence_distribution().copy()

    for _ in range(num_steps):
        model.step_weighted_media()

    final_confidence = model.get_human_confidence_distribution().copy()
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    print(f"Collected model rows: {len(model_df)}")
    print(f"Collected agent rows: {len(agent_df)}")
    print(f"Human confidence count at start: {len(initial_confidence)}")
    print(f"Human confidence count at end: {len(final_confidence)}")

    num_human_opinion_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 0)
    num_human_opinion_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 1)
    num_llm_opinion_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, LLMAgent) and agent.opinion == 0)
    num_llm_opinion_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, LLMAgent) and agent.opinion == 1)

    print(f"Number of Human Agents with Opinion 0: {num_human_opinion_0}")
    print(f"Number of Human Agents with Opinion 1: {num_human_opinion_1}")
    print(f"Number of LLM Agents with Opinion 0: {num_llm_opinion_0}")
    print(f"Number of LLM Agents with Opinion 1: {num_llm_opinion_1}")

    return model, initial_confidence, final_confidence

def run_delayed_intervention(num_runs, num_steps, num_agents, width, height, alpha, beta):
    """Run repeated simulations and return per-step silent ratios for each run."""
    opinion_0_runs = []
    opinion_1_runs = []
    real_1_runs = []
    real_0_runs = []
    spoken_1_runs = []
    spoken_0_runs = []
    

    for run_idx in range(num_runs):
        model = SocialMedia(num_agents=num_agents, width=width, height=height)
        for step in range(num_steps):
            if step <= 25:
                model.step_weighted_media() # weight intervention in the first 25 timesteps
            else:
                model.step_no_intervention() # no intervention after 25 timesteps

        opinion_0_series = np.asarray(model.silent_ratios['opinion_0'], dtype=float)
        opinion_1_series = np.asarray(model.silent_ratios['opinion_1'], dtype=float)

        if len(opinion_0_series) != num_steps or len(opinion_1_series) != num_steps:
            raise ValueError(
                f"Run {run_idx} generated incomplete series: "
                f"opinion_0={len(opinion_0_series)}, opinion_1={len(opinion_1_series)}, expected={num_steps}."
            )

        opinion_0_runs.append(opinion_0_series)
        opinion_1_runs.append(opinion_1_series)

        agents = model.schedule.agents
        real_0_runs.append(sum(1 for a in agents if a.opinion == 0))
        real_1_runs.append(sum(1 for a in agents if a.opinion == 1))
        spoken_0_runs.append(sum(1 for a in agents if a.opinion == 0 and a.is_speaking))
        spoken_1_runs.append(sum(1 for a in agents if a.opinion == 1 and a.is_speaking))

        print(f"Completed run {run_idx + 1}/{num_runs}")

    return np.asarray(opinion_0_runs), np.asarray(opinion_1_runs), np.asarray(real_1_runs), np.asarray(real_0_runs), np.asarray(spoken_0_runs), np.asarray(spoken_1_runs)

if __name__ == "__main__":
    if not os.path.exists("output_plots"):
        os.makedirs("output_plots")

    NUM_RUNS = 10
    NUM_STEPS = 100
    NUM_AGENTS = 100
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    ALPHA = 0.8
    BETA = 0.8


    print("initialization finished, start running the model")

    opinion_0_runs, opinion_1_runs, real_1_runs, real_0_runs, spoken_0_runs, spoken_1_runs = run_multiple_simulations(
        num_runs=NUM_RUNS,
        num_steps=NUM_STEPS,
        num_agents=NUM_AGENTS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        alpha=ALPHA,
        beta=BETA,
    )

    print("Data collection finished, start plotting")

    Visualization.plot_average_silent_ratios_with_ci(opinion_0_runs, opinion_1_runs, confidence=0.95)
    Visualization.plot_expression_difference_with_ci(spoken_0_runs,spoken_1_runs, real_0_runs, real_1_runs)

    # Optional single-run diagnostics and legacy plots.
    model, initial_confidence, final_confidence = summarize_final_state(
        num_agents=NUM_AGENTS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_steps=NUM_STEPS,
        alpha=ALPHA,
        beta=BETA,

    )
    print(f"Single-run confidence captured: start mean={np.mean(initial_confidence):.3f}, end mean={np.mean(final_confidence):.3f}")
    Visualization.plot_confidence_distribution(initial_confidence, final_confidence)
    #Visualization.plot_silent_ratios(model)
    #Visualization.plot_media_gap(model)
    #Visualization.difference_expressed_real_opinion(model)