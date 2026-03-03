import random
import numpy as np
import matplotlib.pyplot as plt
import os
from mass_media import SocialMedia
from agent import HumanAgent, LLMAgent
from visualization import Visualization

if __name__ == "__main__":
    if not os.path.exists("output_plots"):
        os.makedirs("output_plots")
    
    model = SocialMedia(num_agents=400, width=20, height=20)
    for i in range(300):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    print(f"Collected model rows: {len(model_df)}")
    print(f"Collected agent rows: {len(agent_df)}")

    # print the number of agents with opinion 0 and 1, and the number of human agents and LLM agents
    num_human_opinion_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 0)
    num_human_opinion_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, HumanAgent) and agent.opinion == 1)
    num_llm_opinion_0 = sum(1 for agent in model.schedule.agents if isinstance(agent, LLMAgent) and agent.opinion == 0)
    num_llm_opinion_1 = sum(1 for agent in model.schedule.agents if isinstance(agent, LLMAgent) and agent.opinion == 1)
    print(f"Number of Human Agents with Opinion 0: {num_human_opinion_0}")
    print(f"Number of Human Agents with Opinion 1: {num_human_opinion_1}")
    print(f"Number of LLM Agents with Opinion 0: {num_llm_opinion_0}")
    print(f"Number of LLM Agents with Opinion 1: {num_llm_opinion_1}")
    
    Visualization.plot_silent_ratios(model)
    Visualization.plot_media_gap(model)
    #Visualization.plot_agent_grid(model, title="Agent Distribution on Grid (Final State)")
    Visualization.difference_expressed_real_opinion(model)