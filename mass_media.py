import random
import numpy as np
import matplotlib.pyplot as plt
import os
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import HumanAgent, LLMAgent

class SocialMedia(Model):
    def __init__(self, num_agents=100, width=10, height=10):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.media_opinions = []
        self.silent_ratios = {'opinion_0': [], 'opinion_1': []}
        self._media_gap = []  # Changed to private attribute
        self.media_0 = []
        self.media_1 = []
        self.alpha = 0.2
        
        for i in range(self.num_agents):
            if random.random() < 0.9:
                agent = HumanAgent(i, self)
            else:
                agent = LLMAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            while not self.grid.is_cell_empty((x, y)):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"opinion": "opinion", "opinion_similarity_ratio": "opinion_similarity_ratio"},
            model_reporters={"silent_ratio_0": "silent_ratio_0", "silent_ratio_1": "silent_ratio_1", "media_gap": "media_gap", "media_0":"media_0", "media_1":"media_1"}
        )

    def step(self):
        self.schedule.step()
        self.update_media_opinion_weighted()
        self.calculate_silent_ratios()
        self.calculate_media_gap()
        self.datacollector.collect(self)

    def update_media_opinion_default(self):
        """
        This function updates the media opinions based on the current speaking agents in the model.
        """
        opinions = [agent.opinion for agent in self.schedule.agents if agent.is_speaking]
        self.media_opinions = opinions
        return self.media_opinions
    
    def update_media_opinion_balanced(self):
        """
        This function updates the media opinions by balancing the two opinions (0 and 1)
        """
        opinions = [agent.opinion for agent in self.schedule.agents if agent.is_speaking]
        count_0 = sum(1 for opinion in opinions if opinion == 0)
        count_1 = sum(1 for opinion in opinions if opinion == 1)
        if count_0 >= count_1: # balance the two opinions
            self.media_opinions = [0] * count_1 + [1] * count_1 # go within the minimum count of the opinions
        else:
            self.media_opinions = [0] * count_0 + [1] * count_0
        return self.media_opinions
        
    def update_media_opinion_weighted(self):
        """
        This function updates the media opinions by slightly favoring the minority opinion
        """
        opinions = [agent.opinion for agent in self.schedule.agents if agent.is_speaking]
        count_0 = sum(1 for opinion in opinions if opinion == 0)
        count_1 = sum(1 for opinion in opinions if opinion == 1)
        if count_0 >= count_1:
            weighted = int((1 + self.alpha) * count_1)
            self.media_opinions = [0] * (len(opinions) - weighted) + [1] * weighted
        else:
            weighted = int((1 + self.alpha) * count_0)
            self.media_opinions = [0] * weighted + [1] * (len(opinions) - weighted)
        return self.media_opinions

    def calculate_silent_ratios(self):
        silent_count_0 = sum(1 for agent in self.schedule.agents if agent.opinion == 0 and not agent.is_speaking)
        silent_count_1 = sum(1 for agent in self.schedule.agents if agent.opinion == 1 and not agent.is_speaking)
        num_type_0 = sum(1 for agent in self.schedule.agents if agent.opinion == 0)
        num_type_1 = sum(1 for agent in self.schedule.agents if agent.opinion == 1)
        self.silent_ratios['opinion_0'].append(silent_count_0 / num_type_0)
        self.silent_ratios['opinion_1'].append(silent_count_1 / num_type_1)

    def calculate_media_gap(self):
        # count how many agents are speak for 0 and 1 in opinions list
        self.media_message_0 = sum(1 for opinion in self.media_opinions if opinion == 0)
        self.media_message_1 = sum(1 for opinion in self.media_opinions if opinion == 1)
        self.media_0.append(self.media_message_0)
        self.media_1.append(self.media_message_1)
        if len(self.media_opinions) > 1:
            gap = abs(self.media_message_1 - self.media_message_0)
            self._media_gap.append(gap)

    @property
    def silent_ratio_0(self):
        return self.silent_ratios['opinion_0'][-1] if self.silent_ratios['opinion_0'] else 0

    @property
    def silent_ratio_1(self):
        return self.silent_ratios['opinion_1'][-1] if self.silent_ratios['opinion_1'] else 0

    @property
    def media_gap(self):
        return self._media_gap[-1] if self._media_gap else 0  # Accessing the private attribute

    @property
    def media_gap_series(self):
        return self._media_gap