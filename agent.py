import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent

class HumanAgent(Agent):
    def __init__(self, unique_id, model, beta = 0.8):
        super().__init__(unique_id, model)
        self.opinion = np.random.choice([0, 1], p=[0.6, 0.4]) # create a majority and minority opinion
        self.social_isolation_fear = np.random.beta(a=2, b=2)# Random fear of social isolation
        self.confidence_level = random.uniform(0, 1) # random confidence level
        self.is_speaking = False
        self.opinion_similarity_ratio_local = 0.0 # Proportion of neighbors sharing the same opinion
        self.opinion_similarity_ratio_global = 0.0 # Proportion of media messages sharing the same opinion
        self.permanently_silent = False  # Once silent, always silent
        self.beta = beta # weight for global vs local opinion

    def step(self):
        # get public opinion from media
        self.calculate_public_opinion()
        
        # calculate proportion of neighbors with same opinion
        self.calculate_opinion_similarity()
        
        # Once silent, always silent
        if self.permanently_silent:
            self.is_speaking = False
        else:
            if self.should_speak():
                self.is_speaking = True
            else:
                # First time becoming silent - mark as permanently silent
                self.is_speaking = False
                #self.permanently_silent = True
    
    def calculate_opinion_similarity(self):
        """Calculate proportion of local neighbors that share the same opinion."""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            self.opinion_similarity_ratio = 0.0
            return
        spoken_neighbours = [neighbor for neighbor in neighbors if neighbor.is_speaking]
        same_opinion_count = sum(1 for neighbor in spoken_neighbours if neighbor.opinion == self.opinion)
        self.opinion_similarity_ratio_local = same_opinion_count / len(spoken_neighbours) if spoken_neighbours else 1
        # if no neighbours speaking, then we can assume all neighbors share the same opinion, so ratio is 1
    
    def calculate_public_opinion(self):
        """Calculate the proportion of media messages that share the same opinion."""
        media_opinion = self.model.update_media_opinion_default()
        if len(media_opinion) > 20:  # randomly select 20 media messages
            media_opinion = random.sample(media_opinion, 20)
        same_opinion_count = sum(1 for opinion in media_opinion if opinion == self.opinion)
        self.opinion_similarity_ratio_global = same_opinion_count / len(media_opinion) if media_opinion else 1
        # if no media messages, then we can assume all media messages share the same opinion, so ratio is 1
        
    def should_speak(self):
        """
        This function determines the decision-making process of human agents
        """
        # signal of speaking is a weighted combination of local and global perceived opinions
        signal = self.beta * self.opinion_similarity_ratio_global + (1 - self.beta) * self.opinion_similarity_ratio_local
        if signal < self.social_isolation_fear: # speak if this signal is stronger than fear of isolation, or the agent is very confident
            self.confidence_level = max(0, self.confidence_level - 0.025) # decrease confidence if not speaking
            if self.confidence_level > 0.4:
                return True
            #if self.confidence_level < 0.1:
                #self.permanently_silent = True # if confidence is too low, become permanently silent
            return False
        self.confidence_level = min(1.0, self.confidence_level + 0.025)
        return True

class LLMAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = np.random.choice([0, 1], p=[0.6, 0.4])
        self.is_speaking = True

    def step(self):
        # LLM agents always speak and do not change their opinion
        pass

    