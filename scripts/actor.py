from rcognita_framework.rcognita.actors import Actor, ActorProbabilisticEpisodic

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc


from typing import Optional


class ActorMarsLander(Actor):
    """
        TODO: Mars Lander actor model description
    """
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def update(
        self,
        observation: np.array
    ) -> None:
        """
        TODO: Have to store action in self.action for correct controller work
        """
        pass

    def reset(
        self
    ) -> None:
        super().reset()

    def get_action(self):
        return self.action

class ActorMarsLanderAC(Actor):
    """
        TODO: Mars Lander actor model description
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
    '''
    def update(
        self,
        observation: np.array
    ) -> None:
        """
        TODO: Have to store action in self.action for correct controller work
        """
        action_sample = self.model(observation)
        self.action = np.array(np.clip(action_sample, self.action_bounds[0], self.action_bounds[1]))
        self.action_old = self.action

        #current_gradient = self.model.compute_gradient(action_sample)

        #self.store_gradient(current_gradient)
        #pass
        '''
    def update(
        self,
        observation: np.array
    ) -> None:
        self.losses_iter.append(self.objective(observation))

    def reset(
        self
    ) -> None:
        super().reset()
        self.losses_iter = []

    def get_action(self):
        return self.action

    def parameters(self):
        return self.actor.parameters()

    def objective(
        self,
        observation
    ) -> torch.Tensor:
        """
        Objective of the critic, say, a squared temporal difference.

        """
        actor_objective = 0
        improved_value = self.running_objective + self.discount_factor*self.critic(self.actor(observation))
        actor_objective += -improved_value
        return actor_objective

'''
    def objective(
        self,
        action,
        observation
    ) -> torch.Tensor:
        """
        Objective of the critic, say, a squared temporal difference.

        """
        actor_objective = 0
        improved_value = self.running_objective + self.discount_factor* self.critic(next_state)
        return -improved_value.mean()
'''

class ActorProbabilisticEpisodicACMars(ActorProbabilisticEpisodic):
    def update(self, observation):
        #############################################
        # YOUR CODE BELOW
        #############################################
        action_sample = self.model.sample_from_distribution(observation)
        self.action = np.array(np.clip(action_sample, self.action_bounds[0], self.action_bounds[1]))
        self.action_old = self.action

        current_gradient = self.model.compute_gradient(action_sample)

        self.store_gradient(current_gradient)

        #############################################
        # YOUR CODE ABOVE
        #############################################
