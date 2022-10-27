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

    def reset(
        self
    ) -> None:
        super().reset()

    def get_action(self):
        return self.action

    def parameters(self):
        return self.actor.parameters()

    def objective(
        self,
        data_buffer: Optional[np.array] = None,
        weights: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Objective of the critic, say, a squared temporal difference.

        """
        if data_buffer is None:
            observation_buffer = self.observation_buffer
            action_buffer = self.action_buffer
        else:
            observation_buffer = data_buffer["observation_buffer"]
            action_buffer = data_buffer["action_buffer"]

        actor_objective = 0

        # TODO: code
        for i in range(self.data_buffer_size - 1, 0, -1):
            #x_cur = observation_buffer[i-1, :]
            x_next = observation_buffer[i, :]
            #u_cur = action_buffer[i-1, :]
            #u_next = action_buffer[i, :]

            improved_value = self.running_objective + self.discount_factor*self.critic(x_next)
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
