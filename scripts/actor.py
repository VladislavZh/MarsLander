from rcognita_framework.rcognita.actors import Actor, ActorProbabilisticEpisodic

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc


from typing import Optional


class ActorMarsLander(Actor):
    """
        Mars Lander actor model
    """
    def __init__(
        self,
        state_to_observation,
        *args,
        **kwargs
    ) -> None:
        self.state_to_observation = state_to_observation
        super().__init__(*args, **kwargs)

    def objective(
        self,
        action,
        observation
    ) -> torch.Tensor:
        action_sequence_reshaped = rc.reshape(action, [1, self.dim_input])

        state_sequence = [observation[-5:]]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation[-5:], action_sequence_reshaped
        )

        observation_sequence = rc.vstack(
            (
                rc.reshape(rc.array(observation, prototype=action), [1, self.dim_output]),
                observation_sequence_predicted,
            )
        )
        actor_objective = self.running_objective(
            observation_sequence[0, :], action_sequence_reshaped
        ) + self.discount_factor * self.critic(
            observation_sequence[1, :], self.model(observation_sequence[1, :], use_stored_weights=True), use_stored_weights=True
        )

        return - actor_objective

    def update(
        self,
        observation
    ) -> None:
        loss = self.objective(self.model(observation), observation)
        loss.backward()

    def reset(
        self
    ) -> None:
        super().reset()

    def get_action(self):
        return self.action
