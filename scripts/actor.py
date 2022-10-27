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
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def objective(
        self,
        observation
    ) -> torch.Tensor:
        action_sequence_reshaped = rc.reshape(action, [1, self.dim_input])

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.vstack(
            (
                rc.reshape(observation, [1, self.dim_output]),
                observation_sequence_predicted,
            )
        )

        actor_objective = self.running_objective(
            observation_sequence[0, :], action_sequence_reshaped
        ) + self.discount_factor * self.critic(
            observation_sequence[1, :], use_stored_weights=True
        )

        return actor_objective

    def reset(
        self
    ) -> None:
        super().reset()

    def get_action(self):
        return self.action
