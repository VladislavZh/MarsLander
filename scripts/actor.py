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
        in_bound,
        *args,
        **kwargs
    ) -> None:
        self.state_to_observation = state_to_observation
        self.in_bound = in_bound
        super().__init__(*args, **kwargs)

    def objective(
        self,
        action,
        observation
    ) -> torch.Tensor:
        action_sequence_reshaped = action.unsqueeze(0)

        state_sequence = [observation[-5:]]

        observation_sequence_predicted = self.predictor.predict_sequence(
            torch.Tensor(observation[-5:]), action_sequence_reshaped
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
        if self.in_bound(observation[-5:-3]):
            action = self.model(observation)
            loss = self.objective(action, observation)
            loss = action @ action
            loss.backward()
            self.action = action.detach().cpu().numpy()
            self.action_old = self.action
            self.action[0] = np.clip(self.action[0], a_min=self.action_min[0], a_max=self.action_max[0])
            self.action[1] = np.clip(self.action[1], a_min=self.action_min[1], a_max=self.action_max[1])
        else:
            self.action = np.array([0,0])
            self.action_old = self.action

    def reset(
        self
    ) -> None:
        super().reset()

    def get_action(self):
        return self.action
