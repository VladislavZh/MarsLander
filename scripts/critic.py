from rcognita_framework.rcognita.critics import Critic

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc

from typing import Optional


from rcognita_framework.rcognita.critics import Critic

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc

from typing import Optional


class CriticMarsLander(Critic):
    """
        Mars Lander critic
    """
    def __init__(
        self,
        in_bound,
        *args,
        **kwargs
    ) -> None:
        self.in_bound = in_bound
        super().__init__(*args, **kwargs)

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

        critic_objective = 0

        for i in range(self.data_buffer_size - 1, 0, -1):
            x_cur = observation_buffer[i-1, :]
            x_next = observation_buffer[i, :]
            u_cur = action_buffer[i-1, :]
            u_next = action_buffer[i, :]

            #TD
            critic_cur = self.model(x_cur, u_cur, weights=weights)
            critic_next = self.model(x_next, u_next, use_stored_weights=True)
            reward = self.running_objective(x_cur, u_cur)

            TD = (critic_cur - self.discount_factor * critic_next - reward)

            if self.in_bound(observation_buffer[i-1,-5:-3]):
                critic_objective += 1/2*TD**2
            else:
                critic_objective += 0*TD**2

        return critic_objective

    def update(self, constraint_functions=(), time=None):
        return
