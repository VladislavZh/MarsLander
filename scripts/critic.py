from rcoginta_framework.rcoginta.critics import Critic

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc

from typing import Optional


class CriticMarsLander(Critic):
    """
        TODO: Mars Lander critic model description
    """
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
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

        # TODO: code
        
        return critic_objective
