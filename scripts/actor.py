from rcognita_framework.rcognita.actors import Actor

import numpy as np
import torch
from rcognita_framework.rcognita.utilities import rc


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
