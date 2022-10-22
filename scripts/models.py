from rcoginta_framework.rcoginta.models import ModelNN

import torch
import numpy as np


class ModelActorMarsLander(ModelNN):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # architecture

    def forward(
        self,
        observation: torch.Tensor
    ) -> torch.Tensor:
        """
            Returns action given observation

            args:
                observation - torch.Tensor, shape = (*, dim_observation)
                              observation tensor
            returns:
                action - torch.Tensor, shape = (*, dim_action)
                         action tensor
        """
            pass


class ModelCriticMarsLander(ModelNN):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # architecture

    def forward(
        self,
        observation_action: torch.Tensor
    ) -> torch.Tensor:
        """
            Returns action given observation

            args:
                observation_action - torch.Tensor, shape = (*, dim_observation + dim_action)
                              observation and action tensor
            returns:
                value - torch.Tensor, shape = (*,)
                        critic value function
        """
            pass

class ModelRunningObjectiveMarsLander:
    def __init__(
        self,
        fuel_consumption_coeff: float,
        angle_constraint_coeff: float
    ) -> None:
        """
        Stores fuel_consumption multiplier to increase acceleration punishment
        and angle constraint multiplier to punish falling upside down
        """
        self.fuel_consumption_coeff = fuel_consumption_coeff
        self.angle_constraint_coeff = angle_constraint_coeff

    def __call__(
        self,
        observation: np.array,
        action: np.array
    ) -> np.array:
    """
        Computes Mars Lander running objective
    """
        distance_cost    = observation[...,0]  # distance to the landing platform
        fuel_consumption = action[...,0]       # fuel constraint
        angle_cost       = observation[..., 2] # angle constraint

        return - distance_cost ** 2 \
               - self.fuel_consumption_coeff * fuel_consumption ** 2 \
               - self.angle_constraint_coeff / (np.cos(angle_cost/2) + 1e-3)
