from rcognita_framework.rcognita.models import ModelNN

import torch
import numpy as np
import torch.nn as nn

class ModelActorMarsLander(ModelNN):
    def __init__(
        self,
        dim_observation,
        dim_action,
        *args,
        weights=None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # architecture
        self.net = nn.Sequential(
            nn.Linear(dim_observation, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, dim_action)
        )

        if weights is not None:
            self.load_state_dict(weights)
        self.double()
        self.cache_weights()

    def forward(
        self,
        observation: torch.Tensor,
        weights = None
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
        #pass
        if weights is not None:
            self.update(weights)
        output = self.net(observation)
        tmp = output.clone()
        tmp[0] = torch.nn.functional.softplus(output[0])
        tmp[1] = torch.nn.functional.sigmoid(output[1])
        return tmp


class ModelCriticMarsLander(ModelNN):
    def __init__(self,
                 dim_observation,
                 dim_action,
                 *args,
                 weights = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # architecture
        self.net = nn.Sequential(
            nn.Linear(dim_observation+dim_action, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 1)
        )

        if weights is not None:
            self.load_state_dict(weights)
        self.double()
        self.cache_weights()

    def forward(
        self,
        observation_action: torch.Tensor,
        weights = None
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
        #pass
        if weights is not None:
            self.update(weights)

        output = self.net(observation_action)

        return output


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
