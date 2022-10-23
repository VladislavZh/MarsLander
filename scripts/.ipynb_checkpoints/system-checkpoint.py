from rcognita_framework.rcognita.systems import System

import torch
import numpy as np

from typing import Tuple


class SysMarsLander(System):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.name   = "mars-lander"
        self.width  = 7000
        self.height = 3000

        self.landscape, self.platform = self._generate_landscape(width, height)

    def _generate_landscape(
        self,
        width: float,
        height: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Generates landscape with a landing platform of size 1000m
            Returns landscape points and platform coordinates
        """
        min_h = 0.05 * height
        max_h = 0.90 * height

        platform_height = np.random.random() * (max_h - min_h) + min_h
        platform_left   = np.random.random() * (width - 1000)
        platform_right  = platform_left + 1000

        platform = torch.Tensor([
            platform_left,
            platform_height,
            platform_right,
            platform_height
        ])

        landscape = torch.zeros(12,2)

        # first point
        landscape[0,1] = np.random.random() * (max_h - min_h) + min_h

        # last point
        landscape[-1,0] = width
        landscape[-1,1] = np.random.random() * (max_h - min_h) + min_h

        # platform
        landscape[1,:] = platform[:2]
        landscape[2,:] = platform[2:]

        # other points
        landscape[3:-1,0] = np.random.random() * (width - 1000)
        landscape[3:-1,0] = np.random.random() * (max_h - min_h) + min_h
        landscape[landscape[:,0]>platform_left,0] += 1000
        landscape[-1,0] -= 1000
        landscape[2, 0] -= 1000

        # sorting
        ids = torch.argsort(landscape[:,0])
        landscape = landscape[ids,:]

        return landscape, platform

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        g = self.pars[0]

        Dstate = None# code

        return Dstate

    def out(self, state, time=None, action=None):

        observation = None # code

        return observation

    def reset(self):
        self.landscape, self.platform = self._generate_landscape(width, height)
