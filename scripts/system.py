from rcognita_framework.rcognita.systems import System
from rcognita_framework.rcognita.utilities import rc

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
        self.random_landscape = False
        self.random_initital_point = False #True

        self.landscape, self.platform = self._generate_landscape(
            self.width,
            self.height
        )

    def _generate_landscape(
        self,
        width: float,
        height: float
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
            Generates landscape with a landing platform of size 1000m
            Returns landscape points and platform coordinates
        """
        min_h = 0.05 * height
        max_h = 0.85 * height

        platform_height = torch.rand(1)[0] * (max_h - min_h) + min_h
        platform_left   = torch.rand(1)[0] * (width - 1000)
        platform_right  = platform_left + 1000

        platform = torch.tensor([
            platform_left,
            platform_height,
            platform_right,
            platform_height
        ])

        landscape = torch.zeros((12,2))

        # first point
        landscape[0,1] = torch.rand(1)[0] * (max_h - min_h) + min_h

        # last point
        landscape[-1,0] = width
        landscape[-1,1] = torch.rand(1)[0] * (max_h - min_h) + min_h

        # platform
        landscape[1,:] = platform[:2]
        landscape[2,:] = platform[2:]

        # other points
        landscape[3:-1,0] = torch.rand(8) * (width - 1000)
        landscape[3:-1,1] = torch.rand(8) * (max_h - min_h) + min_h
        landscape[landscape[:,0]>platform_left,0] += 1000
        landscape[-1,0] -= 1000
        landscape[2, 0] -= 1000

        # sorting
        ids = torch.argsort(landscape[:,0])
        landscape = landscape[ids,:]

        self.__state_init = None

        return landscape, platform

    def _compute_ray_intersection(
        self,
        state: torch.tensor,
        angle: float
    ) -> float:
        k = np.tan(angle)
        b = state[1] - k * state[0]
        intersections = []

        # borders
        if angle > np.pi/2 and angle < 3*np.pi/2:
            intersections.append(torch.tensor([0,b]))
        if k!=0 and abs(k) < 100 and angle < np.pi:
            x = (self.height - b)/k
            if (x < state[0] and angle > np.pi/2 and angle < 3*np.pi/2) or \
               (x > state[0] and (angle < np.pi/2 or angle > 3*np.pi/2)):
                intersections.append(torch.tensor([(self.height - b)/k,self.height]))
        elif abs(k) > 100 and angle<np.pi:
            intersections.append(torch.tensor([state[0],self.height]))
        if angle < np.pi/2 or angle > 3*np.pi/2:
            intersections.append(torch.tensor([self.width,k * self.width + b]))

        # landscape
        for i in range(self.landscape.shape[0]-1):
            part = self.landscape[i:i+2,:]
            k_part = (part[1,1] - part[0,1])/(part[1,0] - part[0,0])
            b_part = part[0,1] - (part[1,1] - part[0,1])/(part[1,0] - part[0,0]) * part[0,0]
            if k_part == k:
                continue
            if abs(k) < 100:
                x = (b - b_part)/(k_part - k)
                y = k * x + b
                if x <= part[1,0] and x >= part[0,0]:
                    if (x < state[0] and angle > np.pi/2 and angle < 3*np.pi/2) or \
                       (x > state[0] and (angle < np.pi/2 or angle > 3*np.pi/2)):
                       intersections.append(torch.tensor([x,y]))
            else:
                if angle > np.pi:
                    x = state[0]
                    y = k_part * x + b_part
                    if x <= part[1,0] and x >= part[0,0]:
                        intersections.append(torch.tensor([x,y]))

        if len(intersections) == 0:
            return -1
        else:
            rs = torch.tensor([(state[0] - i[0]) ** 2 + (state[1] - i[1]) for i in intersections])
            ids = torch.argsort(rs)
            tmp = intersections[ids[0]]
            r = rc.sqrt((tmp[0] - state[0]) ** 2 + (tmp[1] - state[1]) ** 2)
            return r

    @staticmethod
    def get_radial(dx,dy):
        r   = rc.sqrt(dx**2 + dy**2)
        sin_psi = dx/r
        if type(sin_psi) == torch.tensor:
            psi = torch.arcsin(sin_psi)
        else:
            psi = np.arcsin(sin_psi)
        if dx > 0 and dy < 0:
            psi = np.pi - psi
        elif dx < 0 and dy < 0:
            psi = - (np.pi + psi)
        elif dx == 0 and dy < 0:
            psi = np.pi

        return [r,psi]


    def _closest_point(self, state, direction):
        closest_points = []
        # left border
        if direction == 'left':
            x1 = 0
            y1 = state[1]
            closest_points.append(self.get_radial(x1 - state[0], y1 - state[1]))
        if direction == 'right':
            x1 = self.width
            y1 = state[1]
            closest_points.append(self.get_radial(x1 - state[0], y1 - state[1]))

        # landscape
        for i in range(self.landscape.shape[0]-1):
            part = self.landscape[i:i+2,:].clone()
            k_part = (part[1,1] - part[0,1])/(part[1,0] - part[0,0])
            b_part = part[0,1] - (part[1,1] - part[0,1])/(part[1,0] - part[0,0]) * part[0,0]
            if direction == 'left' and part[0,0] > state[0]:
                break
            elif direction == 'left' and part[1,0] > state[0]:
                part[1,0] = state[0]
                part[1,1] = k_part * state[0] + b_part
            elif direction == 'right' and part[1,0] < state[0]:
                continue
            elif direction == 'right' and part[0,0] < state[0]:
                part[0,0] = state[0]
                part[0,1] = k_part * state[0] + b_part

            if k_part == 0:
                x1 = state[0]
                y1 = part[0,1]
                if x1 <= part[1,0] and x1 >= part[0,0]:
                    closest_points.append(self.get_radial(x1 - state[0], y1 - state[1]))
                else:
                    left  = self.get_radial(part[0,0] - state[0], part[0,1] - state[1])
                    right = self.get_radial(part[1,0] - state[0], part[1,1] - state[1])
                    if left[0] < right[0]:
                        closest_points.append(left)
                    else:
                        closest_points.append(right)
            else:
                k = -1/k_part
                b = state[1] - k * state[0]

                x1 = (b_part - b)/(k - k_part)
                if x1 <= part[1,0] and x1 >= part[0,0]:
                    y1 = k * x1 + b
                    closest_points.append(self.get_radial(x1 - state[0], y1 - state[1]))
                else:
                    left  = self.get_radial(part[0,0] - state[0], part[0,1] - state[1])
                    right = self.get_radial(part[1,0] - state[0], part[1,1] - state[1])
                    if left[0] < right[0]:
                        closest_points.append(left)
                    else:
                        closest_points.append(right)
        closest_points = torch.tensor(closest_points)
        ids = torch.argsort(closest_points[:,0])
        return closest_points[ids[0],:]

    def _in_borders(self, state):
        if state[0] < 0:
            return False
        if state[0] > self.width:
            return False
        if state[1] > self.height:
            return False
        for i in range(self.landscape.shape[0]-1):
            part = self.landscape[i:i+2,:].clone()
            if state[0] >= part[0,0] and state[0] <= part[1,0]:
                dx   = state[0] - part[0,0]
                dy   = state[1] - part[0,1]
                dx_l = part[1,0] - part[0,0]
                dy_l = part[1,1] - part[0,1]
                if dx_l * dy - dy_l * dx < 0:
                    return False
        return True

    def _generate_state_init(self):
        while True:
            x, y = torch.rand(1)[0] * 7000, torch.rand(1)[0] * 3000
            if self._in_borders(torch.tensor([x,y])):
                self.__state_init = torch.tensor([x,y,0,0,0])
                break

    @property
    def state_init(self):
        if self.__state_init is None:
            self._generate_state_init()
        return self.__state_init

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        g = self.pars[0]

        Dstate = rc.zeros(5)
        if self._in_borders(state):
            Dstate[0] = state[3]  # dx/dt
            Dstate[1] = state[4]  # dy/dt
            Dstate[2] = action[1] # dphi/dt
            Dstate[3] =  action[0] * rc.sin(state[2]) # acceleration x
            Dstate[4] = - g + action[0] * rc.cos(state[2]) # acceleration y

        return Dstate

    def out(self, state, time=None, action=None):
        # platform angular coordinate distance
        platform_midpoint_x = (self.platform[0] + self.platform[2])/2
        platform_midpoint_y = self.platform[1]

        dx  = platform_midpoint_x - state[0]
        dy  = platform_midpoint_y - state[1]

        r_psi = torch.tensor(self.get_radial(dx,dy))

        # lander angle
        phi = state[2]

        # rays
        rays_points = []
        for k in range(16):
            rays_points.append(self._compute_ray_intersection(state, k*np.pi/8))
        rays_points = torch.tensor(rays_points)

        # minimal distance points
        left = self._closest_point(state, 'left')
        right = self._closest_point(state, 'right')

        if type(r_psi) == torch.tensor:
            observation = torch.cat([r_psi,torch.tensor([phi]),rays_points,left,right,torch.tensor(state)], dim=-1) # dim_out = 28
        else:
            observation = np.concatenate([r_psi,np.array([phi]),rays_points,left,right,state])

        return observation

    def reset(self):
        if self.random_landscape:
            self.landscape, self.platform = self._generate_landscape(
                self.width,
                self.height
            )
        if self.random_initital_point:
            self._generate_state_init()
