#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from typing import Tuple
import numpy as np
import numpy.linalg as la
from rcognita.animators import Animator
from rcognita.utilities import update_line
from rcognita.utilities import reset_line
from rcognita.utilities import update_text
from rcognita.utilities import rc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import time
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage
import matplotlib.patches as patches

from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath
from svgpath2mpl import parse_path

# !pip install mpldatacursor <-- to install this
from mpldatacursor import datacursor

# !pip install svgpath2mpl matplotlib <-- to install this
from svgpath2mpl import parse_path

from collections import namedtuple



def getImage(path):
    return OffsetImage(plt.imread(path, format="png"), zoom=0.1)


class AnimatorMarsLander(Animator):
    def __init__(
        self,
        initial_coords: Tuple[float, float, float],
        landscape: np.array,
        xs: np.array,
        ys: np.array,
        angles: np.array,
    ):

        self.x_init, self.y_init, self.angle_init = initial_coords
        self.landscape_x, self.landscape_y = landscape[:, 0], landscape[:, 1]

        self.xs, self.ys, self.angles = xs, ys, angles

        self.colormap = plt.get_cmap("RdYlGn_r")

        self.fig_sim = plt.figure(figsize=(10, 10))

        self.ax = self.fig_sim.add_subplot(
            111,
            xlabel="",
            ylabel="",
        )
        self.ax.set_title(label="Mars Lander")
        self.ax.set_aspect("equal")

        (self.landscape_line,) = self.ax.plot(
            self.landscape_x, 
            self.landscape_y,
            lw=1.5,
        )

        # self.marslander_symbol_init = TextPath((0, 0), "☺") # "🚀") # "🙭") # https://unicode-table.com/en/search/?q=rocket
        # self.marslander_symbol = TextPath((0, 0), r'$\car$') # "$\u25b2$") # "🚀") # "🙭") # https://unicode-table.com/en/search/?q=rocket

        self.marslander_symbol_init = parse_path(
            """m 149.29696,59.506259 c 9.80781,-7.900008 10.75824,-16.233219 6.37567,-27.627905 -0.85296,-2.217698 -1.37907,-4.834869 -3.18784,-6.375671 -3.19606,-2.72257 -7.19466,-4.457397 -11.15742,-5.844365 -17.52648,-6.134266 -40.78203,4.427218 -52.06798,18.064402 -3.337512,4.032826 -11.558074,13.46383 -8.500893,20.189623 1.231787,2.709934 7.19859,6.710524 8.500893,7.438282 14.29629,7.989101 29.81747,9.717743 45.69231,6.375672 2.74016,-0.576876 5.35976,-1.641599 7.96959,-2.65653 1.80026,-0.700104 1.30747,-1.113436 2.65652,-2.125224 0.51085,-0.383129 2.04545,-0.611089 1.59392,-1.062611 -0.25046,-0.25046 -0.72658,-0.112009 -1.06261,0 -3.9331,1.311031 -7.87317,2.628836 -11.68873,4.250446 -8.38263,3.562618 -16.29338,8.14669 -24.44007,12.220035 -8.27841,4.139206 -14.840168,6.964495 -22.314846,12.751343 -3.436871,2.660803 -13.151879,16.777054 -14.345261,21.252234 -0.188148,0.70555 0.243663,1.45406 0.531307,2.12522 1.267548,2.95761 5.999602,5.00862 7.969589,5.84437 8.992227,3.81488 19.165571,5.03777 28.690521,2.65653 10.00868,-2.50217 19.53018,-6.57726 28.69052,-11.15742 2.81196,-1.40599 -3.41739,-2.14104 -7.43829,-2.65653 -4.77212,-0.61181 -9.53862,-1.38494 -14.34526,-1.59392 -9.06002,-0.39391 -23.044465,2.67524 -30.81574,7.43828 -5.683827,3.48364 -11.18317,10.69571 -15.40787,15.40787 -8.542007,9.52762 -3.740822,3.40469 -12.220035,14.87657 -4.957511,6.70722 -8.062259,14.78036 -3.719142,22.84615 3.097646,5.75277 10.161122,11.82613 15.407872,14.87657 21.535565,12.52067 41.767525,23.50907 66.413235,28.15921 7.94692,1.49942 27.28448,4.59387 35.5975,-3.71914"""
        )
        self.marslander_symbol = parse_path(
            """m 149.29696,59.506259 c 9.80781,-7.900008 10.75824,-16.233219 6.37567,-27.627905 -0.85296,-2.217698 -1.37907,-4.834869 -3.18784,-6.375671 -3.19606,-2.72257 -7.19466,-4.457397 -11.15742,-5.844365 -17.52648,-6.134266 -40.78203,4.427218 -52.06798,18.064402 -3.337512,4.032826 -11.558074,13.46383 -8.500893,20.189623 1.231787,2.709934 7.19859,6.710524 8.500893,7.438282 14.29629,7.989101 29.81747,9.717743 45.69231,6.375672 2.74016,-0.576876 5.35976,-1.641599 7.96959,-2.65653 1.80026,-0.700104 1.30747,-1.113436 2.65652,-2.125224 0.51085,-0.383129 2.04545,-0.611089 1.59392,-1.062611 -0.25046,-0.25046 -0.72658,-0.112009 -1.06261,0 -3.9331,1.311031 -7.87317,2.628836 -11.68873,4.250446 -8.38263,3.562618 -16.29338,8.14669 -24.44007,12.220035 -8.27841,4.139206 -14.840168,6.964495 -22.314846,12.751343 -3.436871,2.660803 -13.151879,16.777054 -14.345261,21.252234 -0.188148,0.70555 0.243663,1.45406 0.531307,2.12522 1.267548,2.95761 5.999602,5.00862 7.969589,5.84437 8.992227,3.81488 19.165571,5.03777 28.690521,2.65653 10.00868,-2.50217 19.53018,-6.57726 28.69052,-11.15742 2.81196,-1.40599 -3.41739,-2.14104 -7.43829,-2.65653 -4.77212,-0.61181 -9.53862,-1.38494 -14.34526,-1.59392 -9.06002,-0.39391 -23.044465,2.67524 -30.81574,7.43828 -5.683827,3.48364 -11.18317,10.69571 -15.40787,15.40787 -8.542007,9.52762 -3.740822,3.40469 -12.220035,14.87657 -4.957511,6.70722 -8.062259,14.78036 -3.719142,22.84615 3.097646,5.75277 10.161122,11.82613 15.407872,14.87657 21.535565,12.52067 41.767525,23.50907 66.413235,28.15921 7.94692,1.49942 27.28448,4.59387 35.5975,-3.71914"""
        )

        self.marslander_symbol_init.vertices -= self.marslander_symbol_init.vertices.mean(axis=0)
        self.marslander_symbol.vertices -= self.marslander_symbol.vertices.mean(axis=0)

        self.scale = 5.

        t = Affine2D().scale(self.scale).rotate_deg(180+self.angle_init) # require matplotlib==3.6.0
        marslander_marker_init = MarkerStyle(self.marslander_symbol_init, transform=t)
        marslander_marker = MarkerStyle(self.marslander_symbol)

        self.marslander_line_init = self.ax.scatter(
            self.x_init, 
            self.y_init,
            marker=marslander_marker_init
        )

        self.marslander_line = self.ax.scatter(
            self.x_init, 
            self.y_init,
            marker=marslander_marker
        )

    def update_iteration(self, iter):
        n_iter = min(iter, len(self.angles)-1)
        t = Affine2D().scale(self.scale).rotate_deg(180+self.angles[n_iter])
        # marslander_marker = MarkerStyle(self.marslander_symbol, transform=t)
            
        # if n_iter > 0: 
        #     # print(f'Scatter 2: {self.marslander_line}')
        #     self.marslander_line.remove()
            
        self.marslander_line.set_offsets(
            np.c_[self.xs[n_iter], self.ys[n_iter]],
        )
        self.marslander_line.set_transform(t)

        # print(f'Scatter 1: {type(self.marslander_line)}')

    def animate(self, n_iters):
        n_iters_range = min(n_iters, len(self.xs)-1)
        for k in range(n_iters_range):
            self.update_iteration(k)


