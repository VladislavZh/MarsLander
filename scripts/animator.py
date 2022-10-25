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

        self.marslander_symbol = TextPath((0, 0), "🙭") # https://unicode-table.com/en/search/?q=rocket

        t = Affine2D().rotate_deg(self.angle_init) # require matplotlib==3.6.0
        marslander_marker = MarkerStyle(self.marslander_symbol, transform=t)

        self.marslander_line = self.ax.scatter(
            self.x_init, 
            self.y_init,
            marker=marslander_marker
        )

    def update_iteration(self, iter):
        t = Affine2D().rotate_deg(self.angles[iter])
        marslander_marker = MarkerStyle(self.marslander_symbol, transform=t)

        marslander_line = self.ax.scatter(
            self.xs[iter], 
            self.ys[iter],
            marker=marslander_marker
        )

    def animate(self, n_iters):
        n_iters_range = min(n_iters, len(self.xs))
        for k in range(n_iters_range):
            self.update_iteration(k)

