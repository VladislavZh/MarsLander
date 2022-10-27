#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from typing import Any, Tuple
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
        scenario: Any,
        # xs: np.array,
        # ys: np.array,
        # angles: np.array,
    ):

        x_init, y_init, self.angle_init = initial_coords
        self.xy_init = np.c_[x_init, y_init]
        self.landscape_x, self.landscape_y = landscape[:, 0], landscape[:, 1]

        # self.xs, self.ys, self.angles = xs, ys, angles

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

        marslander_str = """M84.069,344.59c7.188,8.523,53.008,44.805,209.627,44.805c156.62,0,201.495-43.977,207.928-54.019
			c0-55.782-53.084-155.755-181.556-166.778l-0.934-48.857c21.568-7.082,37.051-27.371,36.596-51.263
			c-0.572-29.28-24.778-52.611-54.112-52.074c-29.315,0.572-52.628,24.791-52.068,54.129c0.455,23.915,16.728,43.597,38.587,49.827
			l0.765,39.713c-0.099,0.012-0.123,2.878-0.076,7.748C151.078,171.681,81.442,275.222,84.069,344.59
            M575.447,339.848c0-33.699-32.648-59.763-82.241-77.851c9.82,16.611,16.558,33.361,20.307,48.903
			c16.885,9.832,26.063,20.004,26.063,28.947c0,29.462-98.111,72.27-251.858,72.27c-153.746,0-251.846-42.808-251.846-72.27
			c0-11.069,13.849-24.008,39.282-35.871c5.021-15.799,12.804-31.925,23.29-47.433C39.901,274.685,0,302.598,0,339.848
			c0,35.371,35.931,62.333,89.657,80.49l-41.577,113h37.857l38.226-103.121c43.07,10.451,93.376,16.325,144.306,17.527v111.307
			h35.872V447.803c55.862-1.133,111.096-7.904,157.01-20.154L497.982,533h37.611l-40.088-116.129
			C543.803,398.807,575.447,373.036,575.447,339.848z"""

        #   marslander_str = """M371.873,315.686c2.521-29.395,3.85-55.362,3.997-77.896c-0.142-49.97-7.638-92.951-22.467-128.917
        # 	c-14.841-35.974-39.683-71.384-74.542-106.243c-1.817-1.818-4.094-2.693-6.823-2.625c-2.729,0.073-5.073,1.089-7.032,3.047
        # 	c-36.53,36.537-61.726,71.671-75.594,105.399c-13.855,33.74-20.851,76.855-20.992,129.34c0.141,22.546,1.475,48.507,3.99,77.896
        # 	c-99.731,0-43.88,220.252-43.88,220.252c1.126,4.486,4.205,6.867,9.241,7.143h26.873c2.521,0,4.755-0.979,6.713-2.944
        # 	c0.839-0.838,1.402-1.536,1.683-2.093l40.104-75.796l117.993-0.007l40.104,75.803c1.817,3.219,4.614,4.896,8.396,5.037
        # 	l26.879,0.006c2.791-0.281,5.037-1.261,6.714-2.943c1.261-1.262,2.099-2.656,2.521-4.199
        # 	C415.76,535.943,468.551,315.686,371.873,315.686z M292.295,309.559c-5.6,5.601-12.313,8.409-20.146,8.403
        # 	c-7.84,0-14.554-2.797-20.159-8.403c-5.606-5.605-8.403-12.319-8.403-20.159c-0.006-7.84,2.797-14.559,8.396-20.159
        # 	c5.601-5.594,12.32-8.396,20.159-8.396c7.84,0,14.554,2.797,20.159,8.403c5.601,5.6,8.397,12.32,8.403,20.16
        # 	C300.698,297.252,297.889,303.959,292.295,309.559z M292.295,229.999c-5.6,5.6-12.313,8.409-20.146,8.403
        # 	c-7.84,0-14.554-2.797-20.159-8.403c-5.606-5.606-8.403-12.32-8.403-20.159c-0.006-7.84,2.797-14.56,8.396-20.159
        # 	c5.601-5.594,12.32-8.397,20.159-8.397c7.84,0,14.554,2.797,20.159,8.403c5.601,5.6,8.397,12.319,8.403,20.159
        # 	C300.698,217.691,297.889,224.399,292.295,229.999z M292.295,150.439c-5.6,5.6-12.313,8.409-20.146,8.403
        # 	c-7.84,0-14.554-2.797-20.159-8.403c-5.606-5.606-8.403-12.32-8.403-20.159c-0.006-7.84,2.797-14.56,8.396-20.16
        # 	c5.601-5.593,12.32-8.396,20.159-8.396c7.84,0,14.554,2.797,20.159,8.403c5.601,5.6,8.397,12.32,8.403,20.159
        # 	C300.698,138.132,297.889,144.839,292.295,150.439z"""

        self.marslander_symbol_init = parse_path(marslander_str)
        self.marslander_symbol = parse_path(marslander_str)

        self.marslander_symbol_init.vertices -= (
            self.marslander_symbol_init.vertices.mean(axis=0)
        )
        self.marslander_symbol.vertices -= self.marslander_symbol.vertices.mean(axis=0)

        self.scale = 5.0

        t = (
            Affine2D().scale(self.scale).rotate_deg(180 + self.angle_init)
        )  # require matplotlib==3.6.0
        self.transform_init = t
        marslander_marker_init = MarkerStyle(self.marslander_symbol_init, transform=t)
        marslander_marker = MarkerStyle(
            self.marslander_symbol, transform=t, color="magenta"
        )

        self.marslander_line_init = self.ax.scatter(
            *self.xy_init[0], marker=marslander_marker_init
        )

        self.marslander_line = self.ax.scatter(
            *self.xy_init[0], marker=marslander_marker
        )

        self.episodic_scatter_handles = []
        for _ in range(self.scenario.N_episodes):
            new_handle = self.ax.scatter(
                self.x_init, self.y_init, marker=marslander_marker
            )
            self.episodic_scatter_handles.append(new_handle)

    def update_step(self):
        breakpoint()
        x, y, angle = self.scenario.observation

        t = (
            Affine2D().scale(self.scale).rotate_deg(180 + angle)
        )  # require matplotlib==3.6.0
        self.marslander_line.set_offsets(
            np.c_[x, y],
        )
        self.marslander_line.set_transform(t)

    def update_episode(self):
        offsets = self.marslander_line.get_offsets()
        t = self.marslander_line.get_offset_transform()
        handle = self.episodic_line_handles[self.scenario.episode_counter - 1]
        handle.set_offsets(offsets)
        handle.set_transform(t)
        self.marslander_line.set_offsets(self.xy_init)
        self.marslander_line.set_transform(self.transform_init)

    def update_iteration(self):
        for handle in self.episodic_line_handles:
            handle.set_offsets(self.xy_init)
            handle.set_transform(self.transform_init)

    def animate(self, k):
        sim_status = self.scenario.step()
        if sim_status == "simulation_ended":
            print("Simulation ended")
            self.anm.event_source.stop()
        self.update_step()
        if sim_status == "episode_ended":
            self.update_episode()
        elif sim_status == "iteration_ended":
            self.update_episode()
            self.update_iteration()
