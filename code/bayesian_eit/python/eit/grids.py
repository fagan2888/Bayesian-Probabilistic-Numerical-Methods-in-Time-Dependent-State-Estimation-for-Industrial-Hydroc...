from helpers import pol2cart
import numpy as np

class EITPattern(object):
	def __init__(self, meas_pattern, stim_pattern):
		self.__meas_pattern__ = meas_pattern
		self.__stim_pattern__ = stim_pattern

	@property
	def meas_pattern(self):
		return self.__meas_pattern__

	@property
	def stim_pattern(self):
		return self.__stim_pattern__


class EITGrid(object):
	def __init__(self, interior, boundary, sensors):
		if boundary is None:
			boundary = np.empty((0, 2))
		self.__interior__ = interior
		self.__boundary__ = boundary
		self.__sensors__ = sensors

	@property
	def sensors(self):
		return self.__sensors__

	@property
	def boundary(self):
		return self.__boundary__

	@property
	def interior(self):
		return self.__interior__

	@property
	def interior_plus_boundary(self):
		return np.row_stack([self.interior, self.boundary])

	@property
	def full_boundary(self):
		return np.row_stack([self.boundary, self.sensors])

	@property
	def all(self):
		return np.row_stack([self.interior, self.boundary, self.sensors])


def construct_shell(radii):
    r_spacing = radii[1] - radii[0]
    coords = [np.array([[0.,0.]])]
    for r in radii:
        # at each 'shell' we want a roughly equal number of theta around the diameter.
        # each theta should be about r_spacing apart
        n_theta = np.round(2*np.pi*r / r_spacing)
        thetas = np.linspace(0, 2*np.pi, n_theta+1)[:-1]
        x = np.cos(thetas)
        y = np.sin(thetas)
        coords.append(r*np.c_[x,y])
    coords = np.concatenate(coords)
    return coords


def construct_circular(n_radii, n_bdy, n_sensor):
	shell_radii = np.linspace(0, 1, n_radii + 2)[1:-1]
	interior = construct_shell(shell_radii)

	bdy_theta = np.linspace(0, 2*np.pi, n_bdy+1)
	design_bdy = pol2cart(np.c_[np.ones(n_bdy), bdy_theta[:-1]]) 

	bdy_sensor_skip = (int)(n_bdy / n_sensor)

	sensor_ixs = np.concatenate([np.arange(0, len(design_bdy), bdy_sensor_skip)[1:], [0]])
	is_sensor_flags = np.in1d(np.arange(len(design_bdy)), sensor_ixs)

	sensor_xy = design_bdy[sensor_ixs]
	non_sensor_xy = design_bdy[~is_sensor_flags]

	return EITGrid(interior, non_sensor_xy, sensor_xy)