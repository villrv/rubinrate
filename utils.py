import numpy as np


def mag_to_flux(mag):
	return 10.**(-0.4*(mag+48.6))

def calc_efficiences(metric_tracker):
	efficiences = np.sum(metric_tracker, axis=-1)
	efficiences = efficiences / np.shape(metric_tracker)[-1]
	return efficiences