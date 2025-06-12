import scipy.stats as stats
import numpy as np

def TruncatedNormalDust(mean, stdev, dist_min, dist_max):
    """
    Returns a truncated normal distribution object.
    """
    # Calculate the standard normal (z) bounds
    a, b = (dist_min - mean) / stdev, (dist_max - mean) / stdev

    # Return the truncated normal distribution object
    return stats.truncnorm(a, b, loc=mean, scale=stdev)

def LogUniformDust(minEBV = 1e-5, maxEBV = 1):
    """
    Returns a custom log-uniform distribution object.
    """
    class LogUniform:
        def __init__(self, minEBV, maxEBV):
            self.dist_min = minEBV
            self.dist_max = maxEBV

        def sample(self, size=1):
            return 10.**np.random.uniform(np.log10(self.dist_min), np.log10(self.dist_max), size=size)

    return LogUniform(minEBV, maxEBV)
