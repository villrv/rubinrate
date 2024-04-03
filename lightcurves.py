class LightCurve():
    """
    A basic transient model
    """

    def __init__(self, times, mags, filters, snrs, texp, tpeak, rmag, redshift, theta):
        """
        Parameters:
        ----------
        ...

        """
        self.times = times
        self.mags = mags
        self.filters = filters
        self.snrs = snrs
        self.texp = texp
        self.tpeak = tpeak
        self.rmag = rmag
        self.redshift = redshift
        self.theta = theta