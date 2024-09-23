'''Modified from PhotoMultiEmulator
'''
import os, sys
import numpy as np
import jax

from prospect.models import ProspectorParams

def rt_dir():

    dat_dirs = ['/storage/home/bbw5389/group/',
                '/Users/bwang/research/']

    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

def data_dir(data='pirate'):

    rt_roar = '/storage/home/bbw5389/group/'
    rt_mac = '/Users/bwang/research/'

    if data == 'pirate':
        dat_dirs = [rt_roar + 'sed/pirate/pirate/data/nn/',
                    rt_mac + 'software/MathewsEtAl2023/data/']
    else:
        return None

    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

multiemul_file = os.path.join(data_dir(data='pirate'), "parrot_v4_obsphot_512n_5l_24s_00z24.npy")


class EmulatorBeta(ProspectorParams):
    '''works with the p-beta priors:
    '''

    def __init__(self, model_params, fp=multiemul_file, obs=None, param_order=None):
        super(EmulatorBeta, self).__init__(model_params, param_order=param_order)

        # Load all emulator data.
        self.dat = np.load(fp, allow_pickle=True).all()

        # Determine the indices of the filters in obs['filters']
        self.sorter = np.array([self.dat["filter_redir"][f.name] for f in obs["filters"]], dtype=int)

        # TODO: Check if prior bounds are with emulbounds.
        # This will require messing with the massmet prior (which uses prior.bounds()) to
        # bring it in line with the other priors (which use prior.a and prior.b)

        # TODO: Use dat["resid_quants"] data to modify obs["maggies_unc"]

        # needed for re-ordering thetas
        self.zred_index = self.dat["zred_index"]

        # Convert ANN things to JAX arrays.
        self.nets = self.dat["nets"]
        for (i, net) in enumerate(self.nets):
            # Unpack this sub-network
            norm, denorm, wght, bias, act = net

            # Translate things into JAX and remove irrelevant filters
            norm = [jax.numpy.array(n) for n in norm]
            denorm = [jax.numpy.array(d[self.sorter]) for d in denorm]
            wght = [jax.numpy.array(W) for W in wght]
            bias = [jax.numpy.array(b) for b in bias]
            wght[-1] = wght[-1][self.sorter, :]
            bias[-1] = bias[-1][self.sorter]

            # The activation function is stored as a string - this converts it
            # to the JAX function. Assumes that all nonlinear layers use the same
            # activation function in a given sub-network and that the last layer
            # is linear activation.
            act = getattr(jax.nn, act)

            # Repack
            self.nets[i] = (norm, denorm, wght, bias, act)

    # Split zredmassmet and put each into the correct index
    def modify_theta(self, theta):
        new_theta = theta[1:]
        new_theta = np.insert(new_theta, self.zred_index, theta[0])
        return new_theta

    # Determine the relevant angle needed for the sin and cos functions in
    # transition_coef. z0 is the center of an sub-network overlap and deltaz
    # is the full width of the overlap.
    def transition_theta(self, z, z0, deltaz):
        return (jax.numpy.pi / (2 * deltaz)) * (z - z0 + 0.5*deltaz)

    # Compute how much each sub-network should contribute to the overall
    # prediction. Equal to 1 where the sub-network is the only one trained,
    # 0 where the sub-emulator is not trained, and between 0 and 1 in overlaps.
    def transition_coef(self, z, z0l, deltazl, z0r, deltazr):
        if (z <= (z0l - 0.5*deltazl)) or (z >= (z0r + 0.5*deltazr)):
            return 0.0
        elif (z >= (z0l + 0.5*deltazl)) and (z <= (z0r - 0.5*deltazr)):
            return 1.0
        elif (z > (z0l - 0.5*deltazl)) and (z < (z0l + 0.5*deltazl)):
            return jax.numpy.sin(self.transition_theta(z, z0l, deltazl))**2
        else:
            return jax.numpy.cos(self.transition_theta(z, z0r, deltazr))**2

    # Run a forward pass through a sub-network.
    def forward_pass(self, in_vec, net):
        # Grab NN stuff
        norm, denorm, wght, bias, act = net

        # Normalize layer
        result = (in_vec - norm[0]) / norm[1]

        # Nonlinear layers
        for i in range(len(wght) - 1):
            result = act(jax.numpy.dot(wght[i], result) + bias[i])

        # Linear layer.
        result = jax.numpy.dot(wght[-1], result) + bias[-1]

        # Denormalize layer.
        result = (denorm[1] * result) + denorm[0]

        return result

    # Convert AB magnitudes to maggies.
    #
    # Note: The emulator actually outputs arsinh magnitudes, but here we
    # are assuming that these arsinh mags are exactly equal to their
    # logarithmic AB counterparts. This is a very safe assumption for AB
    # magnitudes brighter than about +33 (they asymptotically become equal
    # as the AB magnitude gets brighter), but becomes very poor fainter than
    # +34.
    def demagify(self, mag):
        result = jax.numpy.multiply(mag, -0.4)
        result = jax.numpy.divide(result, jax.numpy.log10(jax.numpy.e))
        result = jax.numpy.exp(result)

        return result

    # Use the full emulator suite to predict a set of photometry for a given input
    # theta_in parameter vector.
    def predict_phot(self, theta_in, **extras):
        # Add redshift to theta if needed.
        theta = self.modify_theta(theta_in)

        # Grab the redshift from theta.
        z = theta[self.zred_index]

        # Compute the coefficients to multiply each emulator by.
        coefs = [self.transition_coef(z, *emul_lim) for emul_lim in self.dat["emulator_limits"]]

        # Initialize an array for the photometry prediction.
        result = jax.numpy.zeros(len(self.sorter))

        for (i, coef) in enumerate(coefs):
            # If the coef is zero, sub-network doesn't contribute - skip.
            if coef == 0.0:
                continue
            # Otherwise, it does contribute and we need to evaluate that sub-network.
            else:
                result += coef * self.forward_pass(theta, self.nets[i])

        # Convert the emulator's output to maggies (see comments on self.demagify)
        predict_lin = self.demagify(result)

        return predict_lin

    # Run self.predict_phot but return the result as a NumPy array
    def predict_phot_np(self, theta_in, **extras):
        result = self.predict_phot(theta_in)
        result = np.array(result)
        return result

    # We don't predict spectra, both because we don't have an emulator available
    # and because it would take too long anyway. Thus, let's just generate a vector
    # of zeros and return that in case Prospector complains about not having a
    # spectrum predicted.
    def predict_spec(self, theta, obs=None, **extras):
        return np.zeros(5994)

    # Same thing with mfrac, although we *do* have an emulator for this.
    def predict_mfrac(self, theta, obs=None, **extras):
        return -1.0

    # General predict function, returns useless mfrac and spec right now.
    def predict(self, theta, obs=None, **extras):
        return (self.predict_spec(theta, obs=obs, **extras),
                self.predict_phot_np(theta, obs=obs, **extras),
                self.predict_mfrac(theta, obs=obs, **extras))


class EmulatorDefault(ProspectorParams):
    '''works with the full 18 params default settings (p-alpha: MassMet prior, flat SFH)
    '''

    def __init__(self, model_params, fp=multiemul_file, obs=None, param_order=None):
        super(EmulatorDefault, self).__init__(model_params, param_order=param_order)

        # Load all emulator data.
        self.dat = np.load(fp, allow_pickle=True).all()

        # Determine the indices of the filters in obs['filters']
        self.sorter = np.array([self.dat["filter_redir"][f.name] for f in obs["filters"]], dtype=int)

        # TODO: Check if prior bounds are with emulbounds.
        # This will require messing with the massmet prior (which uses prior.bounds()) to
        # bring it in line with the other priors (which use prior.a and prior.b)

        # TODO: Use dat["resid_quants"] data to modify obs["maggies_unc"]

        # needed for re-ordering thetas
        self.zred_index = self.dat["zred_index"]

        # Convert ANN things to JAX arrays.
        self.nets = self.dat["nets"]
        for (i, net) in enumerate(self.nets):
            # Unpack this sub-network
            norm, denorm, wght, bias, act = net

            # Translate things into JAX and remove irrelevant filters
            norm = [jax.numpy.array(n) for n in norm]
            denorm = [jax.numpy.array(d[self.sorter]) for d in denorm]
            wght = [jax.numpy.array(W) for W in wght]
            bias = [jax.numpy.array(b) for b in bias]
            wght[-1] = wght[-1][self.sorter, :]
            bias[-1] = bias[-1][self.sorter]

            # The activation function is stored as a string - this converts it
            # to the JAX function. Assumes that all nonlinear layers use the same
            # activation function in a given sub-network and that the last layer
            # is linear activation.
            act = getattr(jax.nn, act)

            # Repack
            self.nets[i] = (norm, denorm, wght, bias, act)

    # No modifications needed since we use the full 18 params and default priors
    def modify_theta(self, theta):
        return theta

    # Determine the relevant angle needed for the sin and cos functions in
    # transition_coef. z0 is the center of an sub-network overlap and deltaz
    # is the full width of the overlap.
    def transition_theta(self, z, z0, deltaz):
        return (jax.numpy.pi / (2 * deltaz)) * (z - z0 + 0.5*deltaz)

    # Compute how much each sub-network should contribute to the overall
    # prediction. Equal to 1 where the sub-network is the only one trained,
    # 0 where the sub-emulator is not trained, and between 0 and 1 in overlaps.
    def transition_coef(self, z, z0l, deltazl, z0r, deltazr):
        if (z <= (z0l - 0.5*deltazl)) or (z >= (z0r + 0.5*deltazr)):
            return 0.0
        elif (z >= (z0l + 0.5*deltazl)) and (z <= (z0r - 0.5*deltazr)):
            return 1.0
        elif (z > (z0l - 0.5*deltazl)) and (z < (z0l + 0.5*deltazl)):
            return jax.numpy.sin(self.transition_theta(z, z0l, deltazl))**2
        else:
            return jax.numpy.cos(self.transition_theta(z, z0r, deltazr))**2

    # Run a forward pass through a sub-network.
    def forward_pass(self, in_vec, net):
        # Grab NN stuff
        norm, denorm, wght, bias, act = net

        # Normalize layer
        result = (in_vec - norm[0]) / norm[1]

        # Nonlinear layers
        for i in range(len(wght) - 1):
            result = act(jax.numpy.dot(wght[i], result) + bias[i])

        # Linear layer.
        result = jax.numpy.dot(wght[-1], result) + bias[-1]

        # Denormalize layer.
        result = (denorm[1] * result) + denorm[0]

        return result

    # Convert AB magnitudes to maggies.
    #
    # Note: The emulator actually outputs arsinh magnitudes, but here we
    # are assuming that these arsinh mags are exactly equal to their
    # logarithmic AB counterparts. This is a very safe assumption for AB
    # magnitudes brighter than about +33 (they asymptotically become equal
    # as the AB magnitude gets brighter), but becomes very poor fainter than
    # +34.
    def demagify(self, mag):
        result = jax.numpy.multiply(mag, -0.4)
        result = jax.numpy.divide(result, jax.numpy.log10(jax.numpy.e))
        result = jax.numpy.exp(result)

        return result

    # Use the full emulator suite to predict a set of photometry for a given input
    # theta_in parameter vector.
    def predict_phot(self, theta_in, **extras):
        # returns theta_in
        # No modifications needed since we use the full 18 params and default priors
        theta = self.modify_theta(theta_in)

        # Grab the redshift from theta.
        z = theta[self.zred_index]

        # Compute the coefficients to multiply each emulator by.
        coefs = [self.transition_coef(z, *emul_lim) for emul_lim in self.dat["emulator_limits"]]

        # Initialize an array for the photometry prediction.
        result = jax.numpy.zeros(len(self.sorter))

        for (i, coef) in enumerate(coefs):
            # If the coef is zero, sub-network doesn't contribute - skip.
            if coef == 0.0:
                continue
            # Otherwise, it does contribute and we need to evaluate that sub-network.
            else:
                result += coef * self.forward_pass(theta, self.nets[i])

        # Convert the emulator's output to maggies (see comments on self.demagify)
        predict_lin = self.demagify(result)

        return predict_lin

    # Run self.predict_phot but return the result as a NumPy array
    def predict_phot_np(self, theta_in, **extras):
        result = self.predict_phot(theta_in)
        result = np.array(result)
        return result

    # We don't predict spectra, both because we don't have an emulator available
    # and because it would take too long anyway. Thus, let's just generate a vector
    # of zeros and return that in case Prospector complains about not having a
    # spectrum predicted.
    def predict_spec(self, theta, obs=None, **extras):
        return np.zeros(5994)

    # Same thing with mfrac, although we *do* have an emulator for this.
    def predict_mfrac(self, theta, obs=None, **extras):
        return -1.0

    # General predict function, returns useless mfrac and spec right now.
    def predict(self, theta, obs=None, **extras):
        return (self.predict_spec(theta, obs=obs, **extras),
                self.predict_phot_np(theta, obs=obs, **extras),
                self.predict_mfrac(theta, obs=obs, **extras))
