import warnings
from halotools.empirical_models.occupation_models.zheng07_components import *

import jax
from jax import numpy as jnp

class JaxZheng07Cens(Zheng07Cens):
    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = jnp.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            raise HalotoolsError(msg)

        logM = jnp.log10(mass)
        mean_ncen = 0.5*(1.0 + jax.scipy.special.erf(
            (logM - self.param_dict['logMmin']) / self.param_dict['sigma_logM']))

        return mean_ncen

class JaxZheng07Sats(Zheng07Sats):
    def mean_occupation(self, **kwargs):
        if self.modulate_with_cenocc:
            for key, value in list(self.param_dict.items()):
                if key in self.central_occupation_model.param_dict:
                    self.central_occupation_model.param_dict[key] = value

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = jnp.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)

        M0 = 10.**self.param_dict['logM0']
        M1 = 10.**self.param_dict['logM1']

        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning
        mean_nsat = jnp.zeros_like(mass)

        idx_nonzero = jnp.where(mass - M0 > 0)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # mean_nsat[idx_nonzero] = ((mass[idx_nonzero] - M0)/M1)**self.param_dict['alpha']
            mean_nsat = jax.ops.index_update(
                mean_nsat, idx_nonzero,
                ((mass[idx_nonzero] - M0)/M1)**self.param_dict['alpha'])

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(self.central_occupation_model, "baseline_mean_occupation",\
                                    self.central_occupation_model.mean_occupation)(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat
