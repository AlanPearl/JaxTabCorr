from tabcorr.tabcorr import *

import jax
from jax import numpy as jnp
import numpy as np


class JaxTabCorr(TabCorr):
    def predict(self, model, separate_gal_type=False, **occ_kwargs):
        """
        Predicts the number density and correlation function for a certain
        model.
        Parameters
        ----------
        model : HodModelFactory
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made.
        separate_gal_type : boolean, optional
            If True, the return values are dictionaries divided by each galaxy
            types contribution to the output result.
        **occ_kwargs : dict, optional
                Keyword arguments passed to the ``mean_occupation`` functions
                of the model.
        Returns
        -------
        ngal : numpy.array or dict
            Array or dictionary of arrays containing the number densities for
            each galaxy type stored in self.gal_type. The total galaxy number
            density is the sum of all elements of this array.
        xi : numpy.array or dict
            Array or dictionary of arrays storing the prediction for the
            correlation function.
        """

        try:
            assert (sorted(model.gal_types) == sorted(
                ['centrals', 'satellites']))
        except AssertionError:
            raise RuntimeError('The model instance must only have centrals ' +
                               'and satellites as galaxy types. Check the ' +
                               'gal_types attribute of the model instance.')

        try:
            assert (model._input_model_dictionary['centrals_occupation']
                    .prim_haloprop_key == self.attrs['prim_haloprop_key'])
            assert (model._input_model_dictionary['satellites_occupation']
                    .prim_haloprop_key == self.attrs['prim_haloprop_key'])
        except AssertionError:
            raise RuntimeError('Mismatch in the primary halo properties of ' +
                               'the model and the TabCorr instance.')

        try:
            if hasattr(model._input_model_dictionary['centrals_occupation'],
                       'sec_haloprop_key'):
                assert (model._input_model_dictionary['centrals_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
            if hasattr(model._input_model_dictionary['satellites_occupation'],
                       'sec_haloprop_key'):
                assert (model._input_model_dictionary['satellites_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
        except AssertionError:
            raise RuntimeError('Mismatch in the secondary halo properties ' +
                               'of the model and the TabCorr instance.')

        try:
            assert jnp.abs(model.redshift - self.attrs['redshift']) < 0.05
        except AssertionError:
            raise RuntimeError('Mismatch in the redshift of the model and ' +
                               'the TabCorr instance.')

        mean_occupation = jnp.zeros(len(self.gal_type))

        mask = self.gal_type['gal_type'] == 'centrals'

        mean_occupation = jax.ops.index_update(
            mean_occupation, mask,
            model.mean_occupation_centrals(
                prim_haloprop=self.gal_type['prim_haloprop'][mask],
                sec_haloprop_percentile=(
                    self.gal_type['sec_haloprop_percentile'][mask]), **occ_kwargs)
        )
        mean_occupation = jax.ops.index_update(
            mean_occupation, ~mask,
            model.mean_occupation_satellites(
                prim_haloprop=self.gal_type['prim_haloprop'][~mask],
                sec_haloprop_percentile=(
                    self.gal_type['sec_haloprop_percentile'][~mask]), **occ_kwargs)
        )

        ngal = mean_occupation * self.gal_type['n_h'].data

        if self.attrs['mode'] == 'auto':
            ngal_sq = jnp.outer(ngal, ngal)
            ngal_sq = 2 * ngal_sq - jnp.diag(jnp.diag(ngal_sq))
            ngal_sq = symmetric_matrix_to_array(ngal_sq)

            xi = self.tpcf_matrix * ngal_sq / jnp.sum(ngal_sq)
        elif self.attrs['mode'] == 'cross':
            xi = self.tpcf_matrix * ngal / jnp.sum(ngal)

        if not separate_gal_type:
            ngal = jnp.sum(ngal)
            xi = jnp.sum(xi, axis=1).reshape(self.tpcf_shape)
            return ngal, xi
        else:
            ngal_dict = {}
            xi_dict = {}

            for gal_type in np.unique(self.gal_type['gal_type']):
                mask = self.gal_type['gal_type'] == gal_type
                ngal_dict[gal_type] = jnp.sum(ngal[mask])

            if self.attrs['mode'] == 'auto':
                for gal_type_1, gal_type_2 in (
                        itertools.combinations_with_replacement(
                            np.unique(self.gal_type['gal_type']), 2)):
                    mask = symmetric_matrix_to_array(jnp.outer(
                        gal_type_1 == self.gal_type['gal_type'],
                        gal_type_2 == self.gal_type['gal_type']) |
                                                     jnp.outer(
                                                         gal_type_2 == self.gal_type['gal_type'],
                                                         gal_type_1 == self.gal_type['gal_type']))
                    xi_dict['%s-%s' % (gal_type_1, gal_type_2)] = jnp.sum(
                        xi * mask, axis=1).reshape(self.tpcf_shape)

            elif self.attrs['mode'] == 'cross':
                for gal_type in np.unique(self.gal_type['gal_type']):
                    mask = self.gal_type['gal_type'] == gal_type
                    xi_dict[gal_type] = jnp.sum(
                        xi * mask, axis=1).reshape(self.tpcf_shape)

            return ngal_dict, xi_dict
