import jax
from jax import numpy as jnp
# import numpy as np

from tabcorr.tabcorr import *


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

# TODO: Figure out how to add back in the redshift sanity check in a JAX-friendly way?
# ====================================================================================
#         try:
#             assert abs(model.redshift - self.attrs['redshift']) < 0.05
#         except AssertionError:
#             raise RuntimeError('Mismatch in the redshift of the model and ' +
#                                'the TabCorr instance.')

        mean_occupation = jnp.zeros(len(self.gal_type["gal_type"]))
        mask = self.gal_type["gal_type"] == "centrals"

        mean_occupation = jax.ops.index_update(
            mean_occupation, mask,
            model.mean_occupation_centrals(
                prim_haloprop=self.gal_type["prim_haloprop"][mask],
                sec_haloprop_percentile=self.gal_type["sec_haloprop_percentile"][mask],
                **occ_kwargs)
        )
        mean_occupation = jax.ops.index_update(
            mean_occupation, ~mask,
            model.mean_occupation_satellites(
                prim_haloprop=self.gal_type["prim_haloprop"][~mask],
                sec_haloprop_percentile=self.gal_type["sec_haloprop_percentile"][~mask],
                **occ_kwargs)
        )

        return jaxtabcorr_predict(
            mean_occupation,
            self.gal_type["gal_type"] == "centrals",
            self.gal_type["prim_haloprop"].data,
            self.gal_type["sec_haloprop_percentile"].data,
            self.gal_type["n_h"].data, self.tpcf_matrix,
            self.tpcf_shape, self.attrs["mode"] == "cross",
            separate_gal_type)


def jaxtabcorr_predict(mean_occupation, is_centrals, prim_haloprop,
                       sec_haloprop_percentile, n_h, tpcf_matrix,
                       tpcf_shape, do_cross, separate_gal_type):
    ngal = mean_occupation * n_h

    if not do_cross:
        ngal_sq = jnp.outer(ngal, ngal)
        ngal_sq = 2 * ngal_sq - jnp.diag(jnp.diag(ngal_sq))
        ngal_sq = jax_symmetric_matrix_to_array(ngal_sq)

        xi = tpcf_matrix * ngal_sq / jnp.sum(ngal_sq)
    else:
        xi = tpcf_matrix * ngal / jnp.sum(ngal)

    if not separate_gal_type:
        ngal = jnp.sum(ngal)
        xi = jnp.sum(xi, axis=1).reshape(tpcf_shape)
        return ngal, xi
    else:
        ngal_dict = {}
        xi_dict = {}

        for gal_type, key in [(True, "centrals"), (False, "satellites")]:
            mask = is_centrals == gal_type
            ngal_type = jnp.where(mask, ngal, 0)
            ngal_dict[key] = jnp.sum(ngal_type)  # <-- TODO: this will break

        if not do_cross:
            for gal_type_1, gal_type_2, name in [(True, True, "centrals-centrals"),
                                                 (True, False, "centrals-satellites"),
                                                 (False, False, "satellites-satellites")]:
                mask = jax_symmetric_matrix_to_array(jnp.outer(
                    gal_type_1 == is_centrals,
                    gal_type_2 == is_centrals) |
                        jnp.outer(
                             gal_type_2 == is_centrals,
                             gal_type_1 == is_centrals))
                xi_dict[name] = jnp.sum(xi * mask, axis=1).reshape(tpcf_shape)

        else:
            for gal_type, key in [(True, "centrals"), (False, "satellites")]:
                mask = is_centrals == gal_type
                xi_dict[gal_type] = jnp.sum(
                    xi * mask, axis=1).reshape(tpcf_shape)

        return ngal_dict, xi_dict

static_args = ["tpcf_shape", "do_cross", "separate_gal_type"]
jaxtabcorr_predict = jax.jit(jaxtabcorr_predict,
                             static_argnames=static_args)


def jax_symmetric_matrix_to_array(matrix):
    # Assertions not allowed by jit :(
    # try:
    #     assert matrix.shape[0] == matrix.shape[1]
    #     assert np.all(matrix == matrix.T)
    # except AssertionError:
    #     raise RuntimeError('The matrix you provided is not symmetric.')

    n_dim = matrix.shape[0]
    sel = jnp.zeros((n_dim**2 + n_dim) // 2, dtype=int)

    for i in range(matrix.shape[0]):
        sel = jax.ops.index_update(
            sel, slice((i*(i+1))//2, (i*(i+1))//2+(i+1)),
            jnp.arange(i*n_dim, i*n_dim + i + 1))
#         sel[(i*(i+1))//2:(i*(i+1))//2+(i+1)] = jnp.arange(
#             i*n_dim, i*n_dim + i + 1)

    return matrix.ravel()[sel]