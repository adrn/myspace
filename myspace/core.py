# Standard library
import warnings

# Third-party
import numpy as np
import scipy as sp
from scipy.optimize import minimize

# Jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp
from jax.ops import index_update


class MySpace:
    """Find the position-dependent transformation that makes a given set of
    velocities look most like the reference distribution.

    Terminology
    -----------
    We define the "reference distribution" to be the (fitted) Gaussian Mixture
    Model (GMM) of a local (think: <100 pc or so) sample of stars. We define the
    "training set" to be a larger sample of stars that are used to fit the
    parameters of the position- and velocity-dependent transformation that
    maximizes the likelihood of a given set of velocities under the reference
    distribution.

    Parameters
    ----------
    gmm : `~sklearn.mixture.GaussianMixture`
        A Gaussian Mixture Model fitted to the velocity distribution of a local
        sample of stars. This distribution is treated as the reference velocity
        distribution.
    terms : `list`
        A list of strings specifying which terms to use in the transformation.
        By default, this is just `['x']`, meaning linear order in :math:`x`.
        However, this list could include terms like `'xv'` and `'xx'`, and
        (soon) even higher order terms like `'xxv'` and etc. So, for example,
        you could pass `terms=['x', 'xv', 'xx']`.

    Implementation notes
    --------------------
    In the code below, we assume the following index naming conventions:

    - `i`, `j`, `l`, `m` label coordinate components, i.e., x, y, z
    - `k` labels GMM mixture components
    - `n` labels data points

    """

    def __init__(self, gmm, terms=['x']):

        # Some relevant dimensionalities
        self.K = gmm.n_components
        self.dim = gmm.n_features_in_

        # Unpack the input GMM instance to have faster access to these
        # quantities:
        self.w_k = gmm.weights_  # (K, )
        self.mu_ki = gmm.means_  # (K, dim)
        self.C_kij = gmm.covariances_  # (K, dim, dim)
        self.Cinv_kij = gmm.precisions_  # (K, dim, dim)

        # pre-compute log-determinants
        # Note: this gets added to the quadratic form, inside the -1/2
        self._log_const_k = np.array(
            [np.linalg.slogdet(C)[1] + self.dim * np.log(2*np.pi)
             for C in self.C_kij])

        # Parse and validate the specified / desired 'terms' to use in the
        # expansion:
        self._allowed_terms = {
            'x': {'name': 'Aij', 'shape': (self.dim, self.dim)},
            'xv': {'name': 'B(x)ik', 'shape': (8, self.dim)}, # HACK MAGIC 8-ball
            'xx': {'name': 'Cijl', 'shape': (self.dim, self.dim, self.dim),
                   'symmetry': [1, 2]},
            # 'xxv': {'name': 'Dijlm', 'shape': (3, 3, 3, 3), 'symmetry': [1, 2]},
            # 'xvv': {'name': 'Fijlm', 'shape': (3, 3, 3, 3), 'symmetry': [2, 3]},
            # 'xxx': {'name': 'Gijlm', 'shape': (3, 3, 3, 3), 'symmetry': [1, 2, 3]},
        }
        self._tensor_name_to_term = {v['name']: k
                                     for k, v in self._allowed_terms.items()}
        self._Ms = np.array([ # THE HORROR
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[-1, 0, 0], [0, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]]])

        for k in self._allowed_terms:
            if 'symmetry' in self._allowed_terms[k]:
                # TODO: this only supports single symmetry currently! So, for
                # example, the 'xxx' symmetry is not yet supported
                self._allowed_terms[k]['size'] = int(self.dim**(self.dim-1) *
                                                     (self.dim+1) / 2)
            else:
                self._allowed_terms[k]['size'] = np.prod(
                    self._allowed_terms[k]['shape'])

        self._p_size = 0
        for term in terms:
            if term not in self._allowed_terms.keys():
                raise ValueError(
                    f'Term must be one of: {list(self._allowed_terms.keys())}')
            self._p_size += self._allowed_terms[term]['size']

        self.terms = terms
        self.tensors = None

    def _unpack_p(self, p, xs, jax=False):
        """Unpack the parameter array into the individual tensors"""

        if jax:
            xnp = jnp
            expm = jsp.linalg.expm
        else:
            xnp = np
            expm = sp.linalg.expm

        # TODO: these if blocks below should be automated...this means that this
        # class currently only works for the terms listed below!
        unpacked = {}

        i1 = 0

        if 'x' in self.terms:
            meta = self._allowed_terms['x']
            unpacked[meta['name']] = xnp.array(
                p[i1:i1 + meta['size']]).reshape(meta['shape'])
            i1 += meta['size']

        if 'xv' in self.terms:
            meta = self._allowed_terms['xv']
            uvecs = xnp.array(
                p[i1:i1 + meta['size']]).reshape(meta['shape'])
            i1 += meta['size']
            unpacked[meta['name']] = xnp.array(
                [expm(xnp.dot(xnp.dot(uvecs, x),
                              self._Ms.reshape(8, 9)).reshape(3, 3))
                 for x in xs])

        if 'xx' in self.terms:
            meta = self._allowed_terms['xx']

            _c = xnp.array(p[i1:i1 + meta['size']])
            C = xnp.zeros((3, 3, 3))
            for i in range(3):
                if jax:
                    # index_update(x, idx, y) <--> x[idx] = y
                    tri_idx = (jnp.full(6, i,), ) + jnp.triu_indices(3)
                    C = index_update(C, tri_idx, _c[i*6:(i+1)*6])
                    C = index_update(C.T, tri_idx, _c[i*6:(i+1)*6])
                else:
                    C[i][xnp.triu_indices(3)] = _c[i*6:(i+1)*6]
                    C[i].T[xnp.triu_indices(3)] = _c[i*6:(i+1)*6]

            unpacked[meta['name']] = C
            i1 += meta['size']

        return unpacked

    def get_model_v(self, v_ni, x_ni, tensors=None):
        """Compute the 'transformed' or 'adjusted' velocities given a dictionary
        of tensors and a set of positions and velocities.

        Parameters
        ----------
        tensors : dict
            A dictionary mapping from tensor name, e.g., `'Aij'`, to the actual
            tensor data as numpy arrays.
        v_ni : array_like
            The velocity data to transform.
        x_ni : array_like
            The corresponding position data, used to compute the transformed
            velocities.

        Returns
        -------
        model_v : JAX array
            Effectively a numpy array containing the transformed velocity data.
        """

        if tensors is None:
            if self.tensors is None:
                raise ValueError('MySpace does not have cached tensors! Did '
                                 'you run .fit() yet, or did it fail?')

            tensors = self.tensors

        # Used below to map from term components (like x or v) to the vector data:
        xv_data = {'x': jnp.array(x_ni), 'v': jnp.array(v_ni)}

        # Collect the terms used in the transformation:
        summed_terms = jnp.zeros(3)
        for tensor_name, T in tensors.items():
            if tensor_name == 'B(x)ik': # HACK: SPECIAL-CASING the xv term
                summed_terms += np.array([jnp.dot(B, v)
                                          for B, v in zip(T, xv_data['v'])])

            else:
                # Auto-construct the einsum magic
                vector_idx = ','.join([f'n{x}' for x in tensor_name[2:]])
                einsum_str = f"{tensor_name[1:]},{vector_idx}->ni"

                # Get vector arguments by mapping back from tensor name to term name
                term_name = self._tensor_name_to_term[tensor_name]
                vectors = [xv_data[xv] for xv in list(term_name)]

                # Set up einsum with the index string, tensor, and vector
                summed_terms += jnp.einsum(einsum_str, T, *vectors)

        # The model or transformed velocities are the input velocities, plus the
        # "correction" from the various tensor products
        model_v = v_ni + summed_terms

        return model_v

    def objective(self, p, v_ni, x_ni):
        """Compute the objective function given a parameter array.

        The parameter array is effectively a concatenated set of unraveled
        tensors, so it has a shape set by the sum of the number of (independent)
        components in each of the tensors used in this instance's
        transformation.

        The objective function here is the negative log-likelihood of the
        transformed velocities computed with the fitted GMM of the reference
        distribution.

        Parameters
        ----------
        p : array_like
            The parameter array.
        v_ni : array_like (N, 3)
            The velocity data.
        x_ni : array_like (N, 3)
            The position data.

        Returns
        -------
        obj : float
            The value of the objective function.
        """

        tensors = self._unpack_p(p, x_ni, jax=True)
        vv_ni = self.get_model_v(v_ni, x_ni, tensors=tensors)
        delta_nki = vv_ni[:, None] - self.mu_ki[None]  # (N, K, 3)

        # quadratic form
        # delta_nki : (N, K, 3)
        # Cinv_kij : (K, 3, 3)
        quad_nk = jnp.einsum('nki,kij,nkj->nk',
                             delta_nki, self.Cinv_kij, delta_nki)

        # scalar : (N, )
        arg_nk = (-0.5 * (self._log_const_k[None] + quad_nk) +
                  jnp.log(self.w_k)[None])
        scalar = jsp.special.logsumexp(arg_nk, axis=1)

        return -jnp.sum(scalar, axis=0) / scalar.shape[0]

    objective_and_grad = value_and_grad(objective, argnums=1)

    def fit(self, train_x, train_v, p0=None):
        train_x = np.array(train_x)
        train_v = np.array(train_v)
        assert train_x.shape == train_v.shape

        if p0 is None:
            p0 = np.zeros(self._p_size)

        res = minimize(self.objective_and_grad, x0=p0,
                       method='BFGS', jac=True,
                       args=(train_v, train_x))

        if res.success:
            self.tensors = self._unpack_p(res.x, train_x)
        else:
            warnings.warn("WARNING: failed to fit.",
                          category=RuntimeWarning)
            self.tensors = None

        return res, self.tensors
