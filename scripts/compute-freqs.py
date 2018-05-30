# Standard library
import logging
from os import path
import sys
import time

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import h5py
from pyia import GaiaData

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic
import superfreq as sf

from logger import logger

# Global parameters
gc_frame = coord.Galactocentric(z_sun=0*u.pc, galcen_distance=8.3*u.kpc)
mw = gp.MilkyWayPotential()
H = gp.Hamiltonian(mw)
integrate_time = 16. * u.Gyr # HACK: hardcoded! ~64 orbital periods
Nmax = 6 # for find_actions
global cache_file


def cartesian_to_poincare_polar(w):
    r"""
    Convert an array of 6D Cartesian positions to Poincaré
    symplectic polar coordinates. These are similar to cylindrical
    coordinates.
    Parameters
    ----------
    w : array_like
        Input array of 6D Cartesian phase-space positions. Should have
        shape ``(norbits,6)``.
    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        Points represented in 6D Poincaré polar coordinates.
    """

    R = np.sqrt(w[..., 0]**2 + w[..., 1]**2)
    # phi = np.arctan2(w[...,1], w[...,0])
    phi = np.arctan2(w[..., 0], w[..., 1])

    vR = (w[..., 0]*w[..., 0+3] + w[..., 1]*w[..., 1+3]) / R
    vPhi = w[..., 0]*w[..., 1+3] - w[..., 1]*w[..., 0+3]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    z = w[..., 2]
    zdot = w[..., 2+3]

    new_w = np.vstack((R.T, pp_phi.T, z.T,
                       vR.T, pp_phidot.T, zdot.T)).T
    return new_w


def get_Ec(w0):
    x0 = w0.cylindrical.rho.to(u.pc).value
    x0_c = [x0, 0, 0] * u.pc
    v0 = mw.circular_velocity(x0_c)[0].to(u.km/u.s).value
    v0_c = [0, v0, 0] * u.km/u.s
    w0_c = gd.PhaseSpacePosition(pos=x0_c, vel=v0_c)
    Ec = w0_c.energy(H)[0]
    return Ec


def worker(task):
    i, w0 = task
    _time0 = time.time()
    w0 = gd.PhaseSpacePosition.from_w(w0, galactic)

    data = dict()

    try:
        orbit = H.integrate_orbit(w0, dt=0.5*u.Myr, t1=0*u.Myr,
                                  t2=integrate_time)
    except Exception as e:
        logger.error('Failed to integrate orbit {0}\n{1}'
                     .format(i, str(e)))
        return i, None

    # Compute frequencies with superfreq:
    new_ws = cartesian_to_poincare_polar(orbit.w().T).T
    fs = [(new_ws[j] + 1j*new_ws[j+3]) for j in range(3)]

    freq = sf.SuperFreq(orbit.t.value, p=4)
    try:
        res = freq.find_fundamental_frequencies(fs)
        data['sf_freqs'] = res.fund_freqs
    except Exception as e:
        logger.error('Failed to run superfreq for orbit {0}\n{1}'
                     .format(i, str(e)))

    # Compute actions / frequencies / angles
    try:
        res = gd.find_actions(orbit, N_max=Nmax)
        data['actions'] = res['actions'].to(u.km/u.s*u.kpc)
        data['angles'] = res['angles'].to(u.rad)
        data['freqs'] = res['freqs'].to(1/u.Myr)
    except Exception as e:
        logger.error('Failed to run find actions for orbit {0}\n{1}'
                     .format(i, str(e)))

    # Other various things:
    try:
        data['zmax'] = orbit.zmax(approximate=True).to(u.kpc)
        data['rper'] = orbit.pericenter(approximate=True).to(u.kpc)
        data['rapo'] = orbit.apocenter(approximate=True).to(u.kpc)
    except Exception as e:
        logger.error('Failed to compute zmax peri apo for orbit {0}\n{1}'
                     .format(i, str(e)))

    # Lz and E
    try:
        data['Lz'] = orbit[0].angular_momentum()[2].to(u.km/u.s*u.kpc)

        E = orbit.energy()[0]
        Ec = get_Ec(w0)
        data['E-Ec'] = (E-Ec).to((u.kpc/u.Myr)**2)
    except Exception as e:
        logger.error('Failed to compute E Lz for orbit {0}\n{1}'
                     .format(i, str(e)))

    logger.debug('Time spent: {0:.2f}'.format(time.time() - _time0))

    return i, data


def callback(result):
    i, data = result

    if data is None:
        return

    with h5py.File(cache_file) as f:
        for k in data.keys():
            if hasattr(data[k], 'unit') and 'unit' not in f[k].attrs:
                f[k].attrs['unit'] = str(data[k].unit)

            else:
                f[k][i] = data[k]


def main(pool, source_file):
    logger.debug('Starting file {0}...'.format(source_file))
    logger.debug('Writing to cache file {0}'.format(cache_file))

    # Load Gaia data
    g = GaiaData(source_file)

    c = g.skycoord
    galcen = c.transform_to(gc_frame)
    logger.debug('Data loaded...')

    w0 = gd.PhaseSpacePosition(galcen.cartesian)
    w0 = w0.w(galactic).T

    # Make sure output file exists
    if not path.exists(cache_file):
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('sf_freqs', shape=(w0.shape[0], 3),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('freqs', shape=(w0.shape[0], 3),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('actions', shape=(w0.shape[0], 3),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('angles', shape=(w0.shape[0], 3),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('zmax', shape=(w0.shape[0], ),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('rper', shape=(w0.shape[0], ),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('rapo', shape=(w0.shape[0], ),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('E-Ec', shape=(w0.shape[0], ),
                             dtype='f8', fillvalue=np.nan)
            f.create_dataset('Lz', shape=(w0.shape[0], ),
                             dtype='f8', fillvalue=np.nan)

    tasks = [(i, w0[i]) for i in range(w0.shape[0])]
    for r in pool.map(worker, tasks, callback=callback):
        pass

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    import schwimmbad
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")

    logger.setLevel(logging.DEBUG)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-f", "--file", dest="source_file", required=True)

    args = parser.parse_args()

    basename = path.splitext(path.basename(args.source_file))[0]
    cache_file = path.join('../cache/{0}-orbits.hdf5'.format(basename))

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool, args.source_file)
