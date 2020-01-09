import bayesian_pdes as bpdes
import numpy as np
from scipy import stats
import mcmc
import collocate
import simulate

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

def theta_to_a(theta, sz_int, sz_bdy, proposal_dot_mat):
    theta = np.real_if_close(theta)
    theta_mod = np.dot(proposal_dot_mat, theta[:,None])
    kappa_int = theta_mod[:sz_int]
    kappa_bdy = theta_mod[sz_int:sz_int+sz_bdy]
    grad_kappa_x = theta_mod[sz_int+sz_bdy:2*sz_int+sz_bdy]
    grad_kappa_y = theta_mod[2*sz_int+sz_bdy:]
    return kappa_int, kappa_bdy, grad_kappa_x, grad_kappa_y


def construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=False):
    design_int = grid.interior_plus_boundary
    a_int, a_bdy, a_x, a_y = theta_to_a(theta,
                                        design_int.shape[0],
                                        grid.sensors.shape[0],
                                        proposal_dot_mat
                                        )

    augmented_int = np.column_stack([design_int, a_int, a_x, a_y])
    augmented_bdy = np.column_stack([grid.sensors, a_bdy, np.nan * np.zeros((a_bdy.shape[0], 2))])
    obs = [
        (augmented_int, None),
        (augmented_bdy, None)
    ]
    posterior = bpdes.collocate(
        op_system.operators,
        op_system.operators_bar,
        obs,
        op_system,
        collocate_args,
        inverter='np'
    )
    return posterior

def construct_c_posterior(locations, grid, theta, collocate_args, proposal_dot_mat, debug=False):
    design_int = grid.interior_plus_boundary
    a_int, a_bdy, a_x, a_y = theta_to_a(theta,
                                        design_int.shape[0],
                                        grid.sensors.shape[0],
                                        proposal_dot_mat
                                        )

    augmented_int = np.column_stack([design_int, a_int, a_x, a_y])
    augmented_bdy = np.column_stack([grid.sensors, a_bdy, np.nan * np.zeros((a_bdy.shape[0], 2))])
    mu_mult, Sigma = collocate.collocate_no_obs(
        np.asfortranarray(locations),
        np.asfortranarray(augmented_int),
        np.asfortranarray(augmented_bdy),
        np.asfortranarray(collocate_args)
    )
    return mu_mult, Sigma

def phi(grid, op_system, theta, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat, use_c=False, debug=False):
    # first solve forward
    design_int = grid.interior_plus_boundary
    # now determine voltage at the sensor locations
    # we have seven observations so take one for each sensor other than sensor 1, the reference sensor
    augmented_locations = np.column_stack([grid.sensors, np.nan * np.zeros((8, 3))])

    if use_c:
        mu_mult, Sigma = construct_c_posterior(augmented_locations, grid, theta, collocate_args, proposal_dot_mat, debug=debug)
    else:
        posterior = construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=debug)
        mu_mult, Sigma = posterior.no_obs_posterior(augmented_locations)

    

    # now need to iterate the stim patterns and compute the residual
    rhs_int = np.zeros((len(design_int), 1))

    Sigma_obs = np.dot(pattern.meas_pattern, np.dot(Sigma, pattern.meas_pattern.T))
    likelihood_cov = Sigma_obs + likelihood_variance * np.eye(Sigma_obs.shape[0])
    # likelihood_cov = likelihood_variance*np.eye(Sigma_obs.shape[0])
    likelihood_dist = stats.multivariate_normal(np.zeros(Sigma_obs.shape[0]), likelihood_cov)

    if debug:
        print likelihood_cov

    likelihood = 0
    for voltage, current in zip(data, pattern.stim_pattern):
        rhs_bdy = current[:, None]
        rhs = np.row_stack([rhs_int, rhs_bdy])

        model_voltage = np.dot(pattern.meas_pattern, np.dot(mu_mult, rhs))

        residual = voltage.ravel() - model_voltage.ravel()
        this_likelihood = likelihood_dist.logpdf(residual)
        if debug:
            with printoptions(precision=4, suppress=True):
                print("Model | True| Residual | Diag(cov)\n {}".format(np.c_[model_voltage, voltage, residual, np.diag(likelihood_cov)]))
            print("Likelihood: {}   |   Residual: {}".format(this_likelihood, np.abs(residual).sum()))
        likelihood += this_likelihood
    return -likelihood

def phi_c(grid, theta, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat, bayesian=True, debug=False):
    return -collocate.log_likelihood(
        np.asfortranarray(grid.interior_plus_boundary),
        np.asfortranarray(grid.sensors),
        np.asfortranarray(theta),
        np.asfortranarray(proposal_dot_mat),
        np.asfortranarray(collocate_args),
        np.asfortranarray(pattern.stim_pattern),
        np.asfortranarray(pattern.meas_pattern),
        np.asfortranarray(data),
        likelihood_variance,
        bayesian=bayesian,
        debug=debug
    )
def phi_c_tempered(grid, theta, likelihood_variance, pattern, data_1, data_2, temp, collocate_args, proposal_dot_mat, bayesian=True, debug=False):
    return -collocate.log_likelihood_tempered(
        np.asfortranarray(grid.interior_plus_boundary),
        np.asfortranarray(grid.sensors),
        np.asfortranarray(theta),
        np.asfortranarray(proposal_dot_mat),
        np.asfortranarray(collocate_args),
        np.asfortranarray(pattern.stim_pattern),
        np.asfortranarray(pattern.meas_pattern),
        np.asfortranarray(data_1),
        np.asfortranarray(data_2),
        temp,
        likelihood_variance,
        bayesian=bayesian,
        debug=debug
    )

class PCNKernel_C(object):
    def __init__(self, beta, prior_mean, sqrt_prior_cov, grid, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat):
        self.__beta__ = beta
        self.__prior_mean__ = prior_mean
        self.__sqrt_prior_cov__ = sqrt_prior_cov   
        self.__grid__ = grid
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data__ = data
        self.collocate_args = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat



    def phi(self, theta, collocate_args=None, bayesian=True, debug=False):
        return phi_c(
            self.__grid__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data__,
            self.collocate_args if collocate_args is None else collocate_args,
            self.__proposal_dot_mat__,
            bayesian=bayesian,
            debug=debug
        )

    def get_posterior(self, theta, locations, stim=None):
        mu_mult, cov = construct_c_posterior(
            locations, 
            self.__grid__, 
            theta, 
            self.collocate_args, 
            self.__proposal_dot_mat__
        )
        if stim is None:
            return mu_mult, cov
        mu = np.dot(mu_mult, np.r_[
            np.zeros(len(self.__grid__.interior_plus_boundary)),
            stim
        ])
        return mu, cov

    def apply(self, kappa_0, n_iter, n_threads=1, beta=None, bayesian=True):
        if len(kappa_0.shape) == 1:
            kappa_0 = np.copy(kappa_0[None, :])
        return simulate.run_pcn_parallel(
            n_iter,
            self.__beta__ if beta is None else beta,
            np.asfortranarray(kappa_0),
            np.asfortranarray(self.__prior_mean__),
            np.asfortranarray(self.__sqrt_prior_cov__),
            np.asfortranarray(self.__grid__.interior_plus_boundary),
            np.asfortranarray(self.__grid__.sensors),
            np.asfortranarray(self.__proposal_dot_mat__),
            np.asfortranarray(self.collocate_args),
            np.asfortranarray(self.__pattern__.stim_pattern),
            np.asfortranarray(self.__pattern__.meas_pattern),
            np.asfortranarray(self.__data__),
            self.__likelihood_variance__,
            n_threads,
            bayesian
        )

class PCNTemperingKernel_C(object):
    def __init__(self, beta, prior_mean, sqrt_prior_cov, grid, likelihood_variance, pattern, data_1, data_2, temp, collocate_args, proposal_dot_mat):
        self.__beta__ = beta
        self.__prior_mean__ = prior_mean
        self.__sqrt_prior_cov__ = sqrt_prior_cov   
        self.__grid__ = grid
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data_1__ = data_1
        self.__data_2__ = data_2
        self.__temp__ = temp
        self.__collocate_args__ = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat

    def phi(self, theta, bayesian=True, debug=False):
        return phi_c_tempered(
            self.__grid__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data_1__,
            self.__data_2__,
            self.__temp__,
            self.__collocate_args__,
            self.__proposal_dot_mat__,
            bayesian=bayesian,
            debug=debug
        )

    def get_posterior(self, theta, locations):
        return construct_c_posterior(
            locations, 
            self.__grid__, 
            theta, 
            self.__collocate_args__, 
            self.__proposal_dot_mat__
        )

    def apply(self, kappa_0, n_iter, n_threads=1, beta=None, bayesian=True):
        if len(kappa_0.shape) == 1:
            kappa_0 = np.copy(kappa_0[None, :])
        return simulate.run_pcn_parallel_tempered(
            n_iter,
            self.__beta__ if beta is None else beta,
            np.asfortranarray(kappa_0),
            np.asfortranarray(self.__prior_mean__),
            np.asfortranarray(self.__sqrt_prior_cov__),
            np.asfortranarray(self.__grid__.interior_plus_boundary),
            np.asfortranarray(self.__grid__.sensors),
            np.asfortranarray(self.__proposal_dot_mat__),
            np.asfortranarray(self.__collocate_args__),
            np.asfortranarray(self.__pattern__.stim_pattern),
            np.asfortranarray(self.__pattern__.meas_pattern),
            np.asfortranarray(self.__data_1__),
            np.asfortranarray(self.__data_2__),
            self.__temp__,
            self.__likelihood_variance__,
            n_threads,
            bayesian
        )

class PCNKernel(object):
    def __init__(self, proposal, grid, op_system, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat):
        self.__proposal__ = proposal   
        self.__grid__ = grid
        self.__op_system__ = op_system
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data__ = data
        self.__collocate_args__ = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat

    def phi(self, theta, debug=False):
        return phi(
            self.__grid__,
            self.__op_system__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data__,
            self.__collocate_args__,
            self.__proposal_dot_mat__,
            debug=debug
        )

    def apply(self, kappa_0, n_iter, n_threads=1):
        if n_threads > 1:
            raise Exception("Threading not currently supported in the python kernel.")

        if len(kappa_0.shape) == 1 or kappa_0.shape[1] == 1:
            return mcmc.pCN(n_iter, self.__proposal__, self.phi, kappa_0)

        # apply the kernel to each row of kappa_0
        # TODO: currently this kernel parallelises in the matrix mults - difficult to parallelise the for loop as a result
        ret = np.empty_like(kappa_0)
        acceptances = np.empty(kappa_0.shape[0])
        for i in xrange(kappa_0.shape[0]):
            results, accepts = self.apply(kappa_0[i], n_iter)
            ret[i] = results[-1,:]
            acceptances[i] = accepts.mean()
        return ret, acceptances, None