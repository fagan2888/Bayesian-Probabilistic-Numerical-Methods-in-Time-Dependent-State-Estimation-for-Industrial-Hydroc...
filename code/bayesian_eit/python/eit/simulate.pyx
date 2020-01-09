from eigency.core cimport *
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

cdef extern from "simulate.hpp":
	cdef cppclass SimulateResult:
		MatrixXd samples
		VectorXd acceptances
		VectorXd log_likelihoods
		vector[unique_ptr[MatrixXd]] sample_paths
	cdef unique_ptr[SimulateResult] _run_pcn_parallel "run_pcn_parallel"(
		int n_iter,
		double beta,
		Map[MatrixXd] theta_0,
		Map[VectorXd] prior_mean,
		Map[MatrixXd] sqrt_prior_cov,
		Map[MatrixXd] interior,
		Map[MatrixXd] sensors,
		Map[MatrixXd] theta_projection_mat,
		Map[VectorXd] kernel_args,
		Map[MatrixXd] stim_pattern,
		Map[MatrixXd] meas_pattern,
		Map[MatrixXd] data,
		double likelihood_variance,
		int n_threads,
		bint return_samples,
		bint bayesian
	)
	cdef unique_ptr[SimulateResult] _run_pcn_parallel_tempered "run_pcn_parallel_tempered"(
		int n_iter,
		double beta,
		Map[MatrixXd] theta_0,
		Map[VectorXd] prior_mean,
		Map[MatrixXd] sqrt_prior_cov,
		Map[MatrixXd] interior,
		Map[MatrixXd] sensors,
		Map[MatrixXd] theta_projection_mat,
		Map[VectorXd] kernel_args,
		Map[MatrixXd] stim_pattern,
		Map[MatrixXd] meas_pattern,
		Map[MatrixXd] data_1,
		Map[MatrixXd] data_2,
		double temp,
		double likelihood_variance,
		int n_threads,
		bint return_samples,
		bint bayesian
	)

def run_pcn_parallel(
	int n_iter,
	double beta,
	np.ndarray[ndim=2, dtype=np.float_t] theta_0,
	np.ndarray[ndim=1, dtype=np.float_t] prior_mean,
	np.ndarray[ndim=2, dtype=np.float_t] sqrt_prior_cov,
	np.ndarray[ndim=2, dtype=np.float_t] interior,
	np.ndarray[ndim=2, dtype=np.float_t] sensors,
	np.ndarray[ndim=2, dtype=np.float_t] theta_projection_mat,
	np.ndarray[ndim=1, dtype=np.float_t] kernel_args,
	np.ndarray[ndim=2, dtype=np.float_t] stim_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] meas_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] data,
	double likelihood_variance,
	int n_threads,
	bint bayesian
):
	cdef bint return_samples = theta_0.shape[0] == 1
	ret = _run_pcn_parallel(
		n_iter,
		beta,
		Map[MatrixXd](theta_0),
		Map[VectorXd](prior_mean),
		Map[MatrixXd](sqrt_prior_cov),
		Map[MatrixXd](interior),
		Map[MatrixXd](sensors),
		Map[MatrixXd](theta_projection_mat),
		Map[VectorXd](kernel_args),
		Map[MatrixXd](stim_pattern),
		Map[MatrixXd](meas_pattern),
		Map[MatrixXd](data),
		likelihood_variance,
		n_threads,
		return_samples,
		bayesian
	)

	if(return_samples):
		ret_samples = ndarray_copy(deref(deref(ret).sample_paths.at(0).get()))
	else:
		ret_samples = ndarray_copy(deref(ret).samples)

	return ret_samples, ndarray_copy(deref(ret).acceptances), ndarray_copy(deref(ret).log_likelihoods)

def run_pcn_parallel_tempered(
	int n_iter,
	double beta,
	np.ndarray[ndim=2, dtype=np.float_t] theta_0,
	np.ndarray[ndim=1, dtype=np.float_t] prior_mean,
	np.ndarray[ndim=2, dtype=np.float_t] sqrt_prior_cov,
	np.ndarray[ndim=2, dtype=np.float_t] interior,
	np.ndarray[ndim=2, dtype=np.float_t] sensors,
	np.ndarray[ndim=2, dtype=np.float_t] theta_projection_mat,
	np.ndarray[ndim=1, dtype=np.float_t] kernel_args,
	np.ndarray[ndim=2, dtype=np.float_t] stim_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] meas_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] data_1,
	np.ndarray[ndim=2, dtype=np.float_t] data_2,
	double temp,
	double likelihood_variance,
	int n_threads,
	bint bayesian
):
	cdef bint return_samples = theta_0.shape[0] == 1
	ret = _run_pcn_parallel_tempered(
		n_iter,
		beta,
		Map[MatrixXd](theta_0),
		Map[VectorXd](prior_mean),
		Map[MatrixXd](sqrt_prior_cov),
		Map[MatrixXd](interior),
		Map[MatrixXd](sensors),
		Map[MatrixXd](theta_projection_mat),
		Map[VectorXd](kernel_args),
		Map[MatrixXd](stim_pattern),
		Map[MatrixXd](meas_pattern),
		Map[MatrixXd](data_1),
		Map[MatrixXd](data_2),
		temp,
		likelihood_variance,
		n_threads,
		return_samples,
		bayesian
	)

	if(return_samples):
		ret_samples = ndarray_copy(deref(deref(ret).sample_paths.at(0).get()))
	else:
		ret_samples = ndarray_copy(deref(ret).samples)

	return ret_samples, ndarray_copy(deref(ret).acceptances), ndarray_copy(deref(ret).log_likelihoods)