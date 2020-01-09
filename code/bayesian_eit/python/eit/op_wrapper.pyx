from eigency.core cimport *
cimport numpy as np

cdef extern from "operators.hpp":
	MatrixXd c_Id_Id "Id_Id"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)
	MatrixXd c_Id_A "Id_A"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)
	MatrixXd c_Id_B "Id_B"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)
	MatrixXd c_A_A "A_A"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)
	MatrixXd c_A_B "A_B"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)
	MatrixXd c_B_B "B_B"(Map[MatrixXd] &x, Map[MatrixXd] &y, Map[VectorXd] &args)

def Id_Id(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_Id_Id(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))

def Id_A(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_Id_A(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))

def Id_B(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_Id_B(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))

def A_A(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_A_A(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))

def A_B(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_A_B(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))

def B_B(np.ndarray[ndim=2, dtype=np.float_t] x, np.ndarray[ndim=2, dtype=np.float_t] y, np.ndarray[ndim=1, dtype=np.float_t] args):
	return ndarray_copy(c_B_B(Map[MatrixXd](x), Map[MatrixXd](y), Map[VectorXd](args)))