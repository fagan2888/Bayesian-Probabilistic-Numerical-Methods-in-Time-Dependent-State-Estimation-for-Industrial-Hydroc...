import sympy as sp
import numpy as np
import hydrocyclone
import bayesian_pdes as bpdes

def test_c_operators():
    s_x, s_xbar, s_y, s_ybar = sp.symbols('x,xbar,y,ybar')
    a, a_x, a_y = sp.symbols('a,a_x,a_y')
    a_bar, a_x_bar, a_y_bar = sp.symbols('abar,a_xbar,a_ybar')
    s_length_scale, s_variance = sp.symbols('l,sigma')
    kernel = s_variance*sp.exp(-((s_x-s_xbar)**2 + (s_y-s_ybar)**2) / (2.*s_length_scale**2))
    symbols = [[s_x, s_y, a, a_x, a_y], [s_xbar, s_ybar, a_bar, a_x_bar, a_y_bar], [s_length_scale, s_variance]]

    def A(k):
        return sp.exp(a)*(k.diff(s_x,s_x) + k.diff(s_y,s_y) + k.diff(s_x)*a_x + k.diff(s_y)*a_y)
    def A_bar(k):
        return sp.exp(a_bar)*(k.diff(s_xbar,s_xbar) + k.diff(s_ybar,s_ybar) + k.diff(s_xbar)*a_x_bar + k.diff(s_ybar)*a_y_bar)
    def B(k):
        return sp.exp(a)*(k.diff(s_x)*s_x + k.diff(s_y)*s_y)
    def B_bar(k):
        return sp.exp(a_bar)*(k.diff(s_xbar)*s_xbar + k.diff(s_ybar)*s_ybar)

    op_system = bpdes.operator_compilation.sympy_gram.compile_sympy(
        [A, B], [A_bar, B_bar],
        kernel,
        symbols,
        parallel=False
    )

    tmp_a = np.asfortranarray(np.random.normal(size=(500,5)))
    tmp_b = np.asfortranarray(np.random.normal(size=(500,5)))
    fa = np.asfortranarray(np.array([1.0, 1.0]))

    sol_1 = hydrocyclone.op_wrapper.Id_Id(tmp_a, tmp_b, fa)
    sol_2 = op_system[()](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)

    sol_1 = hydrocyclone.op_wrapper.Id_A(tmp_a, tmp_b, fa)
    sol_2 = op_system[(A_bar, )](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)

    sol_1 = hydrocyclone.op_wrapper.Id_B(tmp_a, tmp_b, fa)
    sol_2 = op_system[(B_bar, )](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)

    sol_1 = hydrocyclone.op_wrapper.A_A(tmp_a, tmp_b, fa)
    sol_2 = op_system[(A, A_bar)](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)

    sol_1 = hydrocyclone.op_wrapper.A_B(tmp_a, tmp_b, fa)
    sol_2 = op_system[(A, B_bar)](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)

    sol_1 = hydrocyclone.op_wrapper.B_B(tmp_a, tmp_b, fa)
    sol_2 = op_system[(B, B_bar)](tmp_a, tmp_b, fa)
    np.testing.assert_almost_equal(sol_1, sol_2)