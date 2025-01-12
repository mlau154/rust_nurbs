import numpy as np

from rust_nurbs import *


def test_bernstein_poly():
    """
    Evaluates the Bernstein polynomial and ensures that the
    output is a float value
    """
    B = bernstein_poly(5, 2, 0.3)
    assert isinstance(B, float)


def test_bezier_curve_eval():
    """
    Evaluates sample 2-D and 3-D Bézier curves at a point and ensures
    that the number of dimensions in the evaluated point is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    curve_point = np.array(bezier_curve_eval(p, 0.3))
    assert curve_point.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(bezier_curve_eval(p, 0.1))
    assert curve_point.shape == (3,)


def test_bezier_curve_dcdt():
    """
    Evaluates sample 2-D and 3-D Bézier curve first derivatives at a point and ensures
    that the number of dimensions in the evaluated derivative is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    first_deriv = np.array(bezier_curve_dcdt(p, 0.3))
    assert first_deriv.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bezier_curve_dcdt(p, 0.1))
    assert first_deriv.shape == (3,)


def test_bezier_curve_d2cdt2():
    """
    Evaluates sample 2-D and 3-D Bézier curve second derivatives at a point and ensures
    that the number of dimensions in the evaluated derivative is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    second_deriv = np.array(bezier_curve_d2cdt2(p, 0.3))
    assert second_deriv.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bezier_curve_d2cdt2(p, 0.1))
    assert second_deriv.shape == (3,)


def test_bezier_curve_eval_grid():
    """
    Evaluates sample 2-D and 3-D Bézier curves along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    curve_point = np.array(bezier_curve_eval_grid(p, 50))
    assert curve_point.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(bezier_curve_eval_grid(p, 50))
    assert curve_point.shape == (50, 3)


def test_bezier_curve_dcdt_grid():
    """
    Evaluates sample 2-D and 3-D Bézier curve first derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    first_deriv = np.array(bezier_curve_dcdt_grid(p, 50))
    assert first_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bezier_curve_dcdt_grid(p, 50))
    assert first_deriv.shape == (50, 3)


def test_bezier_curve_d2cdt2_grid():
    """
    Evaluates sample 2-D and 3-D Bézier curve second derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    second_deriv = np.array(bezier_curve_d2cdt2_grid(p, 50))
    assert second_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bezier_curve_d2cdt2_grid(p, 50))
    assert second_deriv.shape == (50, 3)


def test_bezier_curve_eval_tvec():
    """
    Evaluates sample 2-D and 3-D Bézier curves along a vector of :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    curve_point = np.array(bezier_curve_eval_tvec(p, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(bezier_curve_eval_tvec(p, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 3)


def test_bezier_curve_dcdt_tvec():
    """
    Evaluates sample 2-D and 3-D Bézier curve first derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    first_deriv = np.array(bezier_curve_dcdt_tvec(p, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bezier_curve_dcdt_tvec(p, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 3)


def test_bezier_curve_d2cdt2_tvec():
    """
    Evaluates sample 2-D and 3-D Bézier curve second derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    second_deriv = np.array(bezier_curve_d2cdt2_tvec(p, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bezier_curve_d2cdt2_tvec(p, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 3)


def test_bezier_surf_eval():
    """
    Evaluates a 1x3 Bézier surface at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated point 
    is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_point = np.array(bezier_surf_eval(p, 0.3, 0.8))
    assert surf_point.shape == (3,)


def test_bezier_surf_dsdu():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`u` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_deriv = np.array(bezier_surf_dsdu(p, 0.3, 0.8))
    assert first_deriv.shape == (3,)


def test_bezier_surf_dsdv():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`v` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_deriv = np.array(bezier_surf_dsdv(p, 0.3, 0.8))
    assert first_deriv.shape == (3,)


def test_bezier_surf_d2sdu2():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`u` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_deriv = np.array(bezier_surf_d2sdu2(p, 0.3, 0.8))
    assert second_deriv.shape == (3,)


def test_bezier_surf_d2sdv2():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`v` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_deriv = np.array(bezier_surf_d2sdv2(p, 0.3, 0.8))
    assert second_deriv.shape == (3,)


def test_bezier_surf_eval_iso_u():
    """
    Evaluates a 1x3 Bézier surface along a :math:`u`-isoparametric curve
    and ensures that the number of dimensions in the evaluated point 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_points = np.array(bezier_surf_eval_iso_u(p, 0.4, 15))
    assert surf_points.shape == (15, 3)


def test_bezier_surf_eval_iso_v():
    """
    Evaluates a 1x3 Bézier surface along a :math:`v`-isoparametric curve
    and ensures that the number of dimensions in the evaluated point 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_points = np.array(bezier_surf_eval_iso_v(p, 25, 0.6))
    assert surf_points.shape == (25, 3)


def test_bezier_surf_dsdu_iso_u():
    """
    Evaluates the first derivative w.r.t. :math:`u` on a 1x3 Bézier surface 
    along a :math:`u`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdu_iso_u(p, 0.4, 15))
    assert first_derivs.shape == (15, 3)


def test_bezier_surf_dsdu_iso_v():
    """
    Evaluates the first derivative w.r.t. :math:`u` on a 1x3 Bézier surface 
    along a :math:`v`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdu_iso_v(p, 25, 0.6))
    assert first_derivs.shape == (25, 3)


def test_bezier_surf_dsdv_iso_u():
    """
    Evaluates the first derivative w.r.t. :math:`v` on a 1x3 Bézier surface 
    along a :math:`u`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdv_iso_u(p, 0.4, 15))
    assert first_derivs.shape == (15, 3)


def test_bezier_surf_dsdv_iso_v():
    """
    Evaluates the first derivative w.r.t. :math:`v` on a 1x3 Bézier surface 
    along a :math:`v`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdv_iso_v(p, 25, 0.6))
    assert first_derivs.shape == (25, 3)


def test_bezier_surf_d2sdu2_iso_u():
    """
    Evaluates the second derivative w.r.t. :math:`u` on a 1x3 Bézier surface 
    along a :math:`u`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdu2_iso_u(p, 0.4, 15))
    assert second_derivs.shape == (15, 3)


def test_bezier_surf_d2sdu2_iso_v():
    """
    Evaluates the second derivative w.r.t. :math:`u` on a 1x3 Bézier surface 
    along a :math:`v`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdu2_iso_v(p, 25, 0.6))
    assert second_derivs.shape == (25, 3)


def test_bezier_surf_d2sdv2_iso_u():
    """
    Evaluates the second derivative w.r.t. :math:`v` on a 1x3 Bézier surface 
    along a :math:`u`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdv2_iso_u(p, 0.4, 15))
    assert second_derivs.shape == (15, 3)


def test_bezier_surf_d2sdv2_iso_v():
    """
    Evaluates the second derivative w.r.t. :math:`v` on a 1x3 Bézier surface 
    along a :math:`v`-isoparametric curve
    and ensures that the number of dimensions in the evaluated derivative 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdv2_iso_v(p, 25, 0.6))
    assert second_derivs.shape == (25, 3)


def test_bezier_surf_eval_grid():
    """
    Evaluates a 1x3 Bézier surface on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_points = np.array(bezier_surf_eval_grid(p, 25, 15))
    assert surf_points.shape == (25, 15, 3)


def test_bezier_surf_dsdu_grid():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`u` on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdu_grid(p, 25, 15))
    assert first_derivs.shape == (25, 15, 3)


def test_bezier_surf_dsdv_grid():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`v` on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    first_derivs = np.array(bezier_surf_dsdv_grid(p, 25, 15))
    assert first_derivs.shape == (25, 15, 3)


def test_bezier_surf_d2sdu2_grid():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`u` on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdu2_grid(p, 25, 15))
    assert second_derivs.shape == (25, 15, 3)


def test_bezier_surf_d2sdv2_grid():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`v` on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    second_derivs = np.array(bezier_surf_d2sdv2_grid(p, 25, 15))
    assert second_derivs.shape == (25, 15, 3)


def test_bezier_surf_eval_uvvecs():
    """
    Evaluates a 1x3 Bézier surface at several :math:`(u,v)` pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    u = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    v = np.array([0.0, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 1.0])
    surf_points = np.array(bezier_surf_eval_uvvecs(p, u, v))
    assert surf_points.shape == (u.shape[0], v.shape[0], 3)


def test_bezier_surf_dsdu_uvvecs():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`u` at several :math:`(u,v)` pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    u = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    v = np.array([0.0, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 1.0])
    first_derivs = np.array(bezier_surf_dsdu_uvvecs(p, u, v))
    assert first_derivs.shape == (u.shape[0], v.shape[0], 3)


def test_bezier_surf_dsdv_uvvecs():
    """
    Evaluates a 1x3 Bézier surface first derivative with respect to :math:`v` at several :math:`(u,v)` pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    u = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    v = np.array([0.0, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 1.0])
    first_derivs = np.array(bezier_surf_dsdv_uvvecs(p, u, v))
    assert first_derivs.shape == (u.shape[0], v.shape[0], 3)


def test_bezier_surf_d2sdu2_uvvecs():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`u` at several :math:`(u,v)` pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    u = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    v = np.array([0.0, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 1.0])
    second_derivs = np.array(bezier_surf_d2sdu2_uvvecs(p, u, v))
    assert second_derivs.shape == (u.shape[0], v.shape[0], 3)


def test_bezier_surf_d2sdv2_uvvecs():
    """
    Evaluates a 1x3 Bézier surface second derivative with respect to :math:`v` at several :math:`(u,v)` pairs
    and ensures that the number of dimensions in the evaluated derivative array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    u = np.array([0.0, 0.2, 0.3, 0.7, 1.0])
    v = np.array([0.0, 0.01, 0.05, 0.6, 0.7, 0.8, 0.9, 1.0])
    second_derivs = np.array(bezier_surf_d2sdv2_uvvecs(p, u, v))
    assert second_derivs.shape == (u.shape[0], v.shape[0], 3)


def test_rational_bezier_curve_eval():
    """
    Evaluates sample 2-D and 3-D rational Bézier curves at a point and ensures
    that the number of dimensions in the evaluated point is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([
        1.0,
        0.7,
        1.2,
        1.0
    ])
    curve_point = np.array(rational_bezier_curve_eval(p, w, 0.3))
    assert curve_point.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    w = np.array([
        1.0,
        0.8,
        1.1,
        1.0
    ])
    curve_point = np.array(rational_bezier_curve_eval(p, w, 0.1))
    assert curve_point.shape == (3,)


def test_rational_bezier_dcdt():
    """
    Generates a unit quarter circle and ensures that the slope is correct at a point
    """
    p = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    w = np.array([
        1.0, 
        1 / np.sqrt(2.0), 
        1.0
    ])
    curve_point = np.array(rational_bezier_curve_eval(p, w, 0.3))
    first_deriv = np.array(rational_bezier_curve_dcdt(p, w, 0.3))
    assert first_deriv.shape == (2,)
    assert np.isclose(first_deriv[1] / first_deriv[0], -curve_point[0] / curve_point[1])


def test_rational_bezier_dc2dt2():
    """
    Generates a unit quarter circle and confirms that the radius of curvature is equal to 1.0
    by using both the first and second derivative calculations
    """
    p = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    w = np.array([
        1.0, 
        1 / np.sqrt(2.0), 
        1.0
    ])
    t = 0.7
    first_deriv = np.array(rational_bezier_curve_dcdt(p, w, t))
    second_deriv = np.array(rational_bezier_curve_d2cdt2(p, w, t))
    assert second_deriv.shape == (2,)
    k = abs(first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]) / (first_deriv[0]**2 + first_deriv[1]**2) ** 1.5
    assert np.isclose(k, 1.0)


def test_rational_bezier_curve_eval_grid():
    """
    Evaluates sample 2-D and 3-D rational Bézier curves along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    curve_point = np.array(rational_bezier_curve_eval_grid(p, w, 50))
    assert curve_point.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(rational_bezier_curve_eval_grid(p, w, 50))
    assert curve_point.shape == (50, 3)


def test_rational_bezier_curve_dcdt_grid():
    """
    Evaluates sample 2-D and 3-D rational Bézier curve first derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    first_deriv = np.array(rational_bezier_curve_dcdt_grid(p, w, 50))
    assert first_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(rational_bezier_curve_dcdt_grid(p, w, 50))
    assert first_deriv.shape == (50, 3)


def test_rational_bezier_curve_d2cdt2_grid():
    """
    Evaluates sample 2-D and 3-D rational Bézier curve second derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    second_deriv = np.array(rational_bezier_curve_d2cdt2_grid(p, w, 50))
    assert second_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(rational_bezier_curve_d2cdt2_grid(p, w, 50))
    assert second_deriv.shape == (50, 3)


def test_rational_bezier_curve_eval_tvec():
    """
    Evaluates sample 2-D and 3-D rational Bézier curves along a vector of :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    curve_point = np.array(rational_bezier_curve_eval_tvec(p, w, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(rational_bezier_curve_eval_tvec(p, w, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 3)


def test_rational_bezier_curve_dcdt_tvec():
    """
    Evaluates sample 2-D and 3-D rational Bézier curve first derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    first_deriv = np.array(rational_bezier_curve_dcdt_tvec(p, w, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(rational_bezier_curve_dcdt_tvec(p, w, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 3)


def test_rational_bezier_curve_d2cdt2_tvec():
    """
    Evaluates sample 2-D and 3-D rational Bézier curve second derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    second_deriv = np.array(rational_bezier_curve_d2cdt2_tvec(p, w, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(rational_bezier_curve_d2cdt2_tvec(p, w, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 3)


def test_rational_bezier_surf_eval():
    """
    Evaluates a 2x3 rational Bézier surface at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated point 
    is correct. Note that we must have p.shape[:2] == w.shape or
    an error will be thrown.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    w = np.array([
        [1.0, 0.9, 0.7, 1.0],
        [1.0, 1.5, 0.6, 1.0],
        [1.0, 0.7, 1.1, 1.0]
    ])
    assert p.shape[:2] == w.shape
    surf_point = np.array(rational_bezier_surf_eval(p, w, 0.3, 0.4))
    assert surf_point.shape == (3,)


def test_rational_bezier_surf_eval_grid():
    """
    Evaluates a 2x3 rational Bézier surface on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct. Note that we must have p.shape[:2] == w.shape or
    an error will be thrown.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    w = np.array([
        [1.0, 0.9, 0.9, 1.0],
        [1.0, 1.2, 0.8, 1.0],
        [1.0, 1.7, 1.1, 1.0]
    ])
    assert p.shape[:2] == w.shape
    surf_point = np.array(rational_bezier_surf_eval_grid(p, w, 25, 15))
    assert surf_point.shape == (25, 15, 3)


def test_bspline_curve_eval():
    """
    Evaluates sample uniform 2-D and 3-D cubic B-spline curves at a point and 
    ensures that the number of dimensions in the evaluated point is correct.
    The knot vector is uniform because all the internal knots create
    a linear spacing between the starting and ending knots. Additionally,
    we can verify that the degree is 3 because 
    ``q = len(k) - len(p) - 1 = 10 - 6 - 1 = 3``
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.2, 0.1],
        [0.4, 0.2],
        [0.6, 0.1],
        [0.8, 0.0]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(bspline_curve_eval(p, k, 0.7))
    assert curve_point.shape == (2,)
    assert len(k) - len(p) - 1 == 3  # Curve degree

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        [0.2, 0.1, 0.7],
        [0.4, 0.2, 0.6],
        [0.6, 0.1, 0.4],
        [0.8, 0.0, 0.2]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(bspline_curve_eval(p, k, 0.2))
    assert curve_point.shape == (3,)
    assert len(k) - len(p) - 1 == 3  # Curve degree


def test_bspline_curve_dcdt():
    """
    Evaluates the first derivative w.r.t. :math:`t` of sample uniform 2-D 
    and 3-D cubic B-spline curves at a point and 
    ensures that the number of dimensions in the evaluated point is correct.
    The knot vector is uniform because all the internal knots create
    a linear spacing between the starting and ending knots. Additionally,
    we can verify that the degree is 3 because 
    ``q = len(k) - len(p) - 1 = 10 - 6 - 1 = 3``
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.2, 0.1],
        [0.4, 0.2],
        [0.6, 0.1],
        [0.8, 0.0]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    first_deriv = np.array(bspline_curve_dcdt(p, k, 0.7))
    assert first_deriv.shape == (2,)
    assert len(k) - len(p) - 1 == 3  # Curve degree

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        [0.2, 0.1, 0.7],
        [0.4, 0.2, 0.6],
        [0.6, 0.1, 0.4],
        [0.8, 0.0, 0.2]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(bspline_curve_dcdt(p, k, 0.2))
    assert curve_point.shape == (3,)
    assert len(k) - len(p) - 1 == 3  # Curve degree


def test_bspline_curve_dc2dt2():
    """
    Evaluates the second derivative w.r.t. :math:`t` of sample uniform 2-D 
    and 3-D cubic B-spline curves at a point and 
    ensures that the number of dimensions in the evaluated point is correct.
    The knot vector is uniform because all the internal knots create
    a linear spacing between the starting and ending knots. Additionally,
    we can verify that the degree is 3 because 
    ``q = len(k) - len(p) - 1 = 10 - 6 - 1 = 3``
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.2, 0.1],
        [0.4, 0.2],
        [0.6, 0.1],
        [0.8, 0.0]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    first_deriv = np.array(bspline_curve_d2cdt2(p, k, 0.7))
    assert first_deriv.shape == (2,)
    assert len(k) - len(p) - 1 == 3  # Curve degree

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        [0.2, 0.1, 0.7],
        [0.4, 0.2, 0.6],
        [0.6, 0.1, 0.4],
        [0.8, 0.0, 0.2]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(bspline_curve_d2cdt2(p, k, 0.2))
    assert curve_point.shape == (3,)
    assert len(k) - len(p) - 1 == 3  # Curve degree


def test_bspline_surf_eval():
    """
    Evaluates a 1x2 B-spline surface at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated point 
    is correct.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    surf_point = np.array(bspline_surf_eval(p, ku, kv, 0.0, 0.9))
    assert surf_point.shape == (3,)
    assert len(ku) - len(p) - 1 == 1  # Degree in the u-direction (q)
    assert len(kv) - len(p[0]) - 1 == 2  # Degree in the v-direction (r)


def test_bspline_surf_dsdu():
    """
    Evaluates a 1x2 B-spline surface first derivative with respect to :math:`u` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    first_deriv = np.array(bspline_surf_dsdu(p, ku, kv, 0.3, 0.8))
    assert first_deriv.shape == (3,)


def test_bspline_surf_dsdv():
    """
    Evaluates a 1x2 B-spline surface first derivative with respect to :math:`v` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    first_deriv = np.array(bspline_surf_dsdv(p, ku, kv, 0.3, 0.8))
    assert first_deriv.shape == (3,)


def test_bspline_surf_d2sdu2():
    """
    Evaluates a 1x2 B-spline surface derivative w.r.t. :math:`u` and ensures that the derivative is ZERO.
    Also evaluates a 2x2 B-spline surface surface second derivative with respect to :math:`u` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    second_deriv = np.array(bspline_surf_d2sdu2(p, ku, kv, 0.3, 0.8))
    assert second_deriv.shape == (3,)
    assert all([d == 0.0 for d in second_deriv])

    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    second_deriv = np.array(bspline_surf_d2sdu2(p, ku, kv, 0.3, 0.8))
    assert second_deriv.shape == (3,)


def test_bspline_surf_d2sdv2():
    """
    Evaluates both a 1x2 B-spline surface and a 2x2 B-spline surface surface second 
    derivative with respect to :math:`v` at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated derivative is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    second_deriv = np.array(bspline_surf_d2sdu2(p, ku, kv, 0.3, 0.8))
    assert second_deriv.shape == (3,)
    assert all([d == 0.0 for d in second_deriv])

    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    second_deriv = np.array(bspline_surf_d2sdv2(p, ku, kv, 0.3, 0.8))
    assert second_deriv.shape == (3,)


def test_bspline_surf_eval_grid():
    """
    Evaluates a 1x2 B-spline surface on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    surf_point = np.array(bspline_surf_eval_grid(p, ku, kv, 25, 15))
    assert surf_point.shape == (25, 15, 3)
    assert len(ku) - len(p) - 1 == 1  # Degree in the u-direction (q)
    assert len(kv) - len(p[0]) - 1 == 2  # Degree in the v-direction (r)


def test_nurbs_curve_eval():
    """
    Evaluates sample non-uniform 2-D and 3-D quintic B-spline curves at a point and 
    ensures that the number of dimensions in the evaluated point is correct.
    The knot vector is non-uniform because the internal knots do not create a
    a linear spacing between the starting and ending knots. Additionally,
    we can verify that the degree is 5 because 
    ``q = len(k) - len(p) - 1 = 15 - 9 - 1 = 5``
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.2, 0.1],
        [0.4, 0.2],
        [0.6, 0.1],
        [0.8, 0.0],
        [1.0, 0.3],
        [0.8, 0.1],
        [0.6, 0.3]
    ])
    w = np.array([
        1.0,
        0.7,
        0.9,
        0.8,
        1.2,
        1.0,
        1.1,
        1.0,
        1.0
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(nurbs_curve_eval(p, w, k, 0.7))
    assert curve_point.shape == (2,)
    assert len(k) - len(p) - 1 == 5  # Curve degree

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.3],
        [0.2, 0.1, 0.7],
        [0.4, 0.2, 0.6],
        [0.6, 0.1, 0.4],
        [0.8, 0.0, 0.2],
        [0.7, 0.2, 0.3],
        [1.0, 0.3, 0.6],
        [1.1, 0.2, 0.3]
    ])
    w = np.array([
        1.0,
        0.9,
        0.4,
        0.5,
        1.2,
        1.0,
        1.1,
        1.0,
        1.0
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(nurbs_curve_eval(p, w, k, 0.2))
    assert curve_point.shape == (3,)
    assert len(k) - len(p) - 1 == 5  # Curve degree


def test_nurbs_curve_dcdt():
    """
    Generates a quarter circle and ensures that the slope is correct at several points
    """
    p = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0]
    ])
    w = np.array([
        1.0, 
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
    ])
    k = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])

    # Try a point in QI
    t = 0.1
    curve_point = np.array(nurbs_curve_eval(p, w, k, t))
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    assert first_deriv.shape == (2,)
    assert np.isclose(first_deriv[1] / first_deriv[0], -curve_point[0] / curve_point[1])

    # Try a point in QII
    t = 0.4
    curve_point = np.array(nurbs_curve_eval(p, w, k, t))
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    assert first_deriv.shape == (2,)
    assert np.isclose(first_deriv[1] / first_deriv[0], -curve_point[0] / curve_point[1])

    # Try a point in QIII
    t = 0.6
    curve_point = np.array(nurbs_curve_eval(p, w, k, t))
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    assert first_deriv.shape == (2,)
    assert np.isclose(first_deriv[1] / first_deriv[0], -curve_point[0] / curve_point[1])

    # Try a point in QIV
    t = 0.9
    curve_point = np.array(nurbs_curve_eval(p, w, k, t))
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    assert first_deriv.shape == (2,)
    assert np.isclose(first_deriv[1] / first_deriv[0], -curve_point[0] / curve_point[1])


def test_nurbs_curve_d2cdt2():
    """
    Generates a quarter circle and ensures that the slope is correct at several points
    """
    p = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0]
    ])
    w = np.array([
        1.0, 
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
        1 / np.sqrt(2.0), 
        1.0,
    ])
    k = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])

    # Try a point in QI
    t = 0.1
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    second_deriv = np.array(nurbs_curve_d2cdt2(p, w, k, t))
    assert first_deriv.shape == (2,)
    kappa = abs(first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]) / (first_deriv[0]**2 + first_deriv[1]**2) ** 1.5
    assert np.isclose(kappa, 1.0)

    # Try a point in QII
    t = 0.4
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    second_deriv = np.array(nurbs_curve_d2cdt2(p, w, k, t))
    assert first_deriv.shape == (2,)
    kappa = abs(first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]) / (first_deriv[0]**2 + first_deriv[1]**2) ** 1.5
    assert np.isclose(kappa, 1.0)

    # Try a point in QIII
    t = 0.6
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    second_deriv = np.array(nurbs_curve_d2cdt2(p, w, k, t))
    assert first_deriv.shape == (2,)
    kappa = abs(first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]) / (first_deriv[0]**2 + first_deriv[1]**2) ** 1.5
    assert np.isclose(kappa, 1.0)

    # Try a point in QIV
    t = 0.9
    first_deriv = np.array(nurbs_curve_dcdt(p, w, k, t))
    second_deriv = np.array(nurbs_curve_d2cdt2(p, w, k, t))
    assert first_deriv.shape == (2,)
    kappa = abs(first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]) / (first_deriv[0]**2 + first_deriv[1]**2) ** 1.5
    assert np.isclose(kappa, 1.0)

def test_nurbs_curve_eval_grid():
    """
    Evaluates sample 2-D and 3-D NURBS curves along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(nurbs_curve_eval_grid(p, w, k, 50))
    assert curve_point.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(nurbs_curve_eval_grid(p, w, k, 50))
    assert curve_point.shape == (50, 3)


def test_nurbs_curve_dcdt_grid():
    """
    Evaluates sample 2-D and 3-D NURBS curve first derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    first_deriv = np.array(nurbs_curve_dcdt_grid(p, w, k, 50))
    assert first_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(nurbs_curve_dcdt_grid(p, w, k, 50))
    assert first_deriv.shape == (50, 3)


def test_nurbs_curve_d2cdt2_grid():
    """
    Evaluates sample 2-D and 3-D NURBS curve second derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    second_deriv = np.array(nurbs_curve_d2cdt2_grid(p, w, k, 50))
    assert second_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(nurbs_curve_d2cdt2_grid(p, w, k, 50))
    assert second_deriv.shape == (50, 3)


def test_nurbs_curve_eval_tvec():
    """
    Evaluates sample 2-D and 3-D NURBS curves along a vector of :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    curve_point = np.array(nurbs_curve_eval_tvec(p, w, k, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(nurbs_curve_eval_tvec(p, w, k, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 3)


def test_nurbs_curve_dcdt_tvec():
    """
    Evaluates sample 2-D and 3-D NURBS curve first derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    first_deriv = np.array(nurbs_curve_dcdt_tvec(p, w, k, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(nurbs_curve_dcdt_tvec(p, w, k, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 3)


def test_nurbs_curve_d2cdt2_tvec():
    """
    Evaluates sample 2-D and 3-D NURBS curve second derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    w = np.array([1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    second_deriv = np.array(nurbs_curve_d2cdt2_tvec(p, w, k, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(nurbs_curve_d2cdt2_tvec(p, w, k, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 3)


def test_bspline_curve_eval_grid():
    """
    Evaluates sample 2-D and 3-D B-spline curves along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    curve_point = np.array(bspline_curve_eval_grid(p, k, 50))
    assert curve_point.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(bspline_curve_eval_grid(p, k, 50))
    assert curve_point.shape == (50, 3)


def test_bspline_curve_dcdt_grid():
    """
    Evaluates sample 2-D and 3-D B-spline curve first derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    first_deriv = np.array(bspline_curve_dcdt_grid(p, k, 50))
    assert first_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bspline_curve_dcdt_grid(p, k, 50))
    assert first_deriv.shape == (50, 3)


def test_bspline_curve_d2cdt2_grid():
    """
    Evaluates sample 2-D and 3-D B-spline curve second derivatives along a grid with 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    second_deriv = np.array(bspline_curve_d2cdt2_grid(p, k, 50))
    assert second_deriv.shape == (50, 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bspline_curve_d2cdt2_grid(p, k, 50))
    assert second_deriv.shape == (50, 3)


def test_bspline_curve_eval_tvec():
    """
    Evaluates sample 2-D and 3-D B-spline curves along a vector of :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    curve_point = np.array(bspline_curve_eval_tvec(p, k, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    curve_point = np.array(bspline_curve_eval_tvec(p, k, t_vec))
    assert curve_point.shape == (t_vec.shape[0], 3)


def test_bspline_curve_dcdt_tvec():
    """
    Evaluates sample 2-D and 3-D B-spline curve first derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    first_deriv = np.array(bspline_curve_dcdt_tvec(p, k, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bspline_curve_dcdt_tvec(p, k, t_vec))
    assert first_deriv.shape == (t_vec.shape[0], 3)


def test_bspline_curve_d2cdt2_tvec():
    """
    Evaluates sample 2-D and 3-D B-spline curve second derivatives along a vector of 50 :math:`t`-values and ensures
    that the shape of the output array is correct
    """
    # 2-D case
    p = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.5],
        [0.4, 0.2],
        [0.7, 0.1],
        [1.0, 0.1]
    ])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0])
    t_vec = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0])
    second_deriv = np.array(bspline_curve_d2cdt2_tvec(p, k, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 2)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.6],
        [0.7, 0.1, 0.5],
        [0.8, 0.2, 0.4],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bspline_curve_d2cdt2_tvec(p, k, t_vec))
    assert second_deriv.shape == (t_vec.shape[0], 3)


def test_nurbs_surf_eval():
    """
    Evaluates a 1x2 NURBS surface at a single (u,v) pair 
    and ensures that the number of dimensions in the evaluated point 
    is correct.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    w = np.array([
        [1.0, 0.9, 0.9, 1.0],
        [1.0, 1.2, 0.8, 1.0],
        [1.0, 1.7, 1.1, 1.0]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    assert p.shape[:2] == w.shape
    surf_point = np.array(nurbs_surf_eval(p, w, ku, kv, 0.0, 0.9))
    assert surf_point.shape == (3,)
    assert len(ku) - len(p) - 1 == 1  # Degree in the u-direction (q)
    assert len(kv) - len(p[0]) - 1 == 2  # Degree in the v-direction (r)


def test_nurbs_surf_eval_grid():
    """
    Evaluates a 1x2 NURBS surface on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct.
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]],
        [[0.0, 0.1, 2.0], [0.5, 0.3, 2.0], [0.5, -0.3, 2.0], [1.2, 0.3, 2.0]]
    ])
    w = np.array([
        [1.0, 0.9, 0.7, 1.0],
        [1.0, 1.5, 0.6, 1.0],
        [1.0, 0.7, 1.1, 1.0]
    ])
    ku = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    kv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    assert p.shape[:2] == w.shape
    surf_point = np.array(nurbs_surf_eval_grid(p, w, ku, kv, 25, 15))
    assert surf_point.shape == (25, 15, 3)
    assert len(ku) - len(p) - 1 == 1  # Degree in the u-direction (q)
    assert len(kv) - len(p[0]) - 1 == 2  # Degree in the v-direction (r)
