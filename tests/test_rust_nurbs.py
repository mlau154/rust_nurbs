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


def test_bezier_curve_dCdt():
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
    first_deriv = np.array(bezier_curve_dCdt(p, 0.3))
    assert first_deriv.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    first_deriv = np.array(bezier_curve_dCdt(p, 0.1))
    assert first_deriv.shape == (3,)


def test_bezier_curve_d2Cdt2():
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
    second_deriv = np.array(bezier_curve_d2Cdt2(p, 0.3))
    assert second_deriv.shape == (2,)

    # 3-D case
    p = np.array([
        [0.0, 0.0, 0.1],
        [0.3, 0.5, 0.2],
        [0.7, 0.1, 0.5],
        [1.0, 0.1, 0.3]
    ])
    second_deriv = np.array(bezier_curve_d2Cdt2(p, 0.1))
    assert second_deriv.shape == (3,)


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


def test_bezier_surf_grid_eval():
    """
    Evaluates a 1x3 Bézier surface on a grid of (u,v) pairs
    and ensures that the number of dimensions in the evaluated point 
    array is correct
    """
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]],
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_point = np.array(bezier_surf_eval_grid(p, 25, 15))
    assert surf_point.shape == (25, 15, 3)


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
