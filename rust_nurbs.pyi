"""
Evaluation of NURBS objects in Python (implemented in Rust)
"""
from typing import Iterable, List

def bernstein_poly(n: int, i: int, t: float) -> float:
    r"""
    Evaluates the Bernstein polynomial at a single :math:`t`-value. The Bernstein polynomial is given by

    .. math::

        B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

    Parameters
    ----------
    n: int
        Degree of the polynomial
    i: int
        Index
    t: float
        Parameter value :math:`t` at which to evaluate
    
    Returns
    -------
    float
        Value of the Bernstein polynomial at :math:`t`
    """

def bezier_curve_eval(p: Iterable[Iterable[float]], t: float) -> List[float]:
    r"""
    Evaluates a Bézier curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \sum\limits_{i=0}^n B_{i,n}(t) \mathbf{P}_i

    where :math:`B_{i,n}(t)` is the Bernstein polynomial.

    Parameters
    ----------
    p: Iterable[Iterable[float]]
        2-D list or array of control points where the inner dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    t: float
        Parameter value :math:`t` at which to evaluate
    
    Returns
    -------
    List[float]
        Value of the Bézier curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """

def bezier_surf_eval(p: Iterable[Iterable[Iterable[float]]], u: float, v: float) -> List[float]:
    r"""
    Evaluates a Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m B_{i,n}(u) B_{j,m}(v) \mathbf{P}_{i,j}
    
    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface
    
    Returns
    -------
    List[float]
        Value of the Bézier surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """

def bezier_surf_eval_grid(p: Iterable[Iterable[Iterable[float]]], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` points 
    along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m B_{i,n}(u) B_{j,m}(v) \mathbf{P}_{i,j}
    
    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the Bézier surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually either ``2``, ``3``, or ``4``)
    """

def rational_bezier_curve_eval(p: Iterable[Iterable[float]], w: Iterable[float], t: float) -> List[float]:
    r"""
    Evaluates a rational Bézier curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \frac{\sum_{i=0}^n B_{i,n}(t) w_i \mathbf{P}_i}{\sum_{i=0}^n B_{i,n}(t) w_i}

    where :math:`B_{i,n}(t)` is the Bernstein polynomial.

    Parameters
    ----------
    p: Iterable[Iterable[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical 
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[float]
        1-D list or array of weights corresponding to each of control points. Must have length
        equal to the outer dimension of ``p``.
    t: float
        Parameter value :math:`t` at which to evaluate
    
    Returns
    -------
    List[float]
        Value of the rational Bézier curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """

def rational_bezier_surf_eval(p: Iterable[Iterable[Iterable[float]]], w: Iterable[Iterable[float]], u: float, v: float) -> List[float]:
    r"""
    Evaluates a rational Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j}}
    
    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[Iterable[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the rational Bézier surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """

def rational_bezier_surf_eval_grid(p: Iterable[Iterable[Iterable[float]]], w: Iterable[Iterable[float]], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a rational Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` points along a 
    linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j}}

    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[Iterable[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the rational Bézier surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """

def bspline_curve_eval(p: Iterable[Iterable[float]], k: Iterable[float], t: float) -> List[float]:
    r"""
    Evaluates a B-spline curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \sum\limits_{i=0}^n N_{i,q}(t) \mathbf{P}_i

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`, defined recursively as

    .. math::

        N_{i,q} = \frac{t - t_i}{t_{i+q} - t_i} N_{i,q-1}(t) + \frac{t_{i+q+1} - t}{t_{i+q+1} - t_{i+1}} N_{i+1, q-1}(t)

    with base case

    .. math::

        N_{i,0} = \begin{cases}
            1, & \text{if } t_i \leq t < t_{i+1} \text{ and } t_i < t_{i+1} \\
            0, & \text{otherwise}
        \end{cases}

    The degree of the B-spline is computed as ``q = k.len() - len(p) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical 
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    k: Iterable[float]
        1-D list or array of knots
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the B-spline curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """

def bspline_surf_eval(p: Iterable[Iterable[Iterable[float]]], ku: Iterable[float], kv: Iterable[float], u: float, v: float) -> List[float]:
    r"""
    Evaluates a B-spline surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m N_{i,q}(u) N_{j,r}(v) \mathbf{P}_{i,j}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    ku: Iterable[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: Iterable[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the B-spline surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """

def bspline_surf_eval_grid(p: Iterable[Iterable[Iterable[float]]], ku: Iterable[float], kv: Iterable[float], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a B-spline surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` 
    points along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m N_{i,q}(u) N_{j,r}(v) \mathbf{P}_{i,j}
    
    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    ku: Iterable[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: Iterable[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the B-spline surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """

def nurbs_curve_eval(p: Iterable[Iterable[float]], w: Iterable[float], k: Iterable[float], t: float) -> List[float]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) curve with :math:`n+1` control points at a 
    single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \frac{\sum_{i=0}^n N_{i,q}(t) w_i \mathbf{P}_i}{\sum_{i=0}^n N_{i,q}(t) w_i}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. 
    The degree of the B-spline is computed as ``q = len(k) - len(p) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical 
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[float]
        1-D list or array of weights corresponding to each of control points. Must have length
        equal to the outer dimension of ``p``.
    k: Iterable[float]
        1-D list or array of knots
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the NURBS curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """

def nurbs_surf_eval(p: Iterable[Iterable[Iterable[float]]], w: Iterable[Iterable[float]], ku: Iterable[float], kv: Iterable[float], u: float, v: float) -> List[float]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j}}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[Iterable[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    ku: Iterable[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: Iterable[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the NURBS surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """

def nurbs_surf_eval_grid(p: Iterable[Iterable[Iterable[float]]], w: Iterable[Iterable[float]], ku: Iterable[float], kv: Iterable[float], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` 
    points along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j}}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: Iterable[Iterable[Iterable[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: Iterable[Iterable[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    ku: Iterable[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: Iterable[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the NURBS surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """