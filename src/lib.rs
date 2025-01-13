use pyo3::prelude::*;
use num_integer::binomial;

fn bernstein_poly_rust(n: usize, i: usize, t: f64) -> f64 {
    if i > n {
        return 0.0;
    }
    return (binomial(n, i) as f64) * t.powf(i as f64) * (1.0 - t).powf((n - i) as f64);
}

#[pyfunction]
fn bernstein_poly(n: usize, i: usize, t: f64) -> PyResult<f64> {
    Ok(bernstein_poly_rust(n, i, t))
}

#[pyfunction]
fn bezier_curve_eval(p: Vec<Vec<f64>>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let dim = p[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly = bernstein_poly_rust(n, i, t);
        for j in 0..dim {
            evaluated_point[j] += p[i][j] * b_poly;
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bezier_curve_dcdt(p: Vec<Vec<f64>>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n {
        let b_poly = bernstein_poly_rust(n - 1, i, t);
        for j in 0..dim {
            evaluated_deriv[j] += float_n * (p[i + 1][j] - p[i][j]) * b_poly;
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_curve_d2cdt2(p: Vec<Vec<f64>>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n-1 {
        let b_poly = bernstein_poly_rust(n - 2, i, t);
        for j in 0..dim {
            evaluated_deriv[j] += float_n * (float_n + 1.0) * (p[i + 2][j] - 2.0 * p[i + 1][j] + p[i][j]) * b_poly;
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_curve_eval_grid(p: Vec<Vec<f64>>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t);
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * b_poly;
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_curve_dcdt_grid(p: Vec<Vec<f64>>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n {
            let b_poly = bernstein_poly_rust(n - 1, i, t);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_n * (p[i + 1][j] - p[i][j]) * b_poly;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_curve_d2cdt2_grid(p: Vec<Vec<f64>>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n-1 {
            let b_poly = bernstein_poly_rust(n - 2, i, t);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_n * (float_n + 1.0) * (p[i + 2][j] - 2.0 * p[i + 1][j] + p[i][j]) * b_poly;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_curve_eval_tvec(p: Vec<Vec<f64>>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let nt = t.len();
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t[t_idx]);
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * b_poly;
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_curve_dcdt_tvec(p: Vec<Vec<f64>>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let nt = t.len();
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n {
            let b_poly = bernstein_poly_rust(n - 1, i, t[t_idx]);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_n * (p[i + 1][j] - p[i][j]) * b_poly;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_curve_d2cdt2_tvec(p: Vec<Vec<f64>>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let nt = t.len();
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n-1 {
            let b_poly = bernstein_poly_rust(n - 2, i, t[t_idx]);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_n * (float_n + 1.0) * (p[i + 2][j] - 2.0 * p[i + 1][j] + p[i][j]) * b_poly;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_eval(p: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_u * b_poly_v;
            for k in 0..dim {
                evaluated_point[k] += p[i][j][k] * b_poly_prod;
            }
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bezier_surf_dsdu(p: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_du = float_n * (bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u));
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_du * b_poly_v;
            for k in 0..dim {
                evaluated_deriv[k] += p[i][j][k] * b_poly_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_surf_dsdv(p: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_dv = float_m * (bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v));
            let b_poly_prod = b_poly_u * b_poly_dv;
            for k in 0..dim {
                evaluated_deriv[k] += p[i][j][k] * b_poly_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_surf_d2sdu2(p: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_du = float_n * (float_n - 1.0) * (bernstein_poly_rust(n - 2, i - 2, u) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u) + bernstein_poly_rust(n - 2, i, u));
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_du * b_poly_v;
            for k in 0..dim {
                evaluated_deriv[k] += p[i][j][k] * b_poly_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_surf_d2sdv2(p: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_dv = float_m * (float_m - 1.0) * (bernstein_poly_rust(m - 2, j - 2, v) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v) + bernstein_poly_rust(m - 2, j, v));
            let b_poly_prod = b_poly_u * b_poly_dv;
            for k in 0..dim {
                evaluated_deriv[k] += p[i][j][k] * b_poly_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bezier_surf_eval_iso_u(p: Vec<Vec<Vec<f64>>>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_u * b_poly_v;
                for k in 0..dim {
                    evaluated_points[v_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_surf_eval_iso_v(p: Vec<Vec<Vec<f64>>>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_u * b_poly_v;
                for k in 0..dim {
                    evaluated_points[u_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_surf_dsdu_iso_u(p: Vec<Vec<Vec<f64>>>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_du = float_n * (bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u));
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_du * b_poly_v;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_dsdu_iso_v(p: Vec<Vec<Vec<f64>>>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_du = float_n * (bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u));
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_du * b_poly_v;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_dsdv_iso_u(p: Vec<Vec<Vec<f64>>>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_dv = float_m * (bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v));
                let b_poly_prod = b_poly_u * b_poly_dv;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_dsdv_iso_v(p: Vec<Vec<Vec<f64>>>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_dv = float_m * (bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v));
                let b_poly_prod = b_poly_u * b_poly_dv;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdu2_iso_u(p: Vec<Vec<Vec<f64>>>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_du = float_n * (float_n - 1.0) * (bernstein_poly_rust(n - 2, i - 2, u) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u) + bernstein_poly_rust(n - 2, i, u));
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_du * b_poly_v;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdu2_iso_v(p: Vec<Vec<Vec<f64>>>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_du = float_n * (float_n - 1.0) * (bernstein_poly_rust(n - 2, i - 2, u) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u) + bernstein_poly_rust(n - 2, i, u));
            for j in 0..m+1 {
                let b_poly_v = bernstein_poly_rust(m, j, v);
                let b_poly_prod = b_poly_du * b_poly_v;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdv2_iso_u(p: Vec<Vec<Vec<f64>>>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_dv = float_m * (float_m - 1.0) * (bernstein_poly_rust(m - 2, j - 2, v) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v) + bernstein_poly_rust(m - 2, j, v));
                let b_poly_prod = b_poly_u * b_poly_dv;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdv2_iso_v(p: Vec<Vec<Vec<f64>>>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            for j in 0..m+1 {
                let b_poly_dv = float_m * (float_m - 1.0) * (bernstein_poly_rust(m - 2, j - 2, v) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v) + bernstein_poly_rust(m - 2, j, v));
                let b_poly_prod = b_poly_u * b_poly_dv;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += p[i][j][k] * b_poly_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_eval_grid(p: Vec<Vec<Vec<f64>>>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u);
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v);
                    let b_poly_prod = b_poly_u * b_poly_v;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_surf_dsdu_grid(p: Vec<Vec<Vec<f64>>>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let b_poly_du = float_n * (bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u));
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v);
                    let b_poly_prod = b_poly_du * b_poly_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_dsdv_grid(p: Vec<Vec<Vec<f64>>>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u);
                for j in 0..m+1 {
                    let b_poly_dv = float_m * (bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v));
                    let b_poly_prod = b_poly_u * b_poly_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdu2_grid(p: Vec<Vec<Vec<f64>>>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let b_poly_du = float_n * (float_n - 1.0) * (bernstein_poly_rust(n - 2, i - 2, u) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u) + bernstein_poly_rust(n - 2, i, u));
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v);
                    let b_poly_prod = b_poly_du * b_poly_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdv2_grid(p: Vec<Vec<Vec<f64>>>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u);
                for j in 0..m+1 {
                    let b_poly_dv = float_m * (float_m - 1.0) * (bernstein_poly_rust(m - 2, j - 2, v) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v) + bernstein_poly_rust(m - 2, j, v));
                    let b_poly_prod = b_poly_u * b_poly_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_eval_uvvecs(p: Vec<Vec<Vec<f64>>>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let nu = u.len();
    let nv = v.len();
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u[u_idx]);
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v[v_idx]);
                    let b_poly_prod = b_poly_u * b_poly_v;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bezier_surf_dsdu_uvvecs(p: Vec<Vec<Vec<f64>>>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let nu = u.len();
    let nv = v.len();
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let b_poly_du = float_n * (bernstein_poly_rust(n - 1, i - 1, u[u_idx]) - bernstein_poly_rust(n - 1, i, u[u_idx]));
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v[v_idx]);
                    let b_poly_prod = b_poly_du * b_poly_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_dsdv_uvvecs(p: Vec<Vec<Vec<f64>>>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let nu = u.len();
    let nv = v.len();
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u[u_idx]);
                for j in 0..m+1 {
                    let b_poly_dv = float_m * (bernstein_poly_rust(m - 1, j - 1, v[v_idx]) - bernstein_poly_rust(m - 1, j, v[v_idx]));
                    let b_poly_prod = b_poly_u * b_poly_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdu2_uvvecs(p: Vec<Vec<Vec<f64>>>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let float_n = n as f64;
    let m = p[0].len() - 1;  // Degree in the v-direction
    let nu = u.len();
    let nv = v.len();
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let b_poly_du = float_n * (float_n - 1.0) * (bernstein_poly_rust(n - 2, i - 2, u[u_idx]) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u[u_idx]) + bernstein_poly_rust(n - 2, i, u[u_idx]));
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v[v_idx]);
                    let b_poly_prod = b_poly_du * b_poly_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bezier_surf_d2sdv2_uvvecs(p: Vec<Vec<Vec<f64>>>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let float_m = m as f64;
    let nu = u.len();
    let nv = v.len();
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u[u_idx]);
                for j in 0..m+1 {
                    let b_poly_dv = float_m * (float_m - 1.0) * (bernstein_poly_rust(m - 2, j - 2, v[v_idx]) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v[v_idx]) + bernstein_poly_rust(m - 2, j, v[v_idx]));
                    let b_poly_prod = b_poly_u * b_poly_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn rational_bezier_curve_eval(p: Vec<Vec<f64>>, w: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let dim = p[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly = bernstein_poly_rust(n, i, t);
        w_sum += w[i] * b_poly;
        for j in 0..dim {
            evaluated_point[j] += p[i][j] * w[i] * b_poly;
        }
    }
    for j in 0..dim {
        evaluated_point[j] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn rational_bezier_curve_dcdt(p: Vec<Vec<f64>>, w: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    let mut wp_sum: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly = bernstein_poly_rust(n, i, t);
        let b_poly_diff = bernstein_poly_rust(n - 1, i - 1, t) - bernstein_poly_rust(n - 1, i, t);
        w_sum += w[i] * b_poly;
        wp_sum += w[i] * b_poly_diff;
        for j in 0..dim {
            sum_1[j] += w[i] * p[i][j] * b_poly_diff;
            sum_2[j] += w[i] * p[i][j] * b_poly;
        }
    }
    for j in 0..dim {
        evaluated_deriv[j] = float_n * (sum_1[j] * w_sum - sum_2[j] * wp_sum) / (w_sum * w_sum);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_curve_d2cdt2(p: Vec<Vec<f64>>, w: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly_0 = bernstein_poly_rust(n, i, t);
        let b_poly_1 = bernstein_poly_rust(n - 1, i - 1, t) - bernstein_poly_rust(n - 1, i, t);
        let b_poly_2 = bernstein_poly_rust(n - 2, i - 2, t) - 2.0 * bernstein_poly_rust(n - 2, i - 1, t) + bernstein_poly_rust(n - 2, i, t);
        w_sum_0 += w[i] * b_poly_0;
        w_sum_1 += w[i] * b_poly_1;
        w_sum_2 += w[i] * b_poly_2;
        for j in 0..dim {
            sum_0[j] += w[i] * p[i][j] * b_poly_0;
            sum_1[j] += w[i] * p[i][j] * b_poly_1;
            sum_2[j] += w[i] * p[i][j] * b_poly_2;
        }
    }
    for j in 0..dim {
        evaluated_deriv[j] = (
            float_n * (float_n - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
            float_n * (float_n - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
            2.0 * float_n * float_n * sum_1[j] * w_sum_0 * w_sum_1 +
            2.0 * float_n * float_n * sum_0[j] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_curve_eval_grid(p: Vec<Vec<f64>>, w: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut w_sum: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t);
            w_sum += w[i] * b_poly;
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * w[i] * b_poly;
            }
        }
        for j in 0..dim {
            evaluated_points[t_idx][j] /= w_sum;
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn rational_bezier_curve_dcdt_grid(p: Vec<Vec<f64>>, w: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum: f64 = 0.0;
        let mut wp_sum: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t);
            let b_poly_diff = bernstein_poly_rust(n - 1, i - 1, t) - bernstein_poly_rust(n - 1, i, t);
            w_sum += w[i] * b_poly;
            wp_sum += w[i] * b_poly_diff;
            for j in 0..dim {
                sum_1[j] += w[i] * p[i][j] * b_poly_diff;
                sum_2[j] += w[i] * p[i][j] * b_poly;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = float_n * (sum_1[j] * w_sum - sum_2[j] * wp_sum) / (w_sum * w_sum);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn rational_bezier_curve_d2cdt2_grid(p: Vec<Vec<f64>>, w: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        let mut w_sum_2: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly_0 = bernstein_poly_rust(n, i, t);
            let b_poly_1 = bernstein_poly_rust(n - 1, i - 1, t) - bernstein_poly_rust(n - 1, i, t);
            let b_poly_2 = bernstein_poly_rust(n - 2, i - 2, t) - 2.0 * bernstein_poly_rust(n - 2, i - 1, t) + bernstein_poly_rust(n - 2, i, t);
            w_sum_0 += w[i] * b_poly_0;
            w_sum_1 += w[i] * b_poly_1;
            w_sum_2 += w[i] * b_poly_2;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * b_poly_0;
                sum_1[j] += w[i] * p[i][j] * b_poly_1;
                sum_2[j] += w[i] * p[i][j] * b_poly_2;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = (
                float_n * (float_n - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
                float_n * (float_n - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
                2.0 * float_n * float_n * sum_1[j] * w_sum_0 * w_sum_1 +
                2.0 * float_n * float_n * sum_0[j] * w_sum_1 * w_sum_1
            ) / w_sum_0.powf(3.0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn rational_bezier_curve_eval_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let nt = t.len();
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut w_sum: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t[t_idx]);
            w_sum += w[i] * b_poly;
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * w[i] * b_poly;
            }
        }
        for j in 0..dim {
            evaluated_points[t_idx][j] /= w_sum;
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn rational_bezier_curve_dcdt_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let nt = t.len();
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum: f64 = 0.0;
        let mut wp_sum: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly = bernstein_poly_rust(n, i, t[t_idx]);
            let b_poly_diff = bernstein_poly_rust(n - 1, i - 1, t[t_idx]) - bernstein_poly_rust(n - 1, i, t[t_idx]);
            w_sum += w[i] * b_poly;
            wp_sum += w[i] * b_poly_diff;
            for j in 0..dim {
                sum_1[j] += w[i] * p[i][j] * b_poly_diff;
                sum_2[j] += w[i] * p[i][j] * b_poly;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = float_n * (sum_1[j] * w_sum - sum_2[j] * wp_sum) / (w_sum * w_sum);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn rational_bezier_curve_d2cdt2_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;
    let nt = t.len();
    let float_n = n as f64;
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        let mut w_sum_2: f64 = 0.0;
        for i in 0..n+1 {
            let b_poly_0 = bernstein_poly_rust(n, i, t[t_idx]);
            let b_poly_1 = bernstein_poly_rust(n - 1, i - 1, t[t_idx]) - bernstein_poly_rust(n - 1, i, t[t_idx]);
            let b_poly_2 = bernstein_poly_rust(n - 2, i - 2, t[t_idx]) - 2.0 * bernstein_poly_rust(n - 2, i - 1, t[t_idx]) + bernstein_poly_rust(n - 2, i, t[t_idx]);
            w_sum_0 += w[i] * b_poly_0;
            w_sum_1 += w[i] * b_poly_1;
            w_sum_2 += w[i] * b_poly_2;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * b_poly_0;
                sum_1[j] += w[i] * p[i][j] * b_poly_1;
                sum_2[j] += w[i] * p[i][j] * b_poly_2;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = (
                float_n * (float_n - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
                float_n * (float_n - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
                2.0 * float_n * float_n * sum_1[j] * w_sum_0 * w_sum_1 +
                2.0 * float_n * float_n * sum_0[j] * w_sum_1 * w_sum_1
            ) / w_sum_0.powf(3.0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn rational_bezier_surf_eval(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_u * b_poly_v;
            w_sum += w[i][j] * b_poly_prod;
            for k in 0..dim {
                evaluated_point[k] += p[i][j][k] * w[i][j] * b_poly_prod;
            }
        }
    }
    for k in 0..dim {
        evaluated_point[k] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn rational_bezier_surf_dsdu(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let float_n = n as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        let b_poly_du = bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            w_sum_0 += w[i][j] * b_poly_u * b_poly_v;
            w_sum_1 += w[i][j] * b_poly_du * b_poly_v;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * b_poly_u * b_poly_v;
                sum_1[k] += w[i][j] * p[i][j][k] * b_poly_du * b_poly_v;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = float_n * (sum_1[k] * w_sum_0 - sum_0[k] * w_sum_1) / (w_sum_0 * w_sum_0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_surf_dsdv(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    for j in 0..m+1 {
        let b_poly_v = bernstein_poly_rust(m, j, v);
        let b_poly_dv = bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            w_sum_0 += w[i][j] * b_poly_v * b_poly_u;
            w_sum_1 += w[i][j] * b_poly_dv * b_poly_u;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * b_poly_v * b_poly_u;
                sum_1[k] += w[i][j] * p[i][j][k] * b_poly_dv * b_poly_u;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = float_m * (sum_1[k] * w_sum_0 - sum_0[k] * w_sum_1) / (w_sum_0 * w_sum_0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_surf_d2sdu2(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let float_n = n as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        let b_poly_du = bernstein_poly_rust(n - 1, i - 1, u) - bernstein_poly_rust(n - 1, i, u);
        let b_poly_d2u = bernstein_poly_rust(n - 2, i - 2, u) - 2.0 * bernstein_poly_rust(n - 2, i - 1, u) + bernstein_poly_rust(n - 2, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            w_sum_0 += w[i][j] * b_poly_u * b_poly_v;
            w_sum_1 += w[i][j] * b_poly_du * b_poly_v;
            w_sum_2 += w[i][j] * b_poly_d2u * b_poly_v;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * b_poly_u * b_poly_v;
                sum_1[k] += w[i][j] * p[i][j][k] * b_poly_du * b_poly_v;
                sum_2[k] += w[i][j] * p[i][j][k] * b_poly_d2u * b_poly_v;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = (
            float_n * (float_n - 1.0) * sum_2[k] * w_sum_0 * w_sum_0 - 
            float_n * (float_n - 1.0) * sum_0[k] * w_sum_0 * w_sum_2 -
            2.0 * float_n * float_n * sum_1[k] * w_sum_0 * w_sum_1 +
            2.0 * float_n * float_n * sum_0[k] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_surf_d2sdv2(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let float_m = m as f64;
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for j in 0..m+1 {
        let b_poly_v = bernstein_poly_rust(m, j, v);
        let b_poly_dv = bernstein_poly_rust(m - 1, j - 1, v) - bernstein_poly_rust(m - 1, j, v);
        let b_poly_d2v = bernstein_poly_rust(m - 2, j - 2, v) - 2.0 * bernstein_poly_rust(m - 2, j - 1, v) + bernstein_poly_rust(m - 2, j, v);
        for i in 0..n+1 {
            let b_poly_u = bernstein_poly_rust(n, i, u);
            w_sum_0 += w[i][j] * b_poly_v * b_poly_u;
            w_sum_1 += w[i][j] * b_poly_dv * b_poly_u;
            w_sum_2 += w[i][j] * b_poly_d2v * b_poly_u;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * b_poly_v * b_poly_u;
                sum_1[k] += w[i][j] * p[i][j][k] * b_poly_dv * b_poly_u;
                sum_2[k] += w[i][j] * p[i][j][k] * b_poly_d2v * b_poly_u;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = (
            float_m * (float_m - 1.0) * sum_2[k] * w_sum_0 * w_sum_0 - 
            float_m * (float_m - 1.0) * sum_0[k] * w_sum_0 * w_sum_2 -
            2.0 * float_m * float_m * sum_1[k] * w_sum_0 * w_sum_1 +
            2.0 * float_m * float_m * sum_0[k] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn rational_bezier_surf_eval_grid(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Degree in the u-direction
    let m = p[0].len() - 1;  // Degree in the v-direction
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            let mut w_sum: f64 = 0.0;
            for i in 0..n+1 {
                let b_poly_u = bernstein_poly_rust(n, i, u);
                for j in 0..m+1 {
                    let b_poly_v = bernstein_poly_rust(m, j, v);
                    let b_poly_prod = b_poly_u * b_poly_v;
                    w_sum += w[i][j] * b_poly_prod;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * w[i][j] * b_poly_prod;
                    }
                }
            }
            for k in 0..dim {
                evaluated_points[u_idx][v_idx][k] /= w_sum;
            }
        }
    }
    Ok(evaluated_points)
}

fn get_possible_span_indices(k: &[f64]) -> Vec<usize> {
    let mut possible_span_indices: Vec<usize> = Vec::new();
    let num_knots = k.len();
    for i in 0..num_knots-1 {
        if k[i] == k[i + 1] {
            continue;
        }
        possible_span_indices.push(i);
    }
    return possible_span_indices;
}

fn find_span(k: &[f64], possible_span_indices: &[usize], t: f64) -> usize {
    for &knot_span_idx in possible_span_indices {
        if k[knot_span_idx] <= t && t < k[knot_span_idx + 1] {
            return knot_span_idx;
        }
    }
    // If the parameter value is equal to the last knot, just return the last possible knot span index
    if t == k[k.len() - 1] {
        return possible_span_indices[possible_span_indices.len() - 1];
    }
    let k1: f64 = k[0];
    let k2: f64 = k[k.len() - 1];
    panic!("{}",
        format!("Parameter value t = {t} out of bounds for knot vector with first knot {k1} and last knot {k2}")
    );
}

fn cox_de_boor(k: &[f64], possible_span_indices: &[usize], degree: usize, i: usize, t: f64) -> f64 {
    if degree == 0 {
        if possible_span_indices.contains(&i) && find_span(&k, &possible_span_indices, t) == i {
            return 1.0;
        }
        return 0.0;
    }
    let mut f: f64 = 0.0;
    let mut g: f64 = 0.0;
    if k[i + degree] - k[i] != 0.0 {
        f = (t - k[i]) / (k[i + degree] - k[i]);
    }
    if k[i + degree + 1] - k[i + 1] != 0.0 {
        g = (k[i + degree + 1] - t) / (k[i + degree + 1] - k[i + 1]);
    }
    if f == 0.0 && g == 0.0 {
        return 0.0;
    }
    if g == 0.0 {
        return f * cox_de_boor(&k, &possible_span_indices, degree - 1, i, t);
    }
    if f == 0.0 {
        return g * cox_de_boor(&k, &possible_span_indices, degree - 1, i + 1, t);
    }
    return f * cox_de_boor(&k, &possible_span_indices, degree - 1, i, t) + g * cox_de_boor(
        &k, &possible_span_indices, degree - 1, i + 1, t);
}

#[pyfunction]
fn bspline_curve_eval(p: Vec<Vec<f64>>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;  // B-spline degree
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t);
        for j in 0..dim {
            evaluated_point[j] += p[i][j] * bspline_basis;
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bspline_curve_dcdt(p: Vec<Vec<f64>>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = k[i + q] - k[i];
        let span_b: f64 = k[i + q + 1] - k[i + 1];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
        for j in 0..dim {
            evaluated_deriv[j] += float_q * p[i][j] * bspline_basis_1;
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_curve_d2cdt2(p: Vec<Vec<f64>>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = k[i + q] - k[i];
        let span_b: f64 = k[i + q + 1] - k[i + 1];
        let span_c: f64 = k[i + q - 1] - k[i];
        let span_d: f64 = k[i + q] - k[i + 1];
        let span_e: f64 = k[i + q + 1] - k[i + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t);
        let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t));
        for j in 0..dim {
            evaluated_deriv[j] += float_q * (float_q - 1.0) * p[i][j] * bspline_basis_2
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_curve_eval_grid(p: Vec<Vec<f64>>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;  // B-spline degree
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n+1 {
            let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t);
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * bspline_basis;
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_curve_dcdt_grid(p: Vec<Vec<f64>>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_q * p[i][j] * bspline_basis_1;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_curve_d2cdt2_grid(p: Vec<Vec<f64>>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let span_c: f64 = k[i + q - 1] - k[i];
            let span_d: f64 = k[i + q] - k[i + 1];
            let span_e: f64 = k[i + q + 1] - k[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t);
            let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t));
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_q * (float_q - 1.0) * p[i][j] * bspline_basis_2
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_curve_eval_tvec(p: Vec<Vec<f64>>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;  // B-spline degree
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n+1 {
            let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t[t_idx]);
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * bspline_basis;
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_curve_dcdt_tvec(p: Vec<Vec<f64>>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t[t_idx]) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t[t_idx]);
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_q * p[i][j] * bspline_basis_1;
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_curve_d2cdt2_tvec(p: Vec<Vec<f64>>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let span_c: f64 = k[i + q - 1] - k[i];
            let span_d: f64 = k[i + q] - k[i + 1];
            let span_e: f64 = k[i + q + 1] - k[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t[t_idx]);
            let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t[t_idx]) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t[t_idx]));
            for j in 0..dim {
                evaluated_derivs[t_idx][j] += float_q * (float_q - 1.0) * p[i][j] * bspline_basis_2
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_eval(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
            for k in 0..dim {
                evaluated_point[k] += p[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bspline_surf_dsdu(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = ku[i + q] - ku[i];
        let span_b: f64 = ku[i + q + 1] - ku[i + 1];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_du = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            let bspline_basis_prod = bspline_basis_du * bspline_basis_v;
            for k in 0..dim {
                evaluated_deriv[k] += float_q * p[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_surf_dsdv(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    for j in 0..m+1 { // Switch the loop order for performance
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = kv[j + r] - kv[j];
        let span_b: f64 = kv[j + r + 1] - kv[j + 1];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_dv = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_dv;
            for k in 0..dim {
                evaluated_deriv[k] += float_r * p[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_surf_d2sdu2(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    if q < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_deriv);
    }
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = ku[i + q] - ku[i];
        let span_b: f64 = ku[i + q + 1] - ku[i + 1];
        let span_c: f64 = ku[i + q - 1] - ku[i];
        let span_d: f64 = ku[i + q] - ku[i + 1];
        let span_e: f64 = ku[i + q + 1] - ku[i + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u);
        let bspline_basis_d2u = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u));
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            let bspline_basis_prod = bspline_basis_d2u * bspline_basis_v;
            for k in 0..dim {
                evaluated_deriv[k] += float_q * (float_q - 1.0) * p[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_surf_d2sdv2(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    if r < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_deriv);
    }
    for j in 0..m+1 { // Switch the loop order for performance
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = kv[j + r] - kv[j];
        let span_b: f64 = kv[j + r + 1] - kv[j + 1];
        let span_c: f64 = kv[j + r - 1] - kv[j];
        let span_d: f64 = kv[j + r] - kv[j + 1];
        let span_e: f64 = kv[j + r + 1] - kv[j + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v);
        let bspline_basis_d2v = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v));
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_d2v;
            for k in 0..dim {
                evaluated_deriv[k] += float_r * (float_r - 1.0) * p[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn bspline_surf_eval_iso_u(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>,
    u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
                for k in 0..dim {
                    evaluated_points[v_idx][k] += p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_surf_eval_iso_v(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>,
    nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
                for k in 0..dim {
                    evaluated_points[u_idx][k] += p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_surf_dsdu_iso_u(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = ku[i + q] - ku[i];
            let span_b: f64 = ku[i + q + 1] - ku[i + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_du = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_du * bspline_basis_v;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += float_q * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_dsdu_iso_v(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = ku[i + q] - ku[i];
            let span_b: f64 = ku[i + q + 1] - ku[i + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_du = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_du * bspline_basis_v;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += float_q * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_dsdv_iso_u(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for j in 0..m+1 { // Switch the loop order for performance
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = kv[j + r] - kv[j];
            let span_b: f64 = kv[j + r + 1] - kv[j + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_dv = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_dv;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += float_r * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_dsdv_iso_v(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for j in 0..m+1 { // Switch the loop order for performance
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = kv[j + r] - kv[j];
            let span_b: f64 = kv[j + r + 1] - kv[j + 1];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_dv = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_dv;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += float_r * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdu2_iso_u(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    if q < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = ku[i + q] - ku[i];
            let span_b: f64 = ku[i + q + 1] - ku[i + 1];
            let span_c: f64 = ku[i + q - 1] - ku[i];
            let span_d: f64 = ku[i + q] - ku[i + 1];
            let span_e: f64 = ku[i + q + 1] - ku[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u);
            let bspline_basis_d2u = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u));
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_d2u * bspline_basis_v;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += float_q * (float_q - 1.0) * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdu2_iso_v(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    if q < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = ku[i + q] - ku[i];
            let span_b: f64 = ku[i + q + 1] - ku[i + 1];
            let span_c: f64 = ku[i + q - 1] - ku[i];
            let span_d: f64 = ku[i + q] - ku[i + 1];
            let span_e: f64 = ku[i + q + 1] - ku[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u);
            let bspline_basis_d2u = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u));
            for j in 0..m+1 {
                let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                let bspline_basis_prod = bspline_basis_d2u * bspline_basis_v;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += float_q * (float_q - 1.0) * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdv2_iso_u(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, nv: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];
    if r < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for v_idx in 0..nv {
        let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
        for j in 0..m+1 { // Switch the loop order for performance
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = kv[j + r] - kv[j];
            let span_b: f64 = kv[j + r + 1] - kv[j + 1];
            let span_c: f64 = kv[j + r - 1] - kv[j];
            let span_d: f64 = kv[j + r] - kv[j + 1];
            let span_e: f64 = kv[j + r + 1] - kv[j + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v);
            let bspline_basis_d2v = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v));
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_d2v;
                for k in 0..dim {
                    evaluated_derivs[v_idx][k] += float_r * (float_r - 1.0) * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdv2_iso_v(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, v: f64) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nu];
    if r < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for j in 0..m+1 { // Switch the loop order for performance
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = kv[j + r] - kv[j];
            let span_b: f64 = kv[j + r + 1] - kv[j + 1];
            let span_c: f64 = kv[j + r - 1] - kv[j];
            let span_d: f64 = kv[j + r] - kv[j + 1];
            let span_e: f64 = kv[j + r + 1] - kv[j + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v);
            let bspline_basis_d2v = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v));
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                let bspline_basis_prod = bspline_basis_u * bspline_basis_d2v;
                for k in 0..dim {
                    evaluated_derivs[u_idx][k] += float_r * (float_r - 1.0) * p[i][j][k] * bspline_basis_prod;
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_eval_grid(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>,
    nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_surf_dsdu_grid(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let span_a: f64 = ku[i + q] - ku[i];
                let span_b: f64 = ku[i + q + 1] - ku[i + 1];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                let bspline_basis_du = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                    let bspline_basis_prod = bspline_basis_du * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_q * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_dsdv_grid(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for j in 0..m+1 { // Switch the loop order for performance
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let span_a: f64 = kv[j + r] - kv[j];
                let span_b: f64 = kv[j + r + 1] - kv[j + 1];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                let bspline_basis_dv = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
                for i in 0..n+1 {
                    let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_r * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdu2_grid(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    if q < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for i in 0..n+1 {
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let mut kc: f64 = 0.0;
                let mut kd: f64 = 0.0;
                let mut ke: f64 = 0.0;
                let span_a: f64 = ku[i + q] - ku[i];
                let span_b: f64 = ku[i + q + 1] - ku[i + 1];
                let span_c: f64 = ku[i + q - 1] - ku[i];
                let span_d: f64 = ku[i + q] - ku[i + 1];
                let span_e: f64 = ku[i + q + 1] - ku[i + 2];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                if span_c != 0.0 {
                    kc = 1.0 / span_c;
                }
                if span_d != 0.0 {
                    kd = 1.0 / span_d;
                }
                if span_e != 0.0 {
                    ke = 1.0 / span_e;
                }
                let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u);
                let bspline_basis_d2u = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u));
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                    let bspline_basis_prod = bspline_basis_d2u * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_q * (float_q - 1.0) * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdv2_grid(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    if r < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            for j in 0..m+1 { // Switch the loop order for performance
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let mut kc: f64 = 0.0;
                let mut kd: f64 = 0.0;
                let mut ke: f64 = 0.0;
                let span_a: f64 = kv[j + r] - kv[j];
                let span_b: f64 = kv[j + r + 1] - kv[j + 1];
                let span_c: f64 = kv[j + r - 1] - kv[j];
                let span_d: f64 = kv[j + r] - kv[j + 1];
                let span_e: f64 = kv[j + r + 1] - kv[j + 2];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                if span_c != 0.0 {
                    kc = 1.0 / span_c;
                }
                if span_d != 0.0 {
                    kd = 1.0 / span_d;
                }
                if span_e != 0.0 {
                    ke = 1.0 / span_e;
                }
                let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v);
                let bspline_basis_d2v = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v));
                for i in 0..n+1 {
                    let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_d2v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_r * (float_r - 1.0) * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_eval_uvvecs(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>,
    u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let nu = u.len();
    let nv = v.len();
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u[u_idx]);
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v[v_idx]);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn bspline_surf_dsdu_uvvecs(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let nu = u.len();
    let nv = v.len();
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let span_a: f64 = ku[i + q] - ku[i];
                let span_b: f64 = ku[i + q + 1] - ku[i + 1];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                let bspline_basis_du = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u[u_idx]) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u[u_idx]);
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v[v_idx]);
                    let bspline_basis_prod = bspline_basis_du * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_q * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_dsdv_uvvecs(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let nu = u.len();
    let nv = v.len();
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for j in 0..m+1 { // Switch the loop order for performance
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let span_a: f64 = kv[j + r] - kv[j];
                let span_b: f64 = kv[j + r + 1] - kv[j + 1];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                let bspline_basis_dv = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v[v_idx]) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v[v_idx]);
                for i in 0..n+1 {
                    let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u[u_idx]);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_dv;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_r * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdu2_uvvecs(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let nu = u.len();
    let nv = v.len();
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let float_q = q as f64;
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    if q < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for i in 0..n+1 {
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let mut kc: f64 = 0.0;
                let mut kd: f64 = 0.0;
                let mut ke: f64 = 0.0;
                let span_a: f64 = ku[i + q] - ku[i];
                let span_b: f64 = ku[i + q + 1] - ku[i + 1];
                let span_c: f64 = ku[i + q - 1] - ku[i];
                let span_d: f64 = ku[i + q] - ku[i + 1];
                let span_e: f64 = ku[i + q + 1] - ku[i + 2];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                if span_c != 0.0 {
                    kc = 1.0 / span_c;
                }
                if span_d != 0.0 {
                    kd = 1.0 / span_d;
                }
                if span_e != 0.0 {
                    ke = 1.0 / span_e;
                }
                let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u[u_idx]);
                let bspline_basis_d2u = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u[u_idx]) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u[u_idx]));
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v[v_idx]);
                    let bspline_basis_prod = bspline_basis_d2u * bspline_basis_v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_q * (float_q - 1.0) * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn bspline_surf_d2sdv2_uvvecs(p: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let nu = u.len();
    let nv = v.len();
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_derivs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    if r < 2 { // Degree less than 2 implies the second derivative is zero
        return Ok(evaluated_derivs);
    }
    for u_idx in 0..nu {
        for v_idx in 0..nv {
            for j in 0..m+1 { // Switch the loop order for performance
                let mut ka: f64 = 0.0;
                let mut kb: f64 = 0.0;
                let mut kc: f64 = 0.0;
                let mut kd: f64 = 0.0;
                let mut ke: f64 = 0.0;
                let span_a: f64 = kv[j + r] - kv[j];
                let span_b: f64 = kv[j + r + 1] - kv[j + 1];
                let span_c: f64 = kv[j + r - 1] - kv[j];
                let span_d: f64 = kv[j + r] - kv[j + 1];
                let span_e: f64 = kv[j + r + 1] - kv[j + 2];
                if span_a != 0.0 {
                    ka = 1.0 / span_a;
                }
                if span_b != 0.0 {
                    kb = 1.0 / span_b;
                }
                if span_c != 0.0 {
                    kc = 1.0 / span_c;
                }
                if span_d != 0.0 {
                    kd = 1.0 / span_d;
                }
                if span_e != 0.0 {
                    ke = 1.0 / span_e;
                }
                let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v[v_idx]);
                let bspline_basis_d2v = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v[v_idx]) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v[v_idx]));
                for i in 0..n+1 {
                    let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u[u_idx]);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_d2v;
                    for k in 0..dim {
                        evaluated_derivs[u_idx][v_idx][k] += float_r * (float_r - 1.0) * p[i][j][k] * bspline_basis_prod;
                    }
                }
            }
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn nurbs_curve_eval(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t);
        w_sum += w[i] * bspline_basis;
        for j in 0..dim {
            evaluated_point[j] += p[i][j] * w[i] * bspline_basis;
        }
    }
    for j in 0..dim {
        evaluated_point[j] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn nurbs_curve_dcdt(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = k[i + q] - k[i];
        let span_b: f64 = k[i + q + 1] - k[i + 1];
        let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t);
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
        w_sum_0 += w[i] * bspline_basis_0;
        w_sum_1 += w[i] * bspline_basis_1;
        for j in 0..dim {
            sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
            sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
        }
    }
    for j in 0..dim {
        evaluated_deriv[j] = float_q * (sum_1[j] * w_sum_0 - sum_0[j] * w_sum_1) / (w_sum_0 * w_sum_0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_curve_d2cdt2(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = k[i + q] - k[i];
        let span_b: f64 = k[i + q + 1] - k[i + 1];
        let span_c: f64 = k[i + q - 1] - k[i];
        let span_d: f64 = k[i + q] - k[i + 1];
        let span_e: f64 = k[i + q + 1] - k[i + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t);
        let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
        let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t);
        let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t));
        w_sum_0 += w[i] * bspline_basis_0;
        w_sum_1 += w[i] * bspline_basis_1;
        w_sum_2 += w[i] * bspline_basis_2;
        for j in 0..dim {
            sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
            sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
            sum_2[j] += w[i] * p[i][j] * bspline_basis_2;
        }
    }
    for j in 0..dim {
        evaluated_deriv[j] = (
            float_q * (float_q - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
            float_q * (float_q - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
            2.0 * float_q * float_q * sum_1[j] * w_sum_0 * w_sum_1 +
            2.0 * float_q * float_q * sum_0[j] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_curve_eval_grid(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut w_sum: f64 = 0.0;
        for i in 0..n+1 {
            let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t);
            w_sum += w[i] * bspline_basis;
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * w[i] * bspline_basis;
            }
        }
        for j in 0..dim {
            evaluated_points[t_idx][j] /= w_sum;
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn nurbs_curve_dcdt_grid(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t);
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
            w_sum_0 += w[i] * bspline_basis_0;
            w_sum_1 += w[i] * bspline_basis_1;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
                sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = float_q * (sum_1[j] * w_sum_0 - sum_0[j] * w_sum_1) / (w_sum_0 * w_sum_0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn nurbs_curve_d2cdt2_grid(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, nt: usize) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let t = (t_idx as f64) * 1.0 / (nt as f64 - 1.0);
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        let mut w_sum_2: f64 = 0.0;
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let span_c: f64 = k[i + q - 1] - k[i];
            let span_d: f64 = k[i + q] - k[i + 1];
            let span_e: f64 = k[i + q + 1] - k[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t);
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t);
            let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t);
            let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t));
            w_sum_0 += w[i] * bspline_basis_0;
            w_sum_1 += w[i] * bspline_basis_1;
            w_sum_2 += w[i] * bspline_basis_2;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
                sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
                sum_2[j] += w[i] * p[i][j] * bspline_basis_2;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = (
                float_q * (float_q - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
                float_q * (float_q - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
                2.0 * float_q * float_q * sum_1[j] * w_sum_0 * w_sum_1 +
                2.0 * float_q * float_q * sum_0[j] * w_sum_1 * w_sum_1
            ) / w_sum_0.powf(3.0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn nurbs_curve_eval_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_points: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut w_sum: f64 = 0.0;
        for i in 0..n+1 {
            let bspline_basis = cox_de_boor(&k, &possible_span_indices, q, i, t[t_idx]);
            w_sum += w[i] * bspline_basis;
            for j in 0..dim {
                evaluated_points[t_idx][j] += p[i][j] * w[i] * bspline_basis;
            }
        }
        for j in 0..dim {
            evaluated_points[t_idx][j] /= w_sum;
        }
    }
    Ok(evaluated_points)
}

#[pyfunction]
fn nurbs_curve_dcdt_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t[t_idx]);
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t[t_idx]) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t[t_idx]);
            w_sum_0 += w[i] * bspline_basis_0;
            w_sum_1 += w[i] * bspline_basis_1;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
                sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = float_q * (sum_1[j] * w_sum_0 - sum_0[j] * w_sum_1) / (w_sum_0 * w_sum_0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn nurbs_curve_d2cdt2_tvec(p: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let n = p.len() - 1;  // Number of control points minus 1
    let nt = t.len();
    let num_knots = k.len();
    let q = num_knots - n - 2;
    let float_q = q as f64;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let dim = p[0].len();
    let mut evaluated_derivs: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    for t_idx in 0..nt {
        let mut sum_0: Vec<f64> = vec![0.0; dim];
        let mut sum_1: Vec<f64> = vec![0.0; dim];
        let mut sum_2: Vec<f64> = vec![0.0; dim];
        let mut w_sum_0: f64 = 0.0;
        let mut w_sum_1: f64 = 0.0;
        let mut w_sum_2: f64 = 0.0;
        for i in 0..n+1 {
            let mut ka: f64 = 0.0;
            let mut kb: f64 = 0.0;
            let mut kc: f64 = 0.0;
            let mut kd: f64 = 0.0;
            let mut ke: f64 = 0.0;
            let span_a: f64 = k[i + q] - k[i];
            let span_b: f64 = k[i + q + 1] - k[i + 1];
            let span_c: f64 = k[i + q - 1] - k[i];
            let span_d: f64 = k[i + q] - k[i + 1];
            let span_e: f64 = k[i + q + 1] - k[i + 2];
            if span_a != 0.0 {
                ka = 1.0 / span_a;
            }
            if span_b != 0.0 {
                kb = 1.0 / span_b;
            }
            if span_c != 0.0 {
                kc = 1.0 / span_c;
            }
            if span_d != 0.0 {
                kd = 1.0 / span_d;
            }
            if span_e != 0.0 {
                ke = 1.0 / span_e;
            }
            let bspline_basis_0 = cox_de_boor(&k, &possible_span_indices, q, i, t[t_idx]);
            let bspline_basis_1 = ka * cox_de_boor(&k, &possible_span_indices, q - 1, i, t[t_idx]) - kb * cox_de_boor(&k, &possible_span_indices, q - 1, i + 1, t[t_idx]);
            let bspline_basis_d = kd * cox_de_boor(&k, &possible_span_indices, q - 2, i + 1, t[t_idx]);
            let bspline_basis_2 = ka * (kc * cox_de_boor(&k, &possible_span_indices, q - 2, i, t[t_idx]) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&k, &possible_span_indices, q - 2, i + 2, t[t_idx]));
            w_sum_0 += w[i] * bspline_basis_0;
            w_sum_1 += w[i] * bspline_basis_1;
            w_sum_2 += w[i] * bspline_basis_2;
            for j in 0..dim {
                sum_0[j] += w[i] * p[i][j] * bspline_basis_0;
                sum_1[j] += w[i] * p[i][j] * bspline_basis_1;
                sum_2[j] += w[i] * p[i][j] * bspline_basis_2;
            }
        }
        for j in 0..dim {
            evaluated_derivs[t_idx][j] = (
                float_q * (float_q - 1.0) * sum_2[j] * w_sum_0 * w_sum_0 - 
                float_q * (float_q - 1.0) * sum_0[j] * w_sum_0 * w_sum_2 -
                2.0 * float_q * float_q * sum_1[j] * w_sum_0 * w_sum_1 +
                2.0 * float_q * float_q * sum_0[j] * w_sum_1 * w_sum_1
            ) / w_sum_0.powf(3.0);
        }
    }
    Ok(evaluated_derivs)
}

#[pyfunction]
fn nurbs_surf_eval(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
            w_sum += w[i][j] * bspline_basis_prod;
            for k in 0..dim {
                evaluated_point[k] += p[i][j][k] * w[i][j] * bspline_basis_prod;
            }
        }
    }
    for k in 0..dim {
        evaluated_point[k] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn nurbs_surf_dsdu(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_q = q as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = ku[i + q] - ku[i];
        let span_b: f64 = ku[i + q + 1] - ku[i + 1];
        let bspline_basis_0 = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_1 = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            w_sum_0 += w[i][j] * bspline_basis_0 * bspline_basis_v;
            w_sum_1 += w[i][j] * bspline_basis_1 * bspline_basis_v;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * bspline_basis_0 * bspline_basis_v;
                sum_1[k] += w[i][j] * p[i][j][k] * bspline_basis_1 * bspline_basis_v;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = float_q * (sum_1[k] * w_sum_0 - sum_0[k] * w_sum_1) / (w_sum_0 * w_sum_0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_surf_dsdv(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    for j in 0..m+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let span_a: f64 = kv[j + r] - kv[j];
        let span_b: f64 = kv[j + r + 1] - kv[j + 1];
        let bspline_basis_0 = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        let bspline_basis_1 = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            w_sum_0 += w[i][j] * bspline_basis_0 * bspline_basis_u;
            w_sum_1 += w[i][j] * bspline_basis_1 * bspline_basis_u;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * bspline_basis_0 * bspline_basis_u;
                sum_1[k] += w[i][j] * p[i][j][k] * bspline_basis_1 * bspline_basis_u;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = float_r * (sum_1[k] * w_sum_0 - sum_0[k] * w_sum_1) / (w_sum_0 * w_sum_0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_surf_d2sdu2(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_q = q as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for i in 0..n+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = ku[i + q] - ku[i];
        let span_b: f64 = ku[i + q + 1] - ku[i + 1];
        let span_c: f64 = ku[i + q - 1] - ku[i];
        let span_d: f64 = ku[i + q] - ku[i + 1];
        let span_e: f64 = ku[i + q + 1] - ku[i + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_0 = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
        let bspline_basis_1 = ka * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i, u) - kb * cox_de_boor(&ku, &possible_span_indices_u, q - 1, i + 1, u);
        let bspline_basis_d = kd * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 1, u);
        let bspline_basis_2 = ka * (kc * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i, u) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&ku, &possible_span_indices_u, q - 2, i + 2, u));
        for j in 0..m+1 {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
            w_sum_0 += w[i][j] * bspline_basis_0 * bspline_basis_v;
            w_sum_1 += w[i][j] * bspline_basis_1 * bspline_basis_v;
            w_sum_2 += w[i][j] * bspline_basis_2 * bspline_basis_v;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * bspline_basis_0 * bspline_basis_v;
                sum_1[k] += w[i][j] * p[i][j][k] * bspline_basis_1 * bspline_basis_v;
                sum_2[k] += w[i][j] * p[i][j][k] * bspline_basis_2 * bspline_basis_v;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = (
            float_q * (float_q - 1.0) * sum_2[k] * w_sum_0 * w_sum_0 - 
            float_q * (float_q - 1.0) * sum_0[k] * w_sum_0 * w_sum_2 -
            2.0 * float_q * float_q * sum_1[k] * w_sum_0 * w_sum_1 +
            2.0 * float_q * float_q * sum_0[k] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_surf_d2sdv2(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let float_r = r as f64;
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_deriv: Vec<f64> = vec![0.0; dim];
    let mut sum_0: Vec<f64> = vec![0.0; dim];
    let mut sum_1: Vec<f64> = vec![0.0; dim];
    let mut sum_2: Vec<f64> = vec![0.0; dim];
    let mut w_sum_0: f64 = 0.0;
    let mut w_sum_1: f64 = 0.0;
    let mut w_sum_2: f64 = 0.0;
    for j in 0..m+1 {
        let mut ka: f64 = 0.0;
        let mut kb: f64 = 0.0;
        let mut kc: f64 = 0.0;
        let mut kd: f64 = 0.0;
        let mut ke: f64 = 0.0;
        let span_a: f64 = kv[j + r] - kv[j];
        let span_b: f64 = kv[j + r + 1] - kv[j + 1];
        let span_c: f64 = kv[j + r - 1] - kv[j];
        let span_d: f64 = kv[j + r] - kv[j + 1];
        let span_e: f64 = kv[j + r + 1] - kv[j + 2];
        if span_a != 0.0 {
            ka = 1.0 / span_a;
        }
        if span_b != 0.0 {
            kb = 1.0 / span_b;
        }
        if span_c != 0.0 {
            kc = 1.0 / span_c;
        }
        if span_d != 0.0 {
            kd = 1.0 / span_d;
        }
        if span_e != 0.0 {
            ke = 1.0 / span_e;
        }
        let bspline_basis_0 = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
        let bspline_basis_1 = ka * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j, v) - kb * cox_de_boor(&kv, &possible_span_indices_v, r - 1, j + 1, v);
        let bspline_basis_d = kd * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 1, v);
        let bspline_basis_2 = ka * (kc * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j, v) - bspline_basis_d) - kb * (bspline_basis_d - ke * cox_de_boor(&kv, &possible_span_indices_v, r - 2, j + 2, v));
        for i in 0..n+1 {
            let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
            w_sum_0 += w[i][j] * bspline_basis_0 * bspline_basis_u;
            w_sum_1 += w[i][j] * bspline_basis_1 * bspline_basis_u;
            w_sum_2 += w[i][j] * bspline_basis_2 * bspline_basis_u;
            for k in 0..dim {
                sum_0[k] += w[i][j] * p[i][j][k] * bspline_basis_0 * bspline_basis_u;
                sum_1[k] += w[i][j] * p[i][j][k] * bspline_basis_1 * bspline_basis_u;
                sum_2[k] += w[i][j] * p[i][j][k] * bspline_basis_2 * bspline_basis_u;
            }
        }
    }
    for k in 0..dim {
        evaluated_deriv[k] = (
            float_r * (float_r - 1.0) * sum_2[k] * w_sum_0 * w_sum_0 - 
            float_r * (float_r - 1.0) * sum_0[k] * w_sum_0 * w_sum_2 -
            2.0 * float_r * float_r * sum_1[k] * w_sum_0 * w_sum_1 +
            2.0 * float_r * float_r * sum_0[k] * w_sum_1 * w_sum_1
        ) / w_sum_0.powf(3.0);
    }
    Ok(evaluated_deriv)
}

#[pyfunction]
fn nurbs_surf_eval_grid(p: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, nu: usize, nv: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let n = p.len() - 1;  // Number of control points in the u-direction minus 1
    let m = p[0].len() - 1;  // Number of control points in the v-direction minus 1
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let q = num_knots_u - n - 2;  // Degree in the u-direction
    let r = num_knots_v - m - 2;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = p[0][0].len();  // Number of spatial dimensions
    let mut evaluated_points: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dim]; nv]; nu];
    for u_idx in 0..nu {
        let u = (u_idx as f64) * 1.0 / (nu as f64 - 1.0);
        for v_idx in 0..nv {
            let v = (v_idx as f64) * 1.0 / (nv as f64 - 1.0);
            let mut w_sum: f64 = 0.0;
            for i in 0..n+1 {
                let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, q, i, u);
                for j in 0..m+1 {
                    let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, r, j, v);
                    let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
                    w_sum += w[i][j] * bspline_basis_prod;
                    for k in 0..dim {
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * w[i][j] * bspline_basis_prod;
                    }
                }
            }
            for k in 0..dim {
                evaluated_points[u_idx][v_idx][k] /= w_sum;
            }
        }
    }
    Ok(evaluated_points)
}

#[pymodule]
fn rust_nurbs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bernstein_poly, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_dcdt, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_d2cdt2, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_dcdt_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_d2cdt2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_eval_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_dcdt_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_d2cdt2_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdu, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdv, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdu2, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdv2, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdu_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdu_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdv_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdv_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdu2_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdu2_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdv2_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdv2_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdu_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdv_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdu2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdv2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdu_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_dsdv_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdu2_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_d2sdv2_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_dcdt, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_d2cdt2, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_dcdt_grid, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_d2cdt2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_eval_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_dcdt_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_d2cdt2_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_dsdu, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_dsdv, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_d2sdu2, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_d2sdv2, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_dcdt, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_d2cdt2, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_dcdt_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_d2cdt2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_eval_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_dcdt_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_d2cdt2_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdu, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdv, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdu2, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdv2, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdu_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdu_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdv_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdv_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdu2_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdu2_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdv2_iso_u, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdv2_iso_v, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdu_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdv_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdu2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdv2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdu_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_dsdv_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdu2_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_d2sdv2_uvvecs, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_dcdt, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_d2cdt2, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_eval_grid, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_dcdt_grid, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_d2cdt2_grid, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_eval_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_dcdt_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_d2cdt2_tvec, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_dsdu, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_dsdv, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_d2sdu2, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_d2sdv2, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_eval_grid, m)?)?;
    Ok(())
}
