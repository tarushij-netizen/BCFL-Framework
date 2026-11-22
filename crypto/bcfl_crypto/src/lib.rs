//! Ristretto255 group operations for the BCFL thesis, exposed to Python.
//!
//! Every group element crosses the boundary as its 32-byte canonical compressed encoding and
//! every scalar as its 32-byte canonical little-endian encoding (an integer below the group
//! order L). Python does the scalar arithmetic (polynomials, Lagrange) with ints mod L and
//! draws all randomness with `secrets`; this module only does deterministic group operations,
//! all from curve25519-dalek. Non-canonical or invalid inputs raise ValueError.
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::{Identity, VartimeMultiscalarMul};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn arr32(b: &[u8], what: &str) -> PyResult<[u8; 32]> {
    b.try_into()
        .map_err(|_| PyValueError::new_err(format!("{what} must be 32 bytes, got {}", b.len())))
}

fn scalar(b: &[u8]) -> PyResult<Scalar> {
    Option::from(Scalar::from_canonical_bytes(arr32(b, "scalar")?))
        .ok_or_else(|| PyValueError::new_err("scalar is not canonical (must be < L)"))
}

fn point(b: &[u8]) -> PyResult<RistrettoPoint> {
    CompressedRistretto(arr32(b, "point")?)
        .decompress()
        .ok_or_else(|| PyValueError::new_err("not a valid Ristretto255 encoding"))
}

fn out<'py>(py: Python<'py>, p: RistrettoPoint) -> Bound<'py, PyBytes> {
    PyBytes::new(py, p.compress().as_bytes())
}

/// s * B for the standard basepoint B.
#[pyfunction]
fn basepoint_mul<'py>(py: Python<'py>, s: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    Ok(out(py, RISTRETTO_BASEPOINT_TABLE * &scalar(s)?))
}

/// s * P.
#[pyfunction]
fn point_mul<'py>(py: Python<'py>, p: &[u8], s: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    Ok(out(py, point(p)? * scalar(s)?))
}

/// P + Q.
#[pyfunction]
fn point_add<'py>(py: Python<'py>, p: &[u8], q: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    Ok(out(py, point(p)? + point(q)?))
}

/// P - Q.
#[pyfunction]
fn point_sub<'py>(py: Python<'py>, p: &[u8], q: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    Ok(out(py, point(p)? - point(q)?))
}

/// sum_i s_i * P_i (variable time: inputs are public in every use here).
#[pyfunction]
fn multiscalar_mul<'py>(py: Python<'py>, scalars: Vec<Vec<u8>>, points: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyBytes>> {
    if scalars.len() != points.len() {
        return Err(PyValueError::new_err("scalars and points differ in length"));
    }
    let ss = scalars.iter().map(|b| scalar(b)).collect::<PyResult<Vec<_>>>()?;
    let ps = points.iter().map(|b| point(b)).collect::<PyResult<Vec<_>>>()?;
    Ok(out(py, RistrettoPoint::vartime_multiscalar_mul(ss.iter(), ps.iter())))
}

/// The identity element.
#[pyfunction]
fn identity<'py>(py: Python<'py>) -> Bound<'py, PyBytes> {
    out(py, RistrettoPoint::identity())
}

/// Map 64 uniform bytes (e.g. a SHA-512 output) to a point with unknown discrete log.
#[pyfunction]
fn from_uniform_bytes<'py>(py: Python<'py>, b: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let a: [u8; 64] = b
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("need 64 bytes, got {}", b.len())))?;
    Ok(out(py, RistrettoPoint::from_uniform_bytes(&a)))
}

/// True if b is a canonical encoding of a group element.
#[pyfunction]
fn is_valid_point(b: &[u8]) -> bool {
    point(b).is_ok()
}

#[pymodule]
fn bcfl_crypto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(basepoint_mul, m)?)?;
    m.add_function(wrap_pyfunction!(point_mul, m)?)?;
    m.add_function(wrap_pyfunction!(point_add, m)?)?;
    m.add_function(wrap_pyfunction!(point_sub, m)?)?;
    m.add_function(wrap_pyfunction!(multiscalar_mul, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(from_uniform_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_point, m)?)?;
    Ok(())
}
