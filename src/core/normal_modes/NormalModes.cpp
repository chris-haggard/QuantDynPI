#include "NormalModes.hpp"

#include <cassert>
#include <cmath>

/**
 * @brief Construct a new Normal Mode Converter obect
 *
 * Implements the normal mode transform in 10.1063/1.3489925. When the MKL
 parameter pack format = PACK_FORMAT the FFT output
 * is ordered: R_0,R1,I1,R2,I2...IL-1,RL The result is that the normal modes are
 * in order: 0,-1,+1,-2,+2,-3,+3,+4 (for numberBeads = 8) note the final normal
 * mode is +ve. Sign is not important currently but is for Matsubara dynamics.
 *
 *
 *
 * @param c an armadillo cube, either position or momentum, only needed for
 * dimensions.  */

NormalModes::NormalModes(const arma::cube &c)
    : ForwardScale(c.n_rows, c.n_cols, c.n_slices, arma::fill::ones),
      BackwardScale(c.n_rows, c.n_cols, c.n_slices, arma::fill::ones) {
  status =
      DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_REAL, 1, c.n_rows);
  status = DftiSetValue(desc_handle, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
  status = DftiSetValue(desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (c.n_rows != 1) {
    assert(c.n_rows % 2 == 0);  // allows for classical sim i.e. one bead
    status =
        DftiSetValue(desc_handle, DFTI_FORWARD_SCALE, sqrt(2.0) / c.n_rows);
    status = DftiSetValue(
        desc_handle, DFTI_BACKWARD_SCALE,
        1.0 / sqrt(2.0));  // Forward_Scale ^ -1 * 1/N
    for (arma::uword i = 0; i < c.n_slices; i++) {
      // scale 0th normal mode and n-1 normal mode correctly
      ForwardScale.slice(i).row(0) /= sqrt(2.0);
      ForwardScale.slice(i).row(c.n_rows - 1) /= sqrt(2.0);
    }
    BackwardScale /= ForwardScale;
  }
  status = DftiCommitDescriptor(desc_handle);
}

void NormalModes::CartToNM(arma::cube &cart, arma::cube &nm) {
  for (arma::uword i = 0; i < cart.n_slices; i++) {
    for (arma::uword j = 0; j < cart.n_cols; j++) {
      status = DftiComputeForward(
          desc_handle, cart.slice(i).colptr(j), nm.slice(i).colptr(j));
    }
  }
  // subcube due to the NM transform required for the constraints gradient
  // acting on a individual molecules therefore fewer slices than whole system
  // that ForwardScale is designed for. Not required for backward scale as the
  // constraints gradient is not transformed back.
  nm %= ForwardScale(
      arma::span::all, arma::span::all, arma::span(0, nm.n_slices - 1));
}

void NormalModes::NMToCart(arma::cube &nm, arma::cube &cart) {
  // undoing backward scale is faster than creating a temp via copying
  nm %= BackwardScale;
  for (arma::uword i = 0; i < cart.n_slices; i++) {
    for (arma::uword j = 0; j < cart.n_cols; j++) {
      status = DftiComputeBackward(
          desc_handle, nm.slice(i).colptr(j), cart.slice(i).colptr(j));
    }
  }
  nm /= BackwardScale;
}

//************************************************************************************************************
//
// Testing functions below do not use
//
//************************************************************************************************************

/**
 * @brief Normal mode transform in 10.1063/1.3489925. The results have a
 * different ordering to the FFT above, but the values are the same.
 *
 * @param m
 *
 * @return
 */
arma::mat CartToNM_TEST(const arma::mat &m) {
  // only valid for even number of beads
  assert(m.n_rows % 2 == 0);
  arma::mat nm(arma::size(m));
  // form transformation matrix
  const unsigned int n_beads = m.n_rows;
  const double r_n_beads = 1.0 / (double)n_beads;
  const double rsqrt_n_beads = std::sqrt(r_n_beads);
  const double rsqrt2_n_bead = std::sqrt(2.0) * rsqrt_n_beads;
  arma::mat transformation_mat(m.n_rows, m.n_rows);
  for (arma::uword j = 0; j < n_beads; j++) {
    for (arma::uword k = 0; k < n_beads; k++) {
      if (k == 0) {
        transformation_mat(k, j) = rsqrt_n_beads;
      } else if (k >= 1 && k <= (n_beads / 2) - 1) {
        transformation_mat(k, j) =
            rsqrt2_n_bead * std::cos(2.0 * M_PI * k * j * r_n_beads);
      } else if (k == (n_beads / 2)) {
        transformation_mat(k, j) = rsqrt_n_beads * std::pow(-1, j);
      } else if (k >= (n_beads / 2) + 1 && k <= n_beads - 1) {
        transformation_mat(k, j) =
            rsqrt2_n_bead * std::sin(2.0 * M_PI * k * j * r_n_beads);
      }
    }
  }
  for (arma::uword j = 0; j < m.n_cols; j++) {
    nm.col(j) = (transformation_mat * m.col(j));
  }

  return nm;
}

// not guaranteed to be correct
arma::mat NMToCart_TEST(const arma::mat &m) {
  // only valid for even number of beads
  assert(m.n_rows % 2 == 0);
  arma::mat nm(m.n_rows, m.n_cols, arma::fill::zeros);
  for (arma::uword i = 0; i < m.n_cols; i++) {
    for (arma::uword j = 0; j < m.n_rows; j++) {
      for (arma::uword k = 0; k < m.n_rows; k++) {
        if (k == 0) {
          nm(j, i) += sqrt(1.0 / m.n_rows) * m(k, i);
        } else if (k >= 1 && k <= (m.n_rows / 2) - 1) {
          nm(j, i) += sqrt(2.0 / m.n_rows) *
                      cos(2.0 * M_PI * j * k * (1.0 / m.n_rows)) * m(k, i);
        } else if (k == m.n_rows / 2) {
          nm(j, i) += sqrt(1.0 / m.n_rows) * pow(-1.0, j) * m(k, i);
        } else if (k >= (m.n_rows / 2) + 1 && k <= m.n_rows - 1) {
          nm(j, i) += sqrt(2.0 / m.n_rows) *
                      sin(2.0 * M_PI * j * k * (1.0 / m.n_rows)) * m(k, i);
        }
      }
    }
  }
  return nm;
}

// ******* Below functions are broken ***********

arma::mat convToNormalModes(const arma::mat &m) {
  // only valid for even number of beads
  assert(m.n_rows % 2 == 0);
  arma::cx_mat nm(m.n_rows, m.n_cols, arma::fill::zeros);
  nm = arma::fft(m) * sqrt(2.0 / m.n_rows);
  nm.row(0) /= sqrt(2.0);
  // integer divison coming up
  nm.row(m.n_rows / 2) /= sqrt(2.0);
  for (arma::uword j = 0; j < m.n_cols; j++) {
    for (arma::uword i = (m.n_rows / 2) + 1; i < m.n_rows; i++) {
      nm(i, j) = -nm(i, j).imag();
    }
  }
  return arma::real(nm);
}

arma::mat convToCartesian(const arma::mat &m) {
  // this does not work yet
  assert(m.n_rows % 2 == 0);
  arma::cx_mat nm(m.n_rows, m.n_cols, arma::fill::zeros);
  arma::mat B(m.n_rows, m.n_cols, arma::fill::zeros);
  arma::cx_mat A(m, B);
  nm = arma::ifft(A) / sqrt(2.0 / m.n_rows);
  nm.row(0) *= sqrt(2.0);
  nm.row(m.n_rows - 1) /= sqrt(2.0);
  for (arma::uword j = 0; j < m.n_cols; j++) {
    for (arma::uword i = (m.n_rows / 2) + 1; i < m.n_rows; i++) {
      nm(i, j) = +nm(i, j).real();
    }
  }
  nm.print("\n\n");
  return arma::real(nm);
}

