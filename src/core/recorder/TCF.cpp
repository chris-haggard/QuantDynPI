#include "TCF.hpp"

#include <cassert>
#include <cstddef>

/**
 * @brief Storage of n_total_atoms for position/velocity autocorrelation,
 * n_molec for dipole (as dipole is per molecule). Storage is a field with
 * n_elem = number of rows in dynamical variable i.e. 1 for CMD/QCMD but n_beads
 * for TRPMD
 *
 * @param steps
 * @param n_atoms_or_molecules
 */
TCF::TCF(
    const arma::cube &dynamic_variable_cube, size_t steps, size_t stride_,
    size_t n_atoms_or_molecules, size_t num_mpi_proc)
    : stride(stride_),
      tcf_steps(steps / stride),
      storage(tcf_steps * 2, 3, n_atoms_or_molecules, arma::fill::zeros),
      tcf(tcf_steps, arma::fill::zeros),
      out_buffer(arma::mat(tcf_steps, num_mpi_proc, arma::fill::zeros)) {
  assert(steps % stride == 0 && "steps not multiple of stride\n");
}

/**
 * @brief Record the centroid of the provided cube in storage. storage is
 * a cube where each row in each slice is the centroid recorded at each
 * timestep thus the shape is (n_steps * 2, 3, n_total_atoms). The n_steps
 * * 2 is for 0 padding if length n_step to give better FFT in below
 * correlation function.
 *
 * @param c
 * @param pos
 */
void TCF::Record(const arma::cube &c, size_t pos) {
  // changed for dipole storage. Cube comes with 1 row, 3 cols, n_molec slices
  // where each slice is the dipole moment of the centroids of one water
  // molecule
  assert(c.n_rows == 1);
  if (pos % stride == 0) {
    storage.row(pos / stride) = c.row(0);
  }
}

/**
 * @brief Given a cube of all the positions/velocities of all beads at all
 * timesteps in a trajectory, calculate the time-correlation function using
 * the FFT. Very computationally efficient, especially for autocorrelation.
 * Algorithm:
 * 1. Create a complex matrix with where each row is the position/velocity of
 * the centroid at that timestep and all the imaginary values are 0. There are
 * twice as many rows as timesteps so the bottom half of the real values are
 * also 0.
 * 2. Perform FFT on matrix
 * 3. Element wise multiplication by complex conjugate
 * 4. Inverse FFT
 * 5. Discard bottom half of matrix to leave the n_rows == numberTimesteps
 * (this only works with this implementation of FFT due to packing of results.
 * Other implementations, i.e. MKL, give different packing), take real part
 * and sum across dimensions to give TCF.
 *
 * Zeroes appended to reduce spurious correlation.
 *
 *
 * @param storage Cube of all positions in a trajectory.
 * @param tcf The TCF to be updated i.e. positionTCF
 * @return void
 */

void TCF::CorrelateTCFs() {
  arma::cube zero_padding(arma::size(storage), arma::fill::zeros);
  arma::cx_mat storage_FFT;

  for (arma::uword k = 0; k < storage.n_slices; k++) {
    storage_FFT = arma::cx_mat(storage.slice(k), zero_padding.slice(k));
    storage_FFT = arma::fft(storage_FFT);
    storage_FFT %= arma::conj(storage_FFT);
    storage_FFT = arma::ifft(storage_FFT);
    storage_FFT.shed_rows((storage_FFT.n_rows / 2), storage_FFT.n_rows - 1);
    tcf += arma::sum(arma::real(storage_FFT), 1);
  }
  assert(arma::is_finite(tcf));
}

/**
 * @brief Correctly normalise TCF to account for trajectories and samples
 * (n_molec or n_total_atoms). Apply normalisation according to Eq. 9 in
 * doi.org/10.1016/0021-9991(76)90043-7 (noting that the 1/2N is implicitly
 * included in the inverse DFT).
 *
 * The long time limit is unlikely to be converged so run_length >> tcf_length
 * (i.e. for a converged tcf of length 1ps consider a trajectory of lenth 3
 * ps, therefore the first 1 ps will be converged)
 *
 *
 * @param tcf
 * @param n_traj
 * @param n_total_atoms
 * @param n_timesteps
 */

void TCF::NormaliseTCFs(const unsigned int n_traj) {
  tcf /= (n_traj * storage.n_slices);
  for (arma::uword i = 0; i < tcf.n_rows; i++) {
    tcf.row(i) /= (tcf.n_rows - i);
  }
}

