/**
 * @file RingPolymer.cpp
 * @brief Ring-polymer class defining variables needed to describe a
 * ring-polymer. The key variables are the position and momenta of the beads.
 * These are each stored in an arma::cube with shape (n_beads, 3, n_molec).
 * @version 0.1
 */

#include "RingPolymer.hpp"

#include <stdexcept>

#include "../../util/RingPolymerUtils.hpp"

/**
 * @brief Construct a new Ring Polymer:: Ring Polymer object
 *
 * position_cart = Cartesian position cube
 * position_nm = position normal modes
 * momenta_cart = Cartesian momenta (starts as randn to allow PILE_G to work)
 * momenta_nm = momenta normal modes
 * force_cart = Cartesian forces
 * force_nm = normal mode forces
 * mass_atomic = mass atoms
 * mass_atomic_sqrt = sqrt of above
 * mass_scaled = scaled masses of the normal modes for adaibatic CMD/QCMD
 * (initialized as mass_atomic)
 * mass_scaled_sqrt = sqrt of above
 * freq = normal mode frequencies
 * labels = label for each atom
 * n_beads = number of beads (== n_rows)
 * n_atoms = number of atoms per molecule
 * n_molec = number of molecules
 * n_total_atoms = n_atoms * n_molec
 * freq = normal mode frequencies
 * freq_scaled = scaled normal mode frequencies for adiabatic CMD/QCMD,
 * initialized as freq
 *
 * @param RingPolymerParams
 */

RingPolymer::RingPolymer(
    const InputHandler::RingPolymerInput &RingPolymerParams, Ensemble &Ens)
    : labels(RingPolymerParams.labels_from_init),
      n_beads(RingPolymerParams.n_beads),
      n_atoms(RingPolymerParams.n_atoms),
      n_molec(RingPolymerParams.n_molec),
      n_total_atoms(n_atoms * n_molec),
      beta(Ens.beta),
      beta_n(Ens.beta / n_beads),
      position_cart(RingPolymerParams.position_from_init),
      position_nm(arma::cube(arma::size(position_cart))),
      momenta_cart(arma::cube(arma::size(position_cart), arma::fill::zeros)),
      momenta_nm(arma::cube(arma::size(position_cart), arma::fill::zeros)),
      force_cart(arma::cube(arma::size(position_cart), arma::fill::zeros)),
      force_nm(arma::cube(arma::size(position_cart), arma::fill::zeros)),
      mass_atomic(FillAtomicMass()),
      mass_atomic_sqrt(arma::sqrt(mass_atomic)),
      mass_scaled(mass_atomic),
      mass_scaled_sqrt(arma::sqrt(mass_scaled)),
      freq(FillFreq()),
      freq_scaled(freq),
      NMConv(position_cart) {
  assert(position_cart.n_slices == n_atoms * n_molec);
  momenta_nm = FillInitialMomentaNM();
  NMConv.CartToNM(position_cart, position_nm);
  NMConv.NMToCart(momenta_nm, momenta_cart);
}

/**
 * @brief Ensures that the momenta in normal modes and cartesian coordinates are
 * update when neccessary i.e. before applying a thermostat step which updates
 * just the normal modes;
 */
/*
void RingPolymer::MomentaTracker() {
  switch (current_momenta) {
    case (EQ):
      break;
    case (CART):
      NMConv.CartToNM(momenta_cart, momenta_nm);
      current_momenta = EQ;
      break;
    case (NM):
      NMConv.NMToCart(momenta_nm, momenta_cart);
      current_momenta = EQ;
      break;
  }
}
*/

arma::cube RingPolymer::FillAtomicMass() {
  arma::cube out(n_beads, 3, n_atoms * n_molec);
  for (arma::uword i = 0; i < out.n_slices; i++) {
    assert(out.n_slices == labels.size());
    out.slice(i).fill(AtomData::GetAtomMass(labels.at(i)));
  }
  return out;
}

arma::cube RingPolymer::FillFreq() {
  arma::cube out(n_beads, 3, n_atoms * n_molec);
  for (arma::uword k = 0; k < out.n_rows; k++) {
    // this generates 0, 1, 1, 2, 2 ...
    // see normal modes file for details
    double j = std::round(k / 2.0);
    double omega_k =
        2.0 * (1.0 / beta_n) * std::sin(j * M_PI * (1.0 / n_beads));
    out.row(k).fill(omega_k);
  }
  return out;
}

/**
 * @brief Fill momenta normal modes from Maxwell-Boltzmann distribution.
 */
arma::cube RingPolymer::FillInitialMomentaNM() {
  arma::cube out(arma::size(momenta_nm));
  for (arma::uword k = 0; k < momenta_nm.n_slices; k++) {
    for (arma::uword i = 0; i < momenta_nm.n_rows; i++) {
      out.slice(k).row(i) = InitializeMomenta(
          momenta_nm.slice(k).row(i), mass_scaled.slice(k).row(i), beta);
    }
  }
  return out;
}

arma::cube RingPolymer::Centroid() {
  return arma::mean(position_cart, 0);
}

/**
 * @brief Cube defined by RingPolymer object that will be written to file when
 * saving a trajectory. Necessary virtual function as the QuasiCentroidRP needs
 * to return the quasicentroid positions;
 *
 * @return
 */
arma::cube RingPolymer::TrajectoryVariable() {
  return position_cart;
}

/**
 * @brief Cube defined by RingPolymer object that defines the dynamical variable
 * to pass to TCF functions (such as the dipole function). For RPMD this is the
 * beads, CMD the centroid, QCMD the quasicentroid.
 *
 * @return
 */
arma::cube RingPolymer::DynamicalVariable() {
  return position_cart;
}

void RingPolymer::PlaceQuasicentroid() {
  throw std::runtime_error("Only available for Quasicentroid RP");
}

