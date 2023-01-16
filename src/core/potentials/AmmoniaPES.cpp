#include "AmmoniaPES.hpp"

#include <cmath>
#include <vector>

#include "Ammonia_Dipole_cpp_wrapper.hpp"
#include "Ammonia_PES_Constants.hpp"

AmmoniaPES::AmmoniaPES(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : Potential(SimParams, PotParams) {
}

arma::cube AmmoniaPES::Dipole(const arma::cube &q) {
  assert(q.n_slices % 4 == 0);
  arma::cube mu(q.n_rows, 3, q.n_slices / 4);
  arma::mat temp(4, 3);  // copy to get around const memptr
  arma::vec mu_temp(q.n_cols);

  int x = 4;
  int y = 3;

  for (arma::uword i = 0; i < q.n_slices; i += 4) {
    for (arma::uword j = 0; j < q.n_rows; j++) {
      temp.row(0) = q.slice(i).row(j);
      temp.row(1) = q.slice(i + 1).row(j);
      temp.row(2) = q.slice(i + 2).row(j);
      temp.row(3) = q.slice(i + 3).row(j);
      AQZfc_dipole_wrapper(temp.memptr(), mu_temp.memptr(), &x, &y);
      mu.slice(i / 4).row(j) = mu_temp.t();
    }
  }
  return mu;
}

arma::vec AmmoniaPES::Pot(const arma::cube &q) {
  arma::vec v(q.n_slices / 4, arma::fill::zeros);

  double xh1, yh1, zh1;
  double xh2, yh2, zh2;
  double xh3, yh3, zh3;
  double xn, yn, zn;

  arma::rowvec r1vec(3);
  arma::rowvec r2vec(3);
  arma::rowvec r3vec(3);
  double r1, r2, r3;

  double alp1, alp2, alp3;

  double xi1, xi2, xi3, xi4, xi5, sind;

  for (arma::uword i = 0; i < q.n_slices; i += 4) {
    for (arma::uword j = 0; j < q.n_rows; j++) {
      xh1 = arma::as_scalar(q(j, 0, i));
      yh1 = arma::as_scalar(q(j, 1, i));
      zh1 = arma::as_scalar(q(j, 2, i));
      xh2 = arma::as_scalar(q(j, 0, i + 1));
      yh2 = arma::as_scalar(q(j, 1, i + 1));
      zh2 = arma::as_scalar(q(j, 2, i + 1));
      xh3 = arma::as_scalar(q(j, 0, i + 2));
      yh3 = arma::as_scalar(q(j, 1, i + 2));
      zh3 = arma::as_scalar(q(j, 2, i + 2));
      xn = arma::as_scalar(q(j, 0, i + 3));
      yn = arma::as_scalar(q(j, 1, i + 3));
      zn = arma::as_scalar(q(j, 2, i + 3));

      r1vec = q.slice(i + 3).row(j) - q.slice(i).row(j);
      r2vec = q.slice(i + 3).row(j) - q.slice(i + 1).row(j);
      r3vec = q.slice(i + 3).row(j) - q.slice(i + 2).row(j);

      r1 = arma::norm(r1vec);
      r2 = arma::norm(r2vec);
      r3 = arma::norm(r3vec);

      alp1 = std::acos(arma::norm_dot(r2vec, r3vec));
      alp2 = std::acos(arma::norm_dot(r1vec, r3vec));
      alp3 = std::acos(arma::norm_dot(r1vec, r2vec));

      xi1 = 1.0 - std::exp(-a_const::a * (r1 - a_const::re));
      xi2 = 1.0 - std::exp(-a_const::a * (r2 - a_const::re));
      xi3 = 1.0 - std::exp(-a_const::a * (r3 - a_const::re));
      xi4 = (2.0 * alp1 - alp2 - alp3) / std::sqrt(6.0);
      xi5 = (alp2 - alp3) / std::sqrt(2.0);
      sind =
          -((2.0 / std::sqrt(3.0)) * std::sin((alp1 + alp2 + alp3) / 6.0) -
            std::sin(a_const::rhoe));

      v(i / 4) +=
          a_const::f0a + a_const::f1a * sind +
          a_const::f2a * std::pow(sind, 2) + a_const::f3a * std::pow(sind, 3) +
          a_const::f4a * std::pow(sind, 4) + a_const::f5a * std::pow(sind, 5) +
          a_const::f6a * std::pow(sind, 6) + a_const::f7a * std::pow(sind, 7) +
          a_const::f8a * std::pow(sind, 8) + a_const::ve +
          std::pow(xi1, 6) * (a_const::f1a111111 * sind +
                              a_const::f2a111111 * std::pow(sind, 2)) +
          std::pow(xi1, 5) * xi2 *
              (a_const::f1a111112 * sind +
               a_const::f2a111112 * std::pow(sind, 2)) +
          std::pow(xi1, 5) * xi3 *
              (a_const::f1a111113 * sind +
               a_const::f2a111113 * std::pow(sind, 2)) +
          std::pow(xi1, 5) * xi4 *
              (a_const::f1a111114 * sind +
               a_const::f2a111114 * std::pow(sind, 2)) +
          std::pow(xi1, 5) * xi5 *
              (a_const::f1a111115 * sind +
               a_const::f2a111115 * std::pow(sind, 2)) +
          std::pow(xi1, 5) * (a_const::f0a11111 + a_const::f1a11111 * sind +
                              a_const::f2a11111 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * std::pow(xi2, 2) *
              (a_const::f1a111122 * sind +
               a_const::f2a111122 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 * xi3 *
              (a_const::f1a111123 * sind +
               a_const::f2a111123 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 * xi4 *
              (a_const::f1a111124 * sind +
               a_const::f2a111124 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 * xi5 *
              (a_const::f1a111125 * sind +
               a_const::f2a111125 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 *
              (a_const::f0a11112 + a_const::f1a11112 * sind +
               a_const::f2a11112 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * std::pow(xi3, 2) *
              (a_const::f1a111133 * sind +
               a_const::f2a111133 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 * xi4 *
              (a_const::f1a111134 * sind +
               a_const::f2a111134 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 * xi5 *
              (a_const::f1a111135 * sind +
               a_const::f2a111135 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 *
              (a_const::f0a11113 + a_const::f1a11113 * sind +
               a_const::f2a11113 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * std::pow(xi4, 2) *
              (a_const::f1a111144 * sind +
               a_const::f2a111144 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi4 * xi5 *
              (a_const::f1a111145 * sind +
               a_const::f2a111145 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi4 *
              (a_const::f0a11114 + a_const::f1a11114 * sind +
               a_const::f2a11114 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * std::pow(xi5, 2) *
              (a_const::f1a111155 * sind +
               a_const::f2a111155 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi5 *
              (a_const::f0a11115 + a_const::f1a11115 * sind +
               a_const::f2a11115 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * (a_const::f0a1111 + a_const::f1a1111 * sind +
                              a_const::f2a1111 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 3) *
              (a_const::f1a111222 * sind +
               a_const::f2a111222 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a111223 * sind +
               a_const::f2a111223 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a111224 * sind +
               a_const::f2a111224 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a111225 * sind +
               a_const::f2a111225 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f0a11122 + a_const::f1a11122 * sind +
               a_const::f2a11122 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a111233 * sind +
               a_const::f2a111233 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi3 * xi4 *
              (a_const::f1a111234 * sind +
               a_const::f2a111234 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi3 * xi5 *
              (a_const::f1a111235 * sind +
               a_const::f2a111235 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f0a11123 + a_const::f1a11123 * sind +
               a_const::f2a11123 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a111244 * sind +
               a_const::f2a111244 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi4 * xi5 *
              (a_const::f1a111245 * sind +
               a_const::f2a111245 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f0a11124 + a_const::f1a11124 * sind +
               a_const::f2a11124 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a111255 * sind +
               a_const::f2a111255 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f0a11125 + a_const::f1a11125 * sind +
               a_const::f2a11125 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 *
              (a_const::f0a1112 + a_const::f1a1112 * sind +
               a_const::f2a1112 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 3) *
              (a_const::f1a111333 * sind +
               a_const::f2a111333 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a111334 * sind +
               a_const::f2a111334 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a111335 * sind +
               a_const::f2a111335 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f0a11133 + a_const::f1a11133 * sind +
               a_const::f2a11133 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a111344 * sind +
               a_const::f2a111344 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi4 * xi5 *
              (a_const::f1a111345 * sind +
               a_const::f2a111345 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f0a11134 + a_const::f1a11134 * sind +
               a_const::f2a11134 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a111355 * sind +
               a_const::f2a111355 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f0a11135 + a_const::f1a11135 * sind +
               a_const::f2a11135 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 *
              (a_const::f0a1113 + a_const::f1a1113 * sind +
               a_const::f2a1113 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 3) *
              (a_const::f1a111444 * sind +
               a_const::f2a111444 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a111445 * sind +
               a_const::f2a111445 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f0a11144 + a_const::f1a11144 * sind +
               a_const::f2a11144 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a111455 * sind +
               a_const::f2a111455 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f0a11145 + a_const::f1a11145 * sind +
               a_const::f2a11145 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 *
              (a_const::f0a1114 + a_const::f1a1114 * sind +
               a_const::f2a1114 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi5, 3) *
              (a_const::f1a111555 * sind +
               a_const::f2a111555 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f0a11155 + a_const::f1a11155 * sind +
               a_const::f2a11155 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi5 *
              (a_const::f0a1115 + a_const::f1a1115 * sind +
               a_const::f2a1115 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * (a_const::f0a111 + a_const::f1a111 * sind +
                              a_const::f2a111 * std::pow(sind, 2) +
                              a_const::f3a111 * std::pow(sind, 3)) +
          std::pow(xi1, 2) * std::pow(xi2, 4) *
              (a_const::f1a112222 * sind +
               a_const::f2a112222 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi3 *
              (a_const::f1a112223 * sind +
               a_const::f2a112223 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi4 *
              (a_const::f1a112224 * sind +
               a_const::f2a112224 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi5 *
              (a_const::f1a112225 * sind +
               a_const::f2a112225 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f0a11222 + a_const::f1a11222 * sind +
               a_const::f2a11222 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a112233 * sind +
               a_const::f2a112233 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a112234 * sind +
               a_const::f2a112234 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a112235 * sind +
               a_const::f2a112235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f0a11223 + a_const::f1a11223 * sind +
               a_const::f2a11223 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a112244 * sind +
               a_const::f2a112244 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a112245 * sind +
               a_const::f2a112245 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f0a11224 + a_const::f1a11224 * sind +
               a_const::f2a11224 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a112255 * sind +
               a_const::f2a112255 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f0a11225 + a_const::f1a11225 * sind +
               a_const::f2a11225 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a1122 + a_const::f1a1122 * sind +
               a_const::f2a1122 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 3) *
              (a_const::f1a112333 * sind +
               a_const::f2a112333 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a112334 * sind +
               a_const::f2a112334 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a112335 * sind +
               a_const::f2a112335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f0a11233 + a_const::f1a11233 * sind +
               a_const::f2a11233 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a112344 * sind +
               a_const::f2a112344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f0a11234 + a_const::f1a11234 * sind +
               a_const::f2a11234 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a112355 * sind +
               a_const::f2a112355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f0a11235 + a_const::f1a11235 * sind +
               a_const::f2a11235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a1123 + a_const::f1a1123 * sind +
               a_const::f2a1123 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 3) *
              (a_const::f1a112444 * sind +
               a_const::f2a112444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a112445 * sind +
               a_const::f2a112445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f0a11244 + a_const::f1a11244 * sind +
               a_const::f2a11244 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a112455 * sind +
               a_const::f2a112455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f0a11245 + a_const::f1a11245 * sind +
               a_const::f2a11245 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a1124 + a_const::f1a1124 * sind +
               a_const::f2a1124 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 3) *
              (a_const::f1a112555 * sind +
               a_const::f2a112555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f0a11255 + a_const::f1a11255 * sind +
               a_const::f2a11255 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a1125 + a_const::f1a1125 * sind +
               a_const::f2a1125 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 *
              (a_const::f0a112 + a_const::f1a112 * sind +
               a_const::f2a112 * std::pow(sind, 2) +
               a_const::f3a112 * std::pow(sind, 3)) +
          std::pow(xi1, 2) * std::pow(xi3, 4) *
              (a_const::f1a113333 * sind +
               a_const::f2a113333 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) * xi4 *
              (a_const::f1a113334 * sind +
               a_const::f2a113334 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) * xi5 *
              (a_const::f1a113335 * sind +
               a_const::f2a113335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f0a11333 + a_const::f1a11333 * sind +
               a_const::f2a11333 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a113344 * sind +
               a_const::f2a113344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a113345 * sind +
               a_const::f2a113345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f0a11334 + a_const::f1a11334 * sind +
               a_const::f2a11334 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a113355 * sind +
               a_const::f2a113355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f0a11335 + a_const::f1a11335 * sind +
               a_const::f2a11335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a1133 + a_const::f1a1133 * sind +
               a_const::f2a1133 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 3) *
              (a_const::f1a113444 * sind +
               a_const::f2a113444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a113445 * sind +
               a_const::f2a113445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f0a11344 + a_const::f1a11344 * sind +
               a_const::f2a11344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a113455 * sind +
               a_const::f2a113455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f0a11345 + a_const::f1a11345 * sind +
               a_const::f2a11345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a1134 + a_const::f1a1134 * sind +
               a_const::f2a1134 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 3) *
              (a_const::f1a113555 * sind +
               a_const::f2a113555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f0a11355 + a_const::f1a11355 * sind +
               a_const::f2a11355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a1135 + a_const::f1a1135 * sind +
               a_const::f2a1135 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 *
              (a_const::f0a113 + a_const::f1a113 * sind +
               a_const::f2a113 * std::pow(sind, 2) +
               a_const::f3a113 * std::pow(sind, 3)) +
          std::pow(xi1, 2) * std::pow(xi4, 4) *
              (a_const::f1a114444 * sind +
               a_const::f2a114444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a114445 * sind +
               a_const::f2a114445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f0a11444 + a_const::f1a11444 * sind +
               a_const::f2a11444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a114455 * sind +
               a_const::f2a114455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f0a11445 + a_const::f1a11445 * sind +
               a_const::f2a11445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a1144 + a_const::f1a1144 * sind +
               a_const::f2a1144 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a114555 * sind +
               a_const::f2a114555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f0a11455 + a_const::f1a11455 * sind +
               a_const::f2a11455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a1145 + a_const::f1a1145 * sind +
               a_const::f2a1145 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 *
              (a_const::f0a114 + a_const::f1a114 * sind +
               a_const::f2a114 * std::pow(sind, 2) +
               a_const::f3a114 * std::pow(sind, 3)) +
          std::pow(xi1, 2) * std::pow(xi5, 4) *
              (a_const::f1a115555 * sind +
               a_const::f2a115555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f0a11555 + a_const::f1a11555 * sind +
               a_const::f2a11555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a1155 + a_const::f1a1155 * sind +
               a_const::f2a1155 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi5 *
              (a_const::f0a115 + a_const::f1a115 * sind +
               a_const::f2a115 * std::pow(sind, 2) +
               a_const::f3a115 * std::pow(sind, 3)) +
          std::pow(xi1, 2) * (a_const::f0a11 + a_const::f1a11 * sind +
                              a_const::f2a11 * std::pow(sind, 2) +
                              a_const::f3a11 * std::pow(sind, 3) +
                              a_const::f4a11 * std::pow(sind, 4)) +
          xi1 * std::pow(xi2, 5) *
              (a_const::f1a122222 * sind +
               a_const::f2a122222 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 4) * xi3 *
              (a_const::f1a122223 * sind +
               a_const::f2a122223 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 4) * xi4 *
              (a_const::f1a122224 * sind +
               a_const::f2a122224 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 4) * xi5 *
              (a_const::f1a122225 * sind +
               a_const::f2a122225 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 4) *
              (a_const::f0a12222 + a_const::f1a12222 * sind +
               a_const::f2a12222 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a122233 * sind +
               a_const::f2a122233 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a122234 * sind +
               a_const::f2a122234 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a122235 * sind +
               a_const::f2a122235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f0a12223 + a_const::f1a12223 * sind +
               a_const::f2a12223 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a122244 * sind +
               a_const::f2a122244 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a122245 * sind +
               a_const::f2a122245 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f0a12224 + a_const::f1a12224 * sind +
               a_const::f2a12224 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a122255 * sind +
               a_const::f2a122255 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f0a12225 + a_const::f1a12225 * sind +
               a_const::f2a12225 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) *
              (a_const::f0a1222 + a_const::f1a1222 * sind +
               a_const::f2a1222 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a122333 * sind +
               a_const::f2a122333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a122334 * sind +
               a_const::f2a122334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a122335 * sind +
               a_const::f2a122335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a12233 + a_const::f1a12233 * sind +
               a_const::f2a12233 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a122344 * sind +
               a_const::f2a122344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a12234 + a_const::f1a12234 * sind +
               a_const::f2a12234 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a122355 * sind +
               a_const::f2a122355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a12235 + a_const::f1a12235 * sind +
               a_const::f2a12235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a1223 + a_const::f1a1223 * sind +
               a_const::f2a1223 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a122444 * sind +
               a_const::f2a122444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a122445 * sind +
               a_const::f2a122445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a12244 + a_const::f1a12244 * sind +
               a_const::f2a12244 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a122455 * sind +
               a_const::f2a122455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a12245 + a_const::f1a12245 * sind +
               a_const::f2a12245 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a1224 + a_const::f1a1224 * sind +
               a_const::f2a1224 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a122555 * sind +
               a_const::f2a122555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a12255 + a_const::f1a12255 * sind +
               a_const::f2a12255 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a1225 + a_const::f1a1225 * sind +
               a_const::f2a1225 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) *
              (a_const::f0a122 + a_const::f1a122 * sind +
               a_const::f2a122 * std::pow(sind, 2) +
               a_const::f3a122 * std::pow(sind, 3)) +
          xi1 * xi2 * std::pow(xi3, 4) *
              (a_const::f1a123333 * sind +
               a_const::f2a123333 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a123334 * sind +
               a_const::f2a123334 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a123335 * sind +
               a_const::f2a123335 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f0a12333 + a_const::f1a12333 * sind +
               a_const::f2a12333 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a123344 * sind +
               a_const::f2a123344 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a12334 + a_const::f1a12334 * sind +
               a_const::f2a12334 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a123355 * sind +
               a_const::f2a123355 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a12335 + a_const::f1a12335 * sind +
               a_const::f2a12335 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a1233 + a_const::f1a1233 * sind +
               a_const::f2a1233 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a123444 * sind +
               a_const::f2a123444 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a12344 + a_const::f1a12344 * sind +
               a_const::f2a12344 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a1234 + a_const::f1a1234 * sind +
               a_const::f2a1234 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a123555 * sind +
               a_const::f2a123555 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a12355 + a_const::f1a12355 * sind +
               a_const::f2a12355 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a1235 + a_const::f1a1235 * sind +
               a_const::f2a1235 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 *
              (a_const::f0a123 + a_const::f1a123 * sind +
               a_const::f2a123 * std::pow(sind, 2) +
               a_const::f3a123 * std::pow(sind, 3)) +
          xi1 * xi2 * std::pow(xi4, 4) *
              (a_const::f1a124444 * sind +
               a_const::f2a124444 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a124445 * sind +
               a_const::f2a124445 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f0a12444 + a_const::f1a12444 * sind +
               a_const::f2a12444 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a124455 * sind +
               a_const::f2a124455 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a12445 + a_const::f1a12445 * sind +
               a_const::f2a12445 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a1244 + a_const::f1a1244 * sind +
               a_const::f2a1244 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a124555 * sind +
               a_const::f2a124555 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a12455 + a_const::f1a12455 * sind +
               a_const::f2a12455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a1245 + a_const::f1a1245 * sind +
               a_const::f2a1245 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 *
              (a_const::f0a124 + a_const::f1a124 * sind +
               a_const::f2a124 * std::pow(sind, 2) +
               a_const::f3a124 * std::pow(sind, 3)) +
          xi1 * xi2 * std::pow(xi5, 4) *
              (a_const::f1a125555 * sind +
               a_const::f2a125555 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f0a12555 + a_const::f1a12555 * sind +
               a_const::f2a12555 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a1255 + a_const::f1a1255 * sind +
               a_const::f2a1255 * std::pow(sind, 2)) +
          xi1 * xi2 * xi5 *
              (a_const::f0a125 + a_const::f1a125 * sind +
               a_const::f2a125 * std::pow(sind, 2) +
               a_const::f3a125 * std::pow(sind, 3)) +
          xi1 * xi2 *
              (a_const::f0a12 + a_const::f1a12 * sind +
               a_const::f2a12 * std::pow(sind, 2) +
               a_const::f3a12 * std::pow(sind, 3) +
               a_const::f4a12 * std::pow(sind, 4)) +
          xi1 * std::pow(xi3, 5) *
              (a_const::f1a133333 * sind +
               a_const::f2a133333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a133334 * sind +
               a_const::f2a133334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a133335 * sind +
               a_const::f2a133335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 4) *
              (a_const::f0a13333 + a_const::f1a13333 * sind +
               a_const::f2a13333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a133344 * sind +
               a_const::f2a133344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a133345 * sind +
               a_const::f2a133345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f0a13334 + a_const::f1a13334 * sind +
               a_const::f2a13334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a133355 * sind +
               a_const::f2a133355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f0a13335 + a_const::f1a13335 * sind +
               a_const::f2a13335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) *
              (a_const::f0a1333 + a_const::f1a1333 * sind +
               a_const::f2a1333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a133444 * sind +
               a_const::f2a133444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a133445 * sind +
               a_const::f2a133445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a13344 + a_const::f1a13344 * sind +
               a_const::f2a13344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a133455 * sind +
               a_const::f2a133455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a13345 + a_const::f1a13345 * sind +
               a_const::f2a13345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a1334 + a_const::f1a1334 * sind +
               a_const::f2a1334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a133555 * sind +
               a_const::f2a133555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a13355 + a_const::f1a13355 * sind +
               a_const::f2a13355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a1335 + a_const::f1a1335 * sind +
               a_const::f2a1335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) *
              (a_const::f0a133 + a_const::f1a133 * sind +
               a_const::f2a133 * std::pow(sind, 2) +
               a_const::f3a133 * std::pow(sind, 3)) +
          xi1 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a134444 * sind +
               a_const::f2a134444 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a134445 * sind +
               a_const::f2a134445 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f0a13444 + a_const::f1a13444 * sind +
               a_const::f2a13444 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a134455 * sind +
               a_const::f2a134455 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a13445 + a_const::f1a13445 * sind +
               a_const::f2a13445 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a1344 + a_const::f1a1344 * sind +
               a_const::f2a1344 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a134555 * sind +
               a_const::f2a134555 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a13455 + a_const::f1a13455 * sind +
               a_const::f2a13455 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a1345 + a_const::f1a1345 * sind +
               a_const::f2a1345 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 *
              (a_const::f0a134 + a_const::f1a134 * sind +
               a_const::f2a134 * std::pow(sind, 2) +
               a_const::f3a134 * std::pow(sind, 3)) +
          xi1 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a135555 * sind +
               a_const::f2a135555 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f0a13555 + a_const::f1a13555 * sind +
               a_const::f2a13555 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a1355 + a_const::f1a1355 * sind +
               a_const::f2a1355 * std::pow(sind, 2)) +
          xi1 * xi3 * xi5 *
              (a_const::f0a135 + a_const::f1a135 * sind +
               a_const::f2a135 * std::pow(sind, 2) +
               a_const::f3a135 * std::pow(sind, 3)) +
          xi1 * xi3 *
              (a_const::f0a13 + a_const::f1a13 * sind +
               a_const::f2a13 * std::pow(sind, 2) +
               a_const::f3a13 * std::pow(sind, 3) +
               a_const::f4a13 * std::pow(sind, 4)) +
          xi1 * std::pow(xi4, 5) *
              (a_const::f1a144444 * sind +
               a_const::f2a144444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a144445 * sind +
               a_const::f2a144445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 4) *
              (a_const::f0a14444 + a_const::f1a14444 * sind +
               a_const::f2a14444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a144455 * sind +
               a_const::f2a144455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f0a14445 + a_const::f1a14445 * sind +
               a_const::f2a14445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) *
              (a_const::f0a1444 + a_const::f1a1444 * sind +
               a_const::f2a1444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a144555 * sind +
               a_const::f2a144555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a14455 + a_const::f1a14455 * sind +
               a_const::f2a14455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a1445 + a_const::f1a1445 * sind +
               a_const::f2a1445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) *
              (a_const::f0a144 + a_const::f1a144 * sind +
               a_const::f2a144 * std::pow(sind, 2) +
               a_const::f3a144 * std::pow(sind, 3)) +
          xi1 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a145555 * sind +
               a_const::f2a145555 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f0a14555 + a_const::f1a14555 * sind +
               a_const::f2a14555 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a1455 + a_const::f1a1455 * sind +
               a_const::f2a1455 * std::pow(sind, 2)) +
          xi1 * xi4 * xi5 *
              (a_const::f0a145 + a_const::f1a145 * sind +
               a_const::f2a145 * std::pow(sind, 2) +
               a_const::f3a145 * std::pow(sind, 3)) +
          xi1 * xi4 *
              (a_const::f0a14 + a_const::f1a14 * sind +
               a_const::f2a14 * std::pow(sind, 2) +
               a_const::f3a14 * std::pow(sind, 3) +
               a_const::f4a14 * std::pow(sind, 4)) +
          xi1 * std::pow(xi5, 5) *
              (a_const::f1a155555 * sind +
               a_const::f2a155555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 4) *
              (a_const::f0a15555 + a_const::f1a15555 * sind +
               a_const::f2a15555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 3) *
              (a_const::f0a1555 + a_const::f1a1555 * sind +
               a_const::f2a1555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 2) *
              (a_const::f0a155 + a_const::f1a155 * sind +
               a_const::f2a155 * std::pow(sind, 2) +
               a_const::f3a155 * std::pow(sind, 3)) +
          xi1 * xi5 *
              (a_const::f0a15 + a_const::f1a15 * sind +
               a_const::f2a15 * std::pow(sind, 2) +
               a_const::f3a15 * std::pow(sind, 3) +
               a_const::f4a15 * std::pow(sind, 4)) +
          xi1 * (a_const::f0a1 + a_const::f1a1 * sind +
                 a_const::f2a1 * std::pow(sind, 2) +
                 a_const::f3a1 * std::pow(sind, 3) +
                 a_const::f4a1 * std::pow(sind, 4) +
                 a_const::f5a1 * std::pow(sind, 5) +
                 a_const::f6a1 * std::pow(sind, 6)) +
          std::pow(xi2, 6) * (a_const::f1a222222 * sind +
                              a_const::f2a222222 * std::pow(sind, 2)) +
          std::pow(xi2, 5) * xi3 *
              (a_const::f1a222223 * sind +
               a_const::f2a222223 * std::pow(sind, 2)) +
          std::pow(xi2, 5) * xi4 *
              (a_const::f1a222224 * sind +
               a_const::f2a222224 * std::pow(sind, 2)) +
          std::pow(xi2, 5) * xi5 *
              (a_const::f1a222225 * sind +
               a_const::f2a222225 * std::pow(sind, 2)) +
          std::pow(xi2, 5) * (a_const::f0a22222 + a_const::f1a22222 * sind +
                              a_const::f2a22222 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * std::pow(xi3, 2) *
              (a_const::f1a222233 * sind +
               a_const::f2a222233 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 * xi4 *
              (a_const::f1a222234 * sind +
               a_const::f2a222234 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 * xi5 *
              (a_const::f1a222235 * sind +
               a_const::f2a222235 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 *
              (a_const::f0a22223 + a_const::f1a22223 * sind +
               a_const::f2a22223 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * std::pow(xi4, 2) *
              (a_const::f1a222244 * sind +
               a_const::f2a222244 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi4 * xi5 *
              (a_const::f1a222245 * sind +
               a_const::f2a222245 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi4 *
              (a_const::f0a22224 + a_const::f1a22224 * sind +
               a_const::f2a22224 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * std::pow(xi5, 2) *
              (a_const::f1a222255 * sind +
               a_const::f2a222255 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi5 *
              (a_const::f0a22225 + a_const::f1a22225 * sind +
               a_const::f2a22225 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * (a_const::f0a2222 + a_const::f1a2222 * sind +
                              a_const::f2a2222 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 3) *
              (a_const::f1a222333 * sind +
               a_const::f2a222333 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a222334 * sind +
               a_const::f2a222334 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a222335 * sind +
               a_const::f2a222335 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f0a22233 + a_const::f1a22233 * sind +
               a_const::f2a22233 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a222344 * sind +
               a_const::f2a222344 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi4 * xi5 *
              (a_const::f1a222345 * sind +
               a_const::f2a222345 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f0a22234 + a_const::f1a22234 * sind +
               a_const::f2a22234 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a222355 * sind +
               a_const::f2a222355 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f0a22235 + a_const::f1a22235 * sind +
               a_const::f2a22235 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 *
              (a_const::f0a2223 + a_const::f1a2223 * sind +
               a_const::f2a2223 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 3) *
              (a_const::f1a222444 * sind +
               a_const::f2a222444 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a222445 * sind +
               a_const::f2a222445 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f0a22244 + a_const::f1a22244 * sind +
               a_const::f2a22244 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a222455 * sind +
               a_const::f2a222455 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f0a22245 + a_const::f1a22245 * sind +
               a_const::f2a22245 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 *
              (a_const::f0a2224 + a_const::f1a2224 * sind +
               a_const::f2a2224 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi5, 3) *
              (a_const::f1a222555 * sind +
               a_const::f2a222555 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f0a22255 + a_const::f1a22255 * sind +
               a_const::f2a22255 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi5 *
              (a_const::f0a2225 + a_const::f1a2225 * sind +
               a_const::f2a2225 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * (a_const::f0a222 + a_const::f1a222 * sind +
                              a_const::f2a222 * std::pow(sind, 2) +
                              a_const::f3a222 * std::pow(sind, 3)) +
          std::pow(xi2, 2) * std::pow(xi3, 4) *
              (a_const::f1a223333 * sind +
               a_const::f2a223333 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) * xi4 *
              (a_const::f1a223334 * sind +
               a_const::f2a223334 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) * xi5 *
              (a_const::f1a223335 * sind +
               a_const::f2a223335 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f0a22333 + a_const::f1a22333 * sind +
               a_const::f2a22333 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a223344 * sind +
               a_const::f2a223344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a223345 * sind +
               a_const::f2a223345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f0a22334 + a_const::f1a22334 * sind +
               a_const::f2a22334 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a223355 * sind +
               a_const::f2a223355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f0a22335 + a_const::f1a22335 * sind +
               a_const::f2a22335 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a2233 + a_const::f1a2233 * sind +
               a_const::f2a2233 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 3) *
              (a_const::f1a223444 * sind +
               a_const::f2a223444 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a223445 * sind +
               a_const::f2a223445 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f0a22344 + a_const::f1a22344 * sind +
               a_const::f2a22344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a223455 * sind +
               a_const::f2a223455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f0a22345 + a_const::f1a22345 * sind +
               a_const::f2a22345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a2234 + a_const::f1a2234 * sind +
               a_const::f2a2234 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 3) *
              (a_const::f1a223555 * sind +
               a_const::f2a223555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f0a22355 + a_const::f1a22355 * sind +
               a_const::f2a22355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a2235 + a_const::f1a2235 * sind +
               a_const::f2a2235 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 *
              (a_const::f0a223 + a_const::f1a223 * sind +
               a_const::f2a223 * std::pow(sind, 2) +
               a_const::f3a223 * std::pow(sind, 3)) +
          std::pow(xi2, 2) * std::pow(xi4, 4) *
              (a_const::f1a224444 * sind +
               a_const::f2a224444 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a224445 * sind +
               a_const::f2a224445 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f0a22444 + a_const::f1a22444 * sind +
               a_const::f2a22444 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a224455 * sind +
               a_const::f2a224455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f0a22445 + a_const::f1a22445 * sind +
               a_const::f2a22445 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a2244 + a_const::f1a2244 * sind +
               a_const::f2a2244 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a224555 * sind +
               a_const::f2a224555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f0a22455 + a_const::f1a22455 * sind +
               a_const::f2a22455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a2245 + a_const::f1a2245 * sind +
               a_const::f2a2245 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 *
              (a_const::f0a224 + a_const::f1a224 * sind +
               a_const::f2a224 * std::pow(sind, 2) +
               a_const::f3a224 * std::pow(sind, 3)) +
          std::pow(xi2, 2) * std::pow(xi5, 4) *
              (a_const::f1a225555 * sind +
               a_const::f2a225555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f0a22555 + a_const::f1a22555 * sind +
               a_const::f2a22555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a2255 + a_const::f1a2255 * sind +
               a_const::f2a2255 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi5 *
              (a_const::f0a225 + a_const::f1a225 * sind +
               a_const::f2a225 * std::pow(sind, 2) +
               a_const::f3a225 * std::pow(sind, 3)) +
          std::pow(xi2, 2) * (a_const::f0a22 + a_const::f1a22 * sind +
                              a_const::f2a22 * std::pow(sind, 2) +
                              a_const::f3a22 * std::pow(sind, 3) +
                              a_const::f4a22 * std::pow(sind, 4)) +
          xi2 * std::pow(xi3, 5) *
              (a_const::f1a233333 * sind +
               a_const::f2a233333 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a233334 * sind +
               a_const::f2a233334 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a233335 * sind +
               a_const::f2a233335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 4) *
              (a_const::f0a23333 + a_const::f1a23333 * sind +
               a_const::f2a23333 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a233344 * sind +
               a_const::f2a233344 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a233345 * sind +
               a_const::f2a233345 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f0a23334 + a_const::f1a23334 * sind +
               a_const::f2a23334 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a233355 * sind +
               a_const::f2a233355 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f0a23335 + a_const::f1a23335 * sind +
               a_const::f2a23335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) *
              (a_const::f0a2333 + a_const::f1a2333 * sind +
               a_const::f2a2333 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a233444 * sind +
               a_const::f2a233444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a233445 * sind +
               a_const::f2a233445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a23344 + a_const::f1a23344 * sind +
               a_const::f2a23344 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a233455 * sind +
               a_const::f2a233455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a23345 + a_const::f1a23345 * sind +
               a_const::f2a23345 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a2334 + a_const::f1a2334 * sind +
               a_const::f2a2334 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a233555 * sind +
               a_const::f2a233555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a23355 + a_const::f1a23355 * sind +
               a_const::f2a23355 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a2335 + a_const::f1a2335 * sind +
               a_const::f2a2335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) *
              (a_const::f0a233 + a_const::f1a233 * sind +
               a_const::f2a233 * std::pow(sind, 2) +
               a_const::f3a233 * std::pow(sind, 3)) +
          xi2 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a234444 * sind +
               a_const::f2a234444 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a234445 * sind +
               a_const::f2a234445 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f0a23444 + a_const::f1a23444 * sind +
               a_const::f2a23444 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a234455 * sind +
               a_const::f2a234455 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a23445 + a_const::f1a23445 * sind +
               a_const::f2a23445 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a2344 + a_const::f1a2344 * sind +
               a_const::f2a2344 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a234555 * sind +
               a_const::f2a234555 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a23455 + a_const::f1a23455 * sind +
               a_const::f2a23455 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a2345 + a_const::f1a2345 * sind +
               a_const::f2a2345 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 *
              (a_const::f0a234 + a_const::f1a234 * sind +
               a_const::f2a234 * std::pow(sind, 2) +
               a_const::f3a234 * std::pow(sind, 3)) +
          xi2 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a235555 * sind +
               a_const::f2a235555 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f0a23555 + a_const::f1a23555 * sind +
               a_const::f2a23555 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a2355 + a_const::f1a2355 * sind +
               a_const::f2a2355 * std::pow(sind, 2)) +
          xi2 * xi3 * xi5 *
              (a_const::f0a235 + a_const::f1a235 * sind +
               a_const::f2a235 * std::pow(sind, 2) +
               a_const::f3a235 * std::pow(sind, 3)) +
          xi2 * xi3 *
              (a_const::f0a23 + a_const::f1a23 * sind +
               a_const::f2a23 * std::pow(sind, 2) +
               a_const::f3a23 * std::pow(sind, 3) +
               a_const::f4a23 * std::pow(sind, 4)) +
          xi2 * std::pow(xi4, 5) *
              (a_const::f1a244444 * sind +
               a_const::f2a244444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a244445 * sind +
               a_const::f2a244445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 4) *
              (a_const::f0a24444 + a_const::f1a24444 * sind +
               a_const::f2a24444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a244455 * sind +
               a_const::f2a244455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f0a24445 + a_const::f1a24445 * sind +
               a_const::f2a24445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) *
              (a_const::f0a2444 + a_const::f1a2444 * sind +
               a_const::f2a2444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a244555 * sind +
               a_const::f2a244555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a24455 + a_const::f1a24455 * sind +
               a_const::f2a24455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a2445 + a_const::f1a2445 * sind +
               a_const::f2a2445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) *
              (a_const::f0a244 + a_const::f1a244 * sind +
               a_const::f2a244 * std::pow(sind, 2) +
               a_const::f3a244 * std::pow(sind, 3)) +
          xi2 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a245555 * sind +
               a_const::f2a245555 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f0a24555 + a_const::f1a24555 * sind +
               a_const::f2a24555 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a2455 + a_const::f1a2455 * sind +
               a_const::f2a2455 * std::pow(sind, 2)) +
          xi2 * xi4 * xi5 *
              (a_const::f0a245 + a_const::f1a245 * sind +
               a_const::f2a245 * std::pow(sind, 2) +
               a_const::f3a245 * std::pow(sind, 3)) +
          xi2 * xi4 *
              (a_const::f0a24 + a_const::f1a24 * sind +
               a_const::f2a24 * std::pow(sind, 2) +
               a_const::f3a24 * std::pow(sind, 3) +
               a_const::f4a24 * std::pow(sind, 4)) +
          xi2 * std::pow(xi5, 5) *
              (a_const::f1a255555 * sind +
               a_const::f2a255555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 4) *
              (a_const::f0a25555 + a_const::f1a25555 * sind +
               a_const::f2a25555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 3) *
              (a_const::f0a2555 + a_const::f1a2555 * sind +
               a_const::f2a2555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 2) *
              (a_const::f0a255 + a_const::f1a255 * sind +
               a_const::f2a255 * std::pow(sind, 2) +
               a_const::f3a255 * std::pow(sind, 3)) +
          xi2 * xi5 *
              (a_const::f0a25 + a_const::f1a25 * sind +
               a_const::f2a25 * std::pow(sind, 2) +
               a_const::f3a25 * std::pow(sind, 3) +
               a_const::f4a25 * std::pow(sind, 4)) +
          xi2 * (a_const::f0a2 + a_const::f1a2 * sind +
                 a_const::f2a2 * std::pow(sind, 2) +
                 a_const::f3a2 * std::pow(sind, 3) +
                 a_const::f4a2 * std::pow(sind, 4) +
                 a_const::f5a2 * std::pow(sind, 5) +
                 a_const::f6a2 * std::pow(sind, 6)) +
          std::pow(xi3, 6) * (a_const::f1a333333 * sind +
                              a_const::f2a333333 * std::pow(sind, 2)) +
          std::pow(xi3, 5) * xi4 *
              (a_const::f1a333334 * sind +
               a_const::f2a333334 * std::pow(sind, 2)) +
          std::pow(xi3, 5) * xi5 *
              (a_const::f1a333335 * sind +
               a_const::f2a333335 * std::pow(sind, 2)) +
          std::pow(xi3, 5) * (a_const::f0a33333 + a_const::f1a33333 * sind +
                              a_const::f2a33333 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * std::pow(xi4, 2) *
              (a_const::f1a333344 * sind +
               a_const::f2a333344 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi4 * xi5 *
              (a_const::f1a333345 * sind +
               a_const::f2a333345 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi4 *
              (a_const::f0a33334 + a_const::f1a33334 * sind +
               a_const::f2a33334 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * std::pow(xi5, 2) *
              (a_const::f1a333355 * sind +
               a_const::f2a333355 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi5 *
              (a_const::f0a33335 + a_const::f1a33335 * sind +
               a_const::f2a33335 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * (a_const::f0a3333 + a_const::f1a3333 * sind +
                              a_const::f2a3333 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 3) *
              (a_const::f1a333444 * sind +
               a_const::f2a333444 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a333445 * sind +
               a_const::f2a333445 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f0a33344 + a_const::f1a33344 * sind +
               a_const::f2a33344 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a333455 * sind +
               a_const::f2a333455 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f0a33345 + a_const::f1a33345 * sind +
               a_const::f2a33345 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 *
              (a_const::f0a3334 + a_const::f1a3334 * sind +
               a_const::f2a3334 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi5, 3) *
              (a_const::f1a333555 * sind +
               a_const::f2a333555 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f0a33355 + a_const::f1a33355 * sind +
               a_const::f2a33355 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi5 *
              (a_const::f0a3335 + a_const::f1a3335 * sind +
               a_const::f2a3335 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * (a_const::f0a333 + a_const::f1a333 * sind +
                              a_const::f2a333 * std::pow(sind, 2) +
                              a_const::f3a333 * std::pow(sind, 3)) +
          std::pow(xi3, 2) * std::pow(xi4, 4) *
              (a_const::f1a334444 * sind +
               a_const::f2a334444 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a334445 * sind +
               a_const::f2a334445 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f0a33444 + a_const::f1a33444 * sind +
               a_const::f2a33444 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a334455 * sind +
               a_const::f2a334455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f0a33445 + a_const::f1a33445 * sind +
               a_const::f2a33445 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a3344 + a_const::f1a3344 * sind +
               a_const::f2a3344 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a334555 * sind +
               a_const::f2a334555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f0a33455 + a_const::f1a33455 * sind +
               a_const::f2a33455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a3345 + a_const::f1a3345 * sind +
               a_const::f2a3345 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 *
              (a_const::f0a334 + a_const::f1a334 * sind +
               a_const::f2a334 * std::pow(sind, 2) +
               a_const::f3a334 * std::pow(sind, 3)) +
          std::pow(xi3, 2) * std::pow(xi5, 4) *
              (a_const::f1a335555 * sind +
               a_const::f2a335555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f0a33555 + a_const::f1a33555 * sind +
               a_const::f2a33555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a3355 + a_const::f1a3355 * sind +
               a_const::f2a3355 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi5 *
              (a_const::f0a335 + a_const::f1a335 * sind +
               a_const::f2a335 * std::pow(sind, 2) +
               a_const::f3a335 * std::pow(sind, 3)) +
          std::pow(xi3, 2) * (a_const::f0a33 + a_const::f1a33 * sind +
                              a_const::f2a33 * std::pow(sind, 2) +
                              a_const::f3a33 * std::pow(sind, 3) +
                              a_const::f4a33 * std::pow(sind, 4)) +
          xi3 * std::pow(xi4, 5) *
              (a_const::f1a344444 * sind +
               a_const::f2a344444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a344445 * sind +
               a_const::f2a344445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 4) *
              (a_const::f0a34444 + a_const::f1a34444 * sind +
               a_const::f2a34444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a344455 * sind +
               a_const::f2a344455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f0a34445 + a_const::f1a34445 * sind +
               a_const::f2a34445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) *
              (a_const::f0a3444 + a_const::f1a3444 * sind +
               a_const::f2a3444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a344555 * sind +
               a_const::f2a344555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a34455 + a_const::f1a34455 * sind +
               a_const::f2a34455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a3445 + a_const::f1a3445 * sind +
               a_const::f2a3445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) *
              (a_const::f0a344 + a_const::f1a344 * sind +
               a_const::f2a344 * std::pow(sind, 2) +
               a_const::f3a344 * std::pow(sind, 3)) +
          xi3 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a345555 * sind +
               a_const::f2a345555 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f0a34555 + a_const::f1a34555 * sind +
               a_const::f2a34555 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a3455 + a_const::f1a3455 * sind +
               a_const::f2a3455 * std::pow(sind, 2)) +
          xi3 * xi4 * xi5 *
              (a_const::f0a345 + a_const::f1a345 * sind +
               a_const::f2a345 * std::pow(sind, 2) +
               a_const::f3a345 * std::pow(sind, 3)) +
          xi3 * xi4 *
              (a_const::f0a34 + a_const::f1a34 * sind +
               a_const::f2a34 * std::pow(sind, 2) +
               a_const::f3a34 * std::pow(sind, 3) +
               a_const::f4a34 * std::pow(sind, 4)) +
          xi3 * std::pow(xi5, 5) *
              (a_const::f1a355555 * sind +
               a_const::f2a355555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 4) *
              (a_const::f0a35555 + a_const::f1a35555 * sind +
               a_const::f2a35555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 3) *
              (a_const::f0a3555 + a_const::f1a3555 * sind +
               a_const::f2a3555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 2) *
              (a_const::f0a355 + a_const::f1a355 * sind +
               a_const::f2a355 * std::pow(sind, 2) +
               a_const::f3a355 * std::pow(sind, 3)) +
          xi3 * xi5 *
              (a_const::f0a35 + a_const::f1a35 * sind +
               a_const::f2a35 * std::pow(sind, 2) +
               a_const::f3a35 * std::pow(sind, 3) +
               a_const::f4a35 * std::pow(sind, 4)) +
          xi3 * (a_const::f0a3 + a_const::f1a3 * sind +
                 a_const::f2a3 * std::pow(sind, 2) +
                 a_const::f3a3 * std::pow(sind, 3) +
                 a_const::f4a3 * std::pow(sind, 4) +
                 a_const::f5a3 * std::pow(sind, 5) +
                 a_const::f6a3 * std::pow(sind, 6)) +
          std::pow(xi4, 6) * (a_const::f1a444444 * sind +
                              a_const::f2a444444 * std::pow(sind, 2)) +
          std::pow(xi4, 5) * xi5 *
              (a_const::f1a444445 * sind +
               a_const::f2a444445 * std::pow(sind, 2)) +
          std::pow(xi4, 5) * (a_const::f0a44444 + a_const::f1a44444 * sind +
                              a_const::f2a44444 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * std::pow(xi5, 2) *
              (a_const::f1a444455 * sind +
               a_const::f2a444455 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * xi5 *
              (a_const::f0a44445 + a_const::f1a44445 * sind +
               a_const::f2a44445 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * (a_const::f0a4444 + a_const::f1a4444 * sind +
                              a_const::f2a4444 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * std::pow(xi5, 3) *
              (a_const::f1a444555 * sind +
               a_const::f2a444555 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f0a44455 + a_const::f1a44455 * sind +
               a_const::f2a44455 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * xi5 *
              (a_const::f0a4445 + a_const::f1a4445 * sind +
               a_const::f2a4445 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * (a_const::f0a444 + a_const::f1a444 * sind +
                              a_const::f2a444 * std::pow(sind, 2) +
                              a_const::f3a444 * std::pow(sind, 3)) +
          std::pow(xi4, 2) * std::pow(xi5, 4) *
              (a_const::f1a445555 * sind +
               a_const::f2a445555 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f0a44555 + a_const::f1a44555 * sind +
               a_const::f2a44555 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a4455 + a_const::f1a4455 * sind +
               a_const::f2a4455 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * xi5 *
              (a_const::f0a445 + a_const::f1a445 * sind +
               a_const::f2a445 * std::pow(sind, 2) +
               a_const::f3a445 * std::pow(sind, 3)) +
          std::pow(xi4, 2) * (a_const::f0a44 + a_const::f1a44 * sind +
                              a_const::f2a44 * std::pow(sind, 2) +
                              a_const::f3a44 * std::pow(sind, 3) +
                              a_const::f4a44 * std::pow(sind, 4)) +
          xi4 * std::pow(xi5, 5) *
              (a_const::f1a455555 * sind +
               a_const::f2a455555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 4) *
              (a_const::f0a45555 + a_const::f1a45555 * sind +
               a_const::f2a45555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 3) *
              (a_const::f0a4555 + a_const::f1a4555 * sind +
               a_const::f2a4555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 2) *
              (a_const::f0a455 + a_const::f1a455 * sind +
               a_const::f2a455 * std::pow(sind, 2) +
               a_const::f3a455 * std::pow(sind, 3)) +
          xi4 * xi5 *
              (a_const::f0a45 + a_const::f1a45 * sind +
               a_const::f2a45 * std::pow(sind, 2) +
               a_const::f3a45 * std::pow(sind, 3) +
               a_const::f4a45 * std::pow(sind, 4)) +
          xi4 * (a_const::f0a4 + a_const::f1a4 * sind +
                 a_const::f2a4 * std::pow(sind, 2) +
                 a_const::f3a4 * std::pow(sind, 3) +
                 a_const::f4a4 * std::pow(sind, 4) +
                 a_const::f5a4 * std::pow(sind, 5) +
                 a_const::f6a4 * std::pow(sind, 6)) +
          std::pow(xi5, 6) * (a_const::f1a555555 * sind +
                              a_const::f2a555555 * std::pow(sind, 2)) +
          std::pow(xi5, 5) * (a_const::f0a55555 + a_const::f1a55555 * sind +
                              a_const::f2a55555 * std::pow(sind, 2)) +
          std::pow(xi5, 4) * (a_const::f0a5555 + a_const::f1a5555 * sind +
                              a_const::f2a5555 * std::pow(sind, 2)) +
          std::pow(xi5, 3) * (a_const::f0a555 + a_const::f1a555 * sind +
                              a_const::f2a555 * std::pow(sind, 2) +
                              a_const::f3a555 * std::pow(sind, 3)) +
          std::pow(xi5, 2) * (a_const::f0a55 + a_const::f1a55 * sind +
                              a_const::f2a55 * std::pow(sind, 2) +
                              a_const::f3a55 * std::pow(sind, 3) +
                              a_const::f4a55 * std::pow(sind, 4)) +
          xi5 * (a_const::f0a5 + a_const::f1a5 * sind +
                 a_const::f2a5 * std::pow(sind, 2) +
                 a_const::f3a5 * std::pow(sind, 3) +
                 a_const::f4a5 * std::pow(sind, 4) +
                 a_const::f5a5 * std::pow(sind, 5) +
                 a_const::f6a5 * std::pow(sind, 6));
    }
  }
  return v;
}

// requires HHHN ordering
arma::cube AmmoniaPES::Force(const arma::cube &q) {
  arma::cube f(arma::size(q), arma::fill::zeros);

  // real(dp), dimension(nmolec, 4, 3), intent(in) :: q
  // real(dp), dimension, intent(out) :: v
  // real(dp), dimension(nmolec, 4, 3), intent(out) :: f

  double xh1, yh1, zh1;
  double xh2, yh2, zh2;
  double xh3, yh3, zh3;
  double xn, yn, zn;

  arma::rowvec r1vec(3);
  arma::rowvec r2vec(3);
  arma::rowvec r3vec(3);
  double r1;
  double r2;
  double r3;

  double alp1;
  double alp2;

  double alp3;

  double xi1, xi2, xi3, xi4, xi5, sind;

  double dxi1dxn, dxi1dyn, dxi1dzn, dxi1dxh1, dxi1dyh1, dxi1dzh1, dxi1dxh2,
      dxi1dyh2, dxi1dzh2, dxi1dxh3, dxi1dyh3, dxi1dzh3, dxi2dxn, dxi2dyn,
      dxi2dzn, dxi2dxh1, dxi2dyh1, dxi2dzh1, dxi2dxh2, dxi2dyh2, dxi2dzh2,
      dxi2dxh3, dxi2dyh3, dxi2dzh3, dxi3dxn, dxi3dyn, dxi3dzn, dxi3dxh1,
      dxi3dyh1, dxi3dzh1, dxi3dxh2, dxi3dyh2, dxi3dzh2, dxi3dxh3, dxi3dyh3,
      dxi3dzh3, dxi4dxn, dxi4dyn, dxi4dzn, dxi4dxh1, dxi4dyh1, dxi4dzh1,
      dxi4dxh2, dxi4dyh2, dxi4dzh2, dxi4dxh3, dxi4dyh3, dxi4dzh3, dxi5dxn,
      dxi5dyn, dxi5dzn, dxi5dxh1, dxi5dyh1, dxi5dzh1, dxi5dxh2, dxi5dyh2,
      dxi5dzh2, dxi5dxh3, dxi5dyh3, dxi5dzh3, dsinddxn, dsinddyn, dsinddzn,
      dsinddxh1, dsinddyh1, dsinddzh1, dsinddxh2, dsinddyh2, dsinddzh2,
      dsinddxh3, dsinddyh3, dsinddzh3, dvdxi1, dvdxi2, dvdxi3, dvdxi4, dvdxi5,
      dvdsind, f_xn, f_yn, f_zn, f_xh1, f_yh1, f_zh1, f_xh2, f_yh2, f_zh2,
      f_xh3, f_yh3, f_zh3;

  for (arma::uword i = 0; i < q.n_slices; i += 4) {
    for (arma::uword j = 0; j < q.n_rows; j++) {
      xh1 = arma::as_scalar(q(j, 0, i));
      yh1 = arma::as_scalar(q(j, 1, i));
      zh1 = arma::as_scalar(q(j, 2, i));
      xh2 = arma::as_scalar(q(j, 0, i + 1));
      yh2 = arma::as_scalar(q(j, 1, i + 1));
      zh2 = arma::as_scalar(q(j, 2, i + 1));
      xh3 = arma::as_scalar(q(j, 0, i + 2));
      yh3 = arma::as_scalar(q(j, 1, i + 2));
      zh3 = arma::as_scalar(q(j, 2, i + 2));
      xn = arma::as_scalar(q(j, 0, i + 3));
      yn = arma::as_scalar(q(j, 1, i + 3));
      zn = arma::as_scalar(q(j, 2, i + 3));

      r1vec = q.slice(i + 3).row(j) - q.slice(i).row(j);
      r2vec = q.slice(i + 3).row(j) - q.slice(i + 1).row(j);
      r3vec = q.slice(i + 3).row(j) - q.slice(i + 2).row(j);

      // r1vec = q(j, 4, :) - q(j, 1, :);
      // r2vec = q(j, 4, :) - q(j, 2, :);
      // r3vec = q(j, 4, :) - q(j, 3, :);

      // r1 = std::sqrt(sum(r1vec * r1vec, dim = 2));
      // r2 = std::sqrt(sum(r2vec * r2vec, dim = 2));
      // r3 = std::sqrt(sum(r3vec * r3vec, dim = 2));

      r1 = arma::norm(r1vec);
      r2 = arma::norm(r2vec);
      r3 = arma::norm(r3vec);

      // alp1 = std::acos(sum(r2vec * r3vec, dim = 2) / (r2 * r3));
      // alp2 = std::acos(sum(r1vec * r3vec, dim = 2) / (r1 * r3));
      // alp3= std::acos(sum(r1vec * r2vec, dim = 2) / (r1 * r2));

      alp1 = std::acos(arma::norm_dot(r2vec, r3vec));
      alp2 = std::acos(arma::norm_dot(r1vec, r3vec));
      alp3 = std::acos(arma::norm_dot(r1vec, r2vec));

      // std::cout << r1 << " " << r2 << " " << r3 << std::endl;
      // std::cout << alp1 << " " << alp2 << " " << alp3 << std::endl;

      xi1 = 1.0 - std::exp(-a_const::a * (r1 - a_const::re));
      xi2 = 1.0 - std::exp(-a_const::a * (r2 - a_const::re));
      xi3 = 1.0 - std::exp(-a_const::a * (r3 - a_const::re));
      xi4 = (2.0 * alp1 - alp2 - alp3) / std::sqrt(6.0);
      xi5 = (alp2 - alp3) / std::sqrt(2.0);
      sind =
          -((2.0 / std::sqrt(3.0)) * std::sin((alp1 + alp2 + alp3) / 6.0) -
            std::sin(a_const::rhoe));

      dxi1dxn =
          a_const::a * (-xh1 + xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dyn =
          a_const::a * (-yh1 + yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dzn =
          a_const::a * (-zh1 + zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dxh1 =
          a_const::a * (xh1 - xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dyh1 =
          a_const::a * (yh1 - yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dzh1 =
          a_const::a * (zh1 - zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh1 + xn, 2) +
                                                std::pow(-yh1 + yn, 2) +
                                                std::pow(-zh1 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh1 + xn, 2) + std::pow(-yh1 + yn, 2) +
              std::pow(-zh1 + zn, 2));
      dxi1dxh2 = 0;
      dxi1dyh2 = 0;
      dxi1dzh2 = 0;
      dxi1dxh3 = 0;
      dxi1dyh3 = 0;
      dxi1dzh3 = 0;
      dxi2dxn =
          a_const::a * (-xh2 + xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dyn =
          a_const::a * (-yh2 + yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dzn =
          a_const::a * (-zh2 + zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dxh1 = 0;
      dxi2dyh1 = 0;
      dxi2dzh1 = 0;
      dxi2dxh2 =
          a_const::a * (xh2 - xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dyh2 =
          a_const::a * (yh2 - yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dzh2 =
          a_const::a * (zh2 - zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh2 + xn, 2) +
                                                std::pow(-yh2 + yn, 2) +
                                                std::pow(-zh2 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh2 + xn, 2) + std::pow(-yh2 + yn, 2) +
              std::pow(-zh2 + zn, 2));
      dxi2dxh3 = 0;
      dxi2dyh3 = 0;
      dxi2dzh3 = 0;
      dxi3dxn =
          a_const::a * (-xh3 + xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi3dyn =
          a_const::a * (-yh3 + yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi3dzn =
          a_const::a * (-zh3 + zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi3dxh1 = 0;
      dxi3dyh1 = 0;
      dxi3dzh1 = 0;
      dxi3dxh2 = 0;
      dxi3dyh2 = 0;
      dxi3dzh2 = 0;
      dxi3dxh3 =
          a_const::a * (xh3 - xn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi3dyh3 =
          a_const::a * (yh3 - yn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi3dzh3 =
          a_const::a * (zh3 - zn) *
          exp(-a_const::a * (-a_const::re + std::sqrt(
                                                std::pow(-xh3 + xn, 2) +
                                                std::pow(-yh3 + yn, 2) +
                                                std::pow(-zh3 + zn, 2)))) /
          std::sqrt(
              std::pow(-xh3 + xn, 2) + std::pow(-yh3 + yn, 2) +
              std::pow(-zh3 + zn, 2));
      dxi4dxn = -0.81649658092772615 *
                    ((xh2 - xn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (xh3 - xn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-xh2 - xh3 + 2 * xn) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                                (zh2 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((xh1 - xn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (xh3 - xn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-xh1 - xh3 + 2 * xn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((xh1 - xn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (xh2 - xn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-xh1 - xh2 + 2 * xn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi4dyn = -0.81649658092772615 *
                    ((yh2 - yn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (yh3 - yn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-yh2 - yh3 + 2 * yn) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                                (zh2 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((yh1 - yn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (yh3 - yn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-yh1 - yh3 + 2 * yn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((yh1 - yn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (yh2 - yn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-yh1 - yh2 + 2 * yn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi4dzn = -0.81649658092772615 *
                    ((zh2 - zn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (zh3 - zn) *
                         ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-zh2 - zh3 + 2 * zn) /
                         (std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                                (zh2 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((zh1 - zn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (zh3 - zn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-zh1 - zh3 + 2 * zn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.40824829046386307 *
                    ((zh1 - zn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (zh2 - zn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-zh1 - zh2 + 2 * zn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi4dxh1 =
          0.40824829046386307 *
              ((-xh1 + xn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (xh3 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.40824829046386307 *
              ((-xh1 + xn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (xh2 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi4dyh1 =
          0.40824829046386307 *
              ((-yh1 + yn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (yh3 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.40824829046386307 *
              ((-yh1 + yn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (yh2 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi4dzh1 =
          0.40824829046386307 *
              ((-zh1 + zn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (zh3 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.40824829046386307 *
              ((-zh1 + zn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (zh2 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi4dxh2 =
          0.40824829046386307 *
              ((xh1 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2))) +
               (-xh2 + xn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((-xh2 + xn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (xh3 - xn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi4dyh2 =
          0.40824829046386307 *
              ((yh1 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2))) +
               (-yh2 + yn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((-yh2 + yn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (yh3 - yn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi4dzh2 =
          0.40824829046386307 *
              ((zh1 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2))) +
               (-zh2 + zn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((-zh2 + zn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                            std::pow(zh2 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (zh3 - zn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi4dxh3 =
          0.40824829046386307 *
              ((xh1 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-xh3 + xn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((xh2 - xn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-xh3 + xn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi4dyh3 =
          0.40824829046386307 *
              ((yh1 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-yh3 + yn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((yh2 - yn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-yh3 + yn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi4dzh3 =
          0.40824829046386307 *
              ((zh1 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-zh3 + zn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) -
          0.81649658092772615 *
              ((zh2 - zn) / (std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2))) +
               (-zh3 + zn) *
                   ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                    (zh2 - zn) * (zh3 - zn)) /
                   (std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                    std::pow(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                            std::pow(zh3 - zn, 2),
                        3.0 / 2.0))) /
              std::sqrt(
                  -std::pow(
                      (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                          (zh2 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1);
      dxi5dxn = -0.70710678118654746 *
                    ((xh1 - xn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (xh3 - xn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-xh1 - xh3 + 2 * xn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.70710678118654746 *
                    ((xh1 - xn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (xh2 - xn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-xh1 - xh2 + 2 * xn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi5dyn = -0.70710678118654746 *
                    ((yh1 - yn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (yh3 - yn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-yh1 - yh3 + 2 * yn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.70710678118654746 *
                    ((yh1 - yn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (yh2 - yn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-yh1 - yh2 + 2 * yn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi5dzn = -0.70710678118654746 *
                    ((zh1 - zn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                     (zh3 - zn) *
                         ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                  std::pow(zh3 - zn, 2),
                              3.0 / 2.0)) +
                     (-zh1 - zh3 + 2 * zn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                                (zh1 - zn) * (zh3 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                              std::pow(zh3 - zn, 2))) +
                        1) +
                0.70710678118654746 *
                    ((zh1 - zn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::pow(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                  std::pow(zh1 - zn, 2),
                              3.0 / 2.0) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                     (zh2 - zn) *
                         ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn)) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::pow(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                  std::pow(zh2 - zn, 2),
                              3.0 / 2.0)) +
                     (-zh1 - zh2 + 2 * zn) /
                         (std::sqrt(
                              std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                          std::sqrt(
                              std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2)))) /
                    std::sqrt(
                        -std::pow(
                            (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                                (zh1 - zn) * (zh2 - zn),
                            2) /
                            ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                              std::pow(zh1 - zn, 2)) *
                             (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                              std::pow(zh2 - zn, 2))) +
                        1);
      dxi5dxh1 =
          -0.70710678118654746 *
              ((-xh1 + xn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (xh3 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.70710678118654746 *
              ((-xh1 + xn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (xh2 - xn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi5dyh1 =
          -0.70710678118654746 *
              ((-yh1 + yn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (yh3 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.70710678118654746 *
              ((-yh1 + yn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (yh2 - yn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi5dzh1 =
          -0.70710678118654746 *
              ((-zh1 + zn) *
                   ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                    (zh1 - zn) * (zh3 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
               (zh3 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                                 std::pow(zh3 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                          (zh1 - zn) * (zh3 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2))) +
                  1) +
          0.70710678118654746 *
              ((-zh1 + zn) *
                   ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                    (zh1 - zn) * (zh2 - zn)) /
                   (std::pow(
                        std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                            std::pow(zh1 - zn, 2),
                        3.0 / 2.0) *
                    std::sqrt(
                        std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
               (zh2 - zn) / (std::sqrt(
                                 std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                                 std::pow(zh1 - zn, 2)) *
                             std::sqrt(
                                 std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                                 std::pow(zh2 - zn, 2)))) /
              std::sqrt(
                  -std::pow(
                      (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                          (zh1 - zn) * (zh2 - zn),
                      2) /
                      ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                        std::pow(zh1 - zn, 2)) *
                       (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2))) +
                  1);
      dxi5dxh2 =
          0.70710678118654746 *
          ((xh1 - xn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2))) +
           (-xh2 + xn) *
               ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                (zh1 - zn) * (zh2 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                      (zh1 - zn) * (zh2 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                    std::pow(zh2 - zn, 2))) +
              1);
      dxi5dyh2 =
          0.70710678118654746 *
          ((yh1 - yn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2))) +
           (-yh2 + yn) *
               ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                (zh1 - zn) * (zh2 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                      (zh1 - zn) * (zh2 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                    std::pow(zh2 - zn, 2))) +
              1);
      dxi5dzh2 =
          0.70710678118654746 *
          ((zh1 - zn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2))) +
           (-zh2 + zn) *
               ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                (zh1 - zn) * (zh2 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                        std::pow(zh2 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                      (zh1 - zn) * (zh2 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                    std::pow(zh2 - zn, 2))) +
              1);
      dxi5dxh3 =
          -0.70710678118654746 *
          ((xh1 - xn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2))) +
           (-xh3 + xn) *
               ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                (zh1 - zn) * (zh3 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                      (zh1 - zn) * (zh3 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                    std::pow(zh3 - zn, 2))) +
              1);
      dxi5dyh3 =
          -0.70710678118654746 *
          ((yh1 - yn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2))) +
           (-yh3 + yn) *
               ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                (zh1 - zn) * (zh3 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                      (zh1 - zn) * (zh3 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                    std::pow(zh3 - zn, 2))) +
              1);
      dxi5dzh3 =
          -0.70710678118654746 *
          ((zh1 - zn) / (std::sqrt(
                             std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2)) *
                         std::sqrt(
                             std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2))) +
           (-zh3 + zn) *
               ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                (zh1 - zn) * (zh3 - zn)) /
               (std::sqrt(
                    std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                std::pow(
                    std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                        std::pow(zh3 - zn, 2),
                    3.0 / 2.0))) /
          std::sqrt(
              -std::pow(
                  (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                      (zh1 - zn) * (zh3 - zn),
                  2) /
                  ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                    std::pow(zh1 - zn, 2)) *
                   (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                    std::pow(zh3 - zn, 2))) +
              1);
      dsinddxn =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((xh2 - xn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (xh3 - xn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-xh2 - xh3 + 2 * xn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((xh1 - xn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (xh3 - xn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-xh1 - xh3 + 2 * xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((xh1 - xn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (xh2 - xn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0)) +
                (-xh1 - xh2 + 2 * xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddyn =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((yh2 - yn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (yh3 - yn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-yh2 - yh3 + 2 * yn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((yh1 - yn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (yh3 - yn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-yh1 - yh3 + 2 * yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((yh1 - yn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (yh2 - yn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0)) +
                (-yh1 - yh2 + 2 * yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddzn =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((zh2 - zn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (zh3 - zn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-zh2 - zh3 + 2 * zn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((zh1 - zn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (zh3 - zn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0)) +
                (-zh1 - zh3 + 2 * zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((zh1 - zn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (zh2 - zn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0)) +
                (-zh1 - zh2 + 2 * zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddxh1 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((-xh1 + xn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (xh3 - xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-xh1 + xn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (xh2 - xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddyh1 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((-yh1 + yn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (yh3 - yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-yh1 + yn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (yh2 - yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddzh1 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((-zh1 + zn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (zh3 - zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-zh1 + zn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::pow(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                             std::pow(zh1 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (zh2 - zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddxh2 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((xh1 - xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (-xh2 + xn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-xh2 + xn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (xh3 - xn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddyh2 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((yh1 - yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (-yh2 + yn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-yh2 + yn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (yh3 - yn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddzh2 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((zh1 - zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                (-zh2 + zn) *
                    ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                     (zh1 - zn) * (zh2 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                           (zh1 - zn) * (zh2 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((-zh2 + zn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::pow(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                             std::pow(zh2 - zn, 2),
                         3.0 / 2.0) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (zh3 - zn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2)))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddxh3 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((xh1 - xn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-xh3 + xn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((xh2 - xn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-xh3 + xn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddyh3 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((yh1 - yn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-yh3 + yn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((yh2 - yn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-yh3 + yn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dsinddzh3 =
          -1.1547005383792517 *
          (-0.16666666666666666 *
               ((zh1 - zn) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-zh3 + zn) *
                    ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                     (zh1 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                           (zh1 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                         std::pow(zh1 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1) -
           0.16666666666666666 *
               ((zh2 - zn) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::sqrt(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                (-zh3 + zn) *
                    ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                     (zh2 - zn) * (zh3 - zn)) /
                    (std::sqrt(
                         std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                     std::pow(
                         std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                             std::pow(zh3 - zn, 2),
                         3.0 / 2.0))) /
               std::sqrt(
                   -std::pow(
                       (xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                           (zh2 - zn) * (zh3 - zn),
                       2) /
                       ((std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                         std::pow(zh2 - zn, 2)) *
                        (std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                         std::pow(zh3 - zn, 2))) +
                   1)) *
          cos(0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh2 - xn) + (yh1 - yn) * (yh2 - yn) +
                       (zh1 - zn) * (zh2 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh1 - xn) * (xh3 - xn) + (yh1 - yn) * (yh3 - yn) +
                       (zh1 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh1 - xn, 2) + std::pow(yh1 - yn, 2) +
                           std::pow(zh1 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))) +
              0.16666666666666666 *
                  acos(
                      ((xh2 - xn) * (xh3 - xn) + (yh2 - yn) * (yh3 - yn) +
                       (zh2 - zn) * (zh3 - zn)) /
                      (std::sqrt(
                           std::pow(xh2 - xn, 2) + std::pow(yh2 - yn, 2) +
                           std::pow(zh2 - zn, 2)) *
                       std::sqrt(
                           std::pow(xh3 - xn, 2) + std::pow(yh3 - yn, 2) +
                           std::pow(zh3 - zn, 2)))));
      dvdxi1 =
          a_const::f0a1 + a_const::f1a1 * sind +
          a_const::f2a1 * std::pow(sind, 2) +
          a_const::f3a1 * std::pow(sind, 3) +
          a_const::f4a1 * std::pow(sind, 4) +
          a_const::f5a1 * std::pow(sind, 5) +
          a_const::f6a1 * std::pow(sind, 6) +
          6 * std::pow(xi1, 5) *
              (a_const::f1a111111 * sind +
               a_const::f2a111111 * std::pow(sind, 2)) +
          5 * std::pow(xi1, 4) * xi2 *
              (a_const::f1a111112 * sind +
               a_const::f2a111112 * std::pow(sind, 2)) +
          5 * std::pow(xi1, 4) * xi3 *
              (a_const::f1a111113 * sind +
               a_const::f2a111113 * std::pow(sind, 2)) +
          5 * std::pow(xi1, 4) * xi4 *
              (a_const::f1a111114 * sind +
               a_const::f2a111114 * std::pow(sind, 2)) +
          5 * std::pow(xi1, 4) * xi5 *
              (a_const::f1a111115 * sind +
               a_const::f2a111115 * std::pow(sind, 2)) +
          5 * std::pow(xi1, 4) *
              (a_const::f0a11111 + a_const::f1a11111 * sind +
               a_const::f2a11111 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a111122 * sind +
               a_const::f2a111122 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a111123 * sind +
               a_const::f2a111123 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a111124 * sind +
               a_const::f2a111124 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a111125 * sind +
               a_const::f2a111125 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi2 *
              (a_const::f0a11112 + a_const::f1a11112 * sind +
               a_const::f2a11112 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a111133 * sind +
               a_const::f2a111133 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a111134 * sind +
               a_const::f2a111134 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a111135 * sind +
               a_const::f2a111135 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi3 *
              (a_const::f0a11113 + a_const::f1a11113 * sind +
               a_const::f2a11113 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a111144 * sind +
               a_const::f2a111144 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a111145 * sind +
               a_const::f2a111145 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi4 *
              (a_const::f0a11114 + a_const::f1a11114 * sind +
               a_const::f2a11114 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a111155 * sind +
               a_const::f2a111155 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) * xi5 *
              (a_const::f0a11115 + a_const::f1a11115 * sind +
               a_const::f2a11115 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 3) *
              (a_const::f0a1111 + a_const::f1a1111 * sind +
               a_const::f2a1111 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a111222 * sind +
               a_const::f2a111222 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a111223 * sind +
               a_const::f2a111223 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a111224 * sind +
               a_const::f2a111224 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a111225 * sind +
               a_const::f2a111225 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a11122 + a_const::f1a11122 * sind +
               a_const::f2a11122 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a111233 * sind +
               a_const::f2a111233 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a111234 * sind +
               a_const::f2a111234 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a111235 * sind +
               a_const::f2a111235 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a11123 + a_const::f1a11123 * sind +
               a_const::f2a11123 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a111244 * sind +
               a_const::f2a111244 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a111245 * sind +
               a_const::f2a111245 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a11124 + a_const::f1a11124 * sind +
               a_const::f2a11124 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a111255 * sind +
               a_const::f2a111255 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a11125 + a_const::f1a11125 * sind +
               a_const::f2a11125 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 *
              (a_const::f0a1112 + a_const::f1a1112 * sind +
               a_const::f2a1112 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a111333 * sind +
               a_const::f2a111333 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a111334 * sind +
               a_const::f2a111334 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a111335 * sind +
               a_const::f2a111335 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a11133 + a_const::f1a11133 * sind +
               a_const::f2a11133 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a111344 * sind +
               a_const::f2a111344 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a111345 * sind +
               a_const::f2a111345 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a11134 + a_const::f1a11134 * sind +
               a_const::f2a11134 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a111355 * sind +
               a_const::f2a111355 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a11135 + a_const::f1a11135 * sind +
               a_const::f2a11135 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 *
              (a_const::f0a1113 + a_const::f1a1113 * sind +
               a_const::f2a1113 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a111444 * sind +
               a_const::f2a111444 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a111445 * sind +
               a_const::f2a111445 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a11144 + a_const::f1a11144 * sind +
               a_const::f2a11144 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a111455 * sind +
               a_const::f2a111455 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a11145 + a_const::f1a11145 * sind +
               a_const::f2a11145 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi4 *
              (a_const::f0a1114 + a_const::f1a1114 * sind +
               a_const::f2a1114 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a111555 * sind +
               a_const::f2a111555 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a11155 + a_const::f1a11155 * sind +
               a_const::f2a11155 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi5 *
              (a_const::f0a1115 + a_const::f1a1115 * sind +
               a_const::f2a1115 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) *
              (a_const::f0a111 + a_const::f1a111 * sind +
               a_const::f2a111 * std::pow(sind, 2) +
               a_const::f3a111 * std::pow(sind, 3)) +
          2 * xi1 * std::pow(xi2, 4) *
              (a_const::f1a112222 * sind +
               a_const::f2a112222 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a112223 * sind +
               a_const::f2a112223 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a112224 * sind +
               a_const::f2a112224 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a112225 * sind +
               a_const::f2a112225 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) *
              (a_const::f0a11222 + a_const::f1a11222 * sind +
               a_const::f2a11222 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a112233 * sind +
               a_const::f2a112233 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a112234 * sind +
               a_const::f2a112234 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a112235 * sind +
               a_const::f2a112235 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a11223 + a_const::f1a11223 * sind +
               a_const::f2a11223 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a112244 * sind +
               a_const::f2a112244 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a112245 * sind +
               a_const::f2a112245 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a11224 + a_const::f1a11224 * sind +
               a_const::f2a11224 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a112255 * sind +
               a_const::f2a112255 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a11225 + a_const::f1a11225 * sind +
               a_const::f2a11225 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) *
              (a_const::f0a1122 + a_const::f1a1122 * sind +
               a_const::f2a1122 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a112333 * sind +
               a_const::f2a112333 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a112334 * sind +
               a_const::f2a112334 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a112335 * sind +
               a_const::f2a112335 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a11233 + a_const::f1a11233 * sind +
               a_const::f2a11233 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a112344 * sind +
               a_const::f2a112344 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a11234 + a_const::f1a11234 * sind +
               a_const::f2a11234 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a112355 * sind +
               a_const::f2a112355 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a11235 + a_const::f1a11235 * sind +
               a_const::f2a11235 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 *
              (a_const::f0a1123 + a_const::f1a1123 * sind +
               a_const::f2a1123 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a112444 * sind +
               a_const::f2a112444 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a112445 * sind +
               a_const::f2a112445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a11244 + a_const::f1a11244 * sind +
               a_const::f2a11244 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a112455 * sind +
               a_const::f2a112455 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a11245 + a_const::f1a11245 * sind +
               a_const::f2a11245 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 *
              (a_const::f0a1124 + a_const::f1a1124 * sind +
               a_const::f2a1124 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a112555 * sind +
               a_const::f2a112555 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a11255 + a_const::f1a11255 * sind +
               a_const::f2a11255 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi5 *
              (a_const::f0a1125 + a_const::f1a1125 * sind +
               a_const::f2a1125 * std::pow(sind, 2)) +
          2 * xi1 * xi2 *
              (a_const::f0a112 + a_const::f1a112 * sind +
               a_const::f2a112 * std::pow(sind, 2) +
               a_const::f3a112 * std::pow(sind, 3)) +
          2 * xi1 * std::pow(xi3, 4) *
              (a_const::f1a113333 * sind +
               a_const::f2a113333 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a113334 * sind +
               a_const::f2a113334 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a113335 * sind +
               a_const::f2a113335 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 3) *
              (a_const::f0a11333 + a_const::f1a11333 * sind +
               a_const::f2a11333 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a113344 * sind +
               a_const::f2a113344 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a113345 * sind +
               a_const::f2a113345 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a11334 + a_const::f1a11334 * sind +
               a_const::f2a11334 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a113355 * sind +
               a_const::f2a113355 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a11335 + a_const::f1a11335 * sind +
               a_const::f2a11335 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) *
              (a_const::f0a1133 + a_const::f1a1133 * sind +
               a_const::f2a1133 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a113444 * sind +
               a_const::f2a113444 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a113445 * sind +
               a_const::f2a113445 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a11344 + a_const::f1a11344 * sind +
               a_const::f2a11344 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a113455 * sind +
               a_const::f2a113455 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a11345 + a_const::f1a11345 * sind +
               a_const::f2a11345 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 *
              (a_const::f0a1134 + a_const::f1a1134 * sind +
               a_const::f2a1134 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a113555 * sind +
               a_const::f2a113555 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a11355 + a_const::f1a11355 * sind +
               a_const::f2a11355 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi5 *
              (a_const::f0a1135 + a_const::f1a1135 * sind +
               a_const::f2a1135 * std::pow(sind, 2)) +
          2 * xi1 * xi3 *
              (a_const::f0a113 + a_const::f1a113 * sind +
               a_const::f2a113 * std::pow(sind, 2) +
               a_const::f3a113 * std::pow(sind, 3)) +
          2 * xi1 * std::pow(xi4, 4) *
              (a_const::f1a114444 * sind +
               a_const::f2a114444 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a114445 * sind +
               a_const::f2a114445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 3) *
              (a_const::f0a11444 + a_const::f1a11444 * sind +
               a_const::f2a11444 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a114455 * sind +
               a_const::f2a114455 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a11445 + a_const::f1a11445 * sind +
               a_const::f2a11445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 2) *
              (a_const::f0a1144 + a_const::f1a1144 * sind +
               a_const::f2a1144 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a114555 * sind +
               a_const::f2a114555 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a11455 + a_const::f1a11455 * sind +
               a_const::f2a11455 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * xi5 *
              (a_const::f0a1145 + a_const::f1a1145 * sind +
               a_const::f2a1145 * std::pow(sind, 2)) +
          2 * xi1 * xi4 *
              (a_const::f0a114 + a_const::f1a114 * sind +
               a_const::f2a114 * std::pow(sind, 2) +
               a_const::f3a114 * std::pow(sind, 3)) +
          2 * xi1 * std::pow(xi5, 4) *
              (a_const::f1a115555 * sind +
               a_const::f2a115555 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi5, 3) *
              (a_const::f0a11555 + a_const::f1a11555 * sind +
               a_const::f2a11555 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi5, 2) *
              (a_const::f0a1155 + a_const::f1a1155 * sind +
               a_const::f2a1155 * std::pow(sind, 2)) +
          2 * xi1 * xi5 *
              (a_const::f0a115 + a_const::f1a115 * sind +
               a_const::f2a115 * std::pow(sind, 2) +
               a_const::f3a115 * std::pow(sind, 3)) +
          2 * xi1 *
              (a_const::f0a11 + a_const::f1a11 * sind +
               a_const::f2a11 * std::pow(sind, 2) +
               a_const::f3a11 * std::pow(sind, 3) +
               a_const::f4a11 * std::pow(sind, 4)) +
          std::pow(xi2, 5) * (a_const::f1a122222 * sind +
                              a_const::f2a122222 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 *
              (a_const::f1a122223 * sind +
               a_const::f2a122223 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi4 *
              (a_const::f1a122224 * sind +
               a_const::f2a122224 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi5 *
              (a_const::f1a122225 * sind +
               a_const::f2a122225 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * (a_const::f0a12222 + a_const::f1a12222 * sind +
                              a_const::f2a12222 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a122233 * sind +
               a_const::f2a122233 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a122234 * sind +
               a_const::f2a122234 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a122235 * sind +
               a_const::f2a122235 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 *
              (a_const::f0a12223 + a_const::f1a12223 * sind +
               a_const::f2a12223 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a122244 * sind +
               a_const::f2a122244 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a122245 * sind +
               a_const::f2a122245 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 *
              (a_const::f0a12224 + a_const::f1a12224 * sind +
               a_const::f2a12224 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a122255 * sind +
               a_const::f2a122255 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi5 *
              (a_const::f0a12225 + a_const::f1a12225 * sind +
               a_const::f2a12225 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * (a_const::f0a1222 + a_const::f1a1222 * sind +
                              a_const::f2a1222 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a122333 * sind +
               a_const::f2a122333 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a122334 * sind +
               a_const::f2a122334 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a122335 * sind +
               a_const::f2a122335 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a12233 + a_const::f1a12233 * sind +
               a_const::f2a12233 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a122344 * sind +
               a_const::f2a122344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a12234 + a_const::f1a12234 * sind +
               a_const::f2a12234 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a122355 * sind +
               a_const::f2a122355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a12235 + a_const::f1a12235 * sind +
               a_const::f2a12235 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 *
              (a_const::f0a1223 + a_const::f1a1223 * sind +
               a_const::f2a1223 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a122444 * sind +
               a_const::f2a122444 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a122445 * sind +
               a_const::f2a122445 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a12244 + a_const::f1a12244 * sind +
               a_const::f2a12244 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a122455 * sind +
               a_const::f2a122455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a12245 + a_const::f1a12245 * sind +
               a_const::f2a12245 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 *
              (a_const::f0a1224 + a_const::f1a1224 * sind +
               a_const::f2a1224 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a122555 * sind +
               a_const::f2a122555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a12255 + a_const::f1a12255 * sind +
               a_const::f2a12255 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi5 *
              (a_const::f0a1225 + a_const::f1a1225 * sind +
               a_const::f2a1225 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * (a_const::f0a122 + a_const::f1a122 * sind +
                              a_const::f2a122 * std::pow(sind, 2) +
                              a_const::f3a122 * std::pow(sind, 3)) +
          xi2 * std::pow(xi3, 4) *
              (a_const::f1a123333 * sind +
               a_const::f2a123333 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a123334 * sind +
               a_const::f2a123334 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a123335 * sind +
               a_const::f2a123335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) *
              (a_const::f0a12333 + a_const::f1a12333 * sind +
               a_const::f2a12333 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a123344 * sind +
               a_const::f2a123344 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a12334 + a_const::f1a12334 * sind +
               a_const::f2a12334 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a123355 * sind +
               a_const::f2a123355 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a12335 + a_const::f1a12335 * sind +
               a_const::f2a12335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) *
              (a_const::f0a1233 + a_const::f1a1233 * sind +
               a_const::f2a1233 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a123444 * sind +
               a_const::f2a123444 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a12344 + a_const::f1a12344 * sind +
               a_const::f2a12344 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 *
              (a_const::f0a1234 + a_const::f1a1234 * sind +
               a_const::f2a1234 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a123555 * sind +
               a_const::f2a123555 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a12355 + a_const::f1a12355 * sind +
               a_const::f2a12355 * std::pow(sind, 2)) +
          xi2 * xi3 * xi5 *
              (a_const::f0a1235 + a_const::f1a1235 * sind +
               a_const::f2a1235 * std::pow(sind, 2)) +
          xi2 * xi3 *
              (a_const::f0a123 + a_const::f1a123 * sind +
               a_const::f2a123 * std::pow(sind, 2) +
               a_const::f3a123 * std::pow(sind, 3)) +
          xi2 * std::pow(xi4, 4) *
              (a_const::f1a124444 * sind +
               a_const::f2a124444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a124445 * sind +
               a_const::f2a124445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) *
              (a_const::f0a12444 + a_const::f1a12444 * sind +
               a_const::f2a12444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a124455 * sind +
               a_const::f2a124455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a12445 + a_const::f1a12445 * sind +
               a_const::f2a12445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) *
              (a_const::f0a1244 + a_const::f1a1244 * sind +
               a_const::f2a1244 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a124555 * sind +
               a_const::f2a124555 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a12455 + a_const::f1a12455 * sind +
               a_const::f2a12455 * std::pow(sind, 2)) +
          xi2 * xi4 * xi5 *
              (a_const::f0a1245 + a_const::f1a1245 * sind +
               a_const::f2a1245 * std::pow(sind, 2)) +
          xi2 * xi4 *
              (a_const::f0a124 + a_const::f1a124 * sind +
               a_const::f2a124 * std::pow(sind, 2) +
               a_const::f3a124 * std::pow(sind, 3)) +
          xi2 * std::pow(xi5, 4) *
              (a_const::f1a125555 * sind +
               a_const::f2a125555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 3) *
              (a_const::f0a12555 + a_const::f1a12555 * sind +
               a_const::f2a12555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 2) *
              (a_const::f0a1255 + a_const::f1a1255 * sind +
               a_const::f2a1255 * std::pow(sind, 2)) +
          xi2 * xi5 *
              (a_const::f0a125 + a_const::f1a125 * sind +
               a_const::f2a125 * std::pow(sind, 2) +
               a_const::f3a125 * std::pow(sind, 3)) +
          xi2 * (a_const::f0a12 + a_const::f1a12 * sind +
                 a_const::f2a12 * std::pow(sind, 2) +
                 a_const::f3a12 * std::pow(sind, 3) +
                 a_const::f4a12 * std::pow(sind, 4)) +
          std::pow(xi3, 5) * (a_const::f1a133333 * sind +
                              a_const::f2a133333 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi4 *
              (a_const::f1a133334 * sind +
               a_const::f2a133334 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi5 *
              (a_const::f1a133335 * sind +
               a_const::f2a133335 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * (a_const::f0a13333 + a_const::f1a13333 * sind +
                              a_const::f2a13333 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a133344 * sind +
               a_const::f2a133344 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a133345 * sind +
               a_const::f2a133345 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 *
              (a_const::f0a13334 + a_const::f1a13334 * sind +
               a_const::f2a13334 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a133355 * sind +
               a_const::f2a133355 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi5 *
              (a_const::f0a13335 + a_const::f1a13335 * sind +
               a_const::f2a13335 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * (a_const::f0a1333 + a_const::f1a1333 * sind +
                              a_const::f2a1333 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a133444 * sind +
               a_const::f2a133444 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a133445 * sind +
               a_const::f2a133445 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a13344 + a_const::f1a13344 * sind +
               a_const::f2a13344 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a133455 * sind +
               a_const::f2a133455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a13345 + a_const::f1a13345 * sind +
               a_const::f2a13345 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 *
              (a_const::f0a1334 + a_const::f1a1334 * sind +
               a_const::f2a1334 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a133555 * sind +
               a_const::f2a133555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a13355 + a_const::f1a13355 * sind +
               a_const::f2a13355 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi5 *
              (a_const::f0a1335 + a_const::f1a1335 * sind +
               a_const::f2a1335 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * (a_const::f0a133 + a_const::f1a133 * sind +
                              a_const::f2a133 * std::pow(sind, 2) +
                              a_const::f3a133 * std::pow(sind, 3)) +
          xi3 * std::pow(xi4, 4) *
              (a_const::f1a134444 * sind +
               a_const::f2a134444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a134445 * sind +
               a_const::f2a134445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) *
              (a_const::f0a13444 + a_const::f1a13444 * sind +
               a_const::f2a13444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a134455 * sind +
               a_const::f2a134455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a13445 + a_const::f1a13445 * sind +
               a_const::f2a13445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) *
              (a_const::f0a1344 + a_const::f1a1344 * sind +
               a_const::f2a1344 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a134555 * sind +
               a_const::f2a134555 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a13455 + a_const::f1a13455 * sind +
               a_const::f2a13455 * std::pow(sind, 2)) +
          xi3 * xi4 * xi5 *
              (a_const::f0a1345 + a_const::f1a1345 * sind +
               a_const::f2a1345 * std::pow(sind, 2)) +
          xi3 * xi4 *
              (a_const::f0a134 + a_const::f1a134 * sind +
               a_const::f2a134 * std::pow(sind, 2) +
               a_const::f3a134 * std::pow(sind, 3)) +
          xi3 * std::pow(xi5, 4) *
              (a_const::f1a135555 * sind +
               a_const::f2a135555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 3) *
              (a_const::f0a13555 + a_const::f1a13555 * sind +
               a_const::f2a13555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 2) *
              (a_const::f0a1355 + a_const::f1a1355 * sind +
               a_const::f2a1355 * std::pow(sind, 2)) +
          xi3 * xi5 *
              (a_const::f0a135 + a_const::f1a135 * sind +
               a_const::f2a135 * std::pow(sind, 2) +
               a_const::f3a135 * std::pow(sind, 3)) +
          xi3 * (a_const::f0a13 + a_const::f1a13 * sind +
                 a_const::f2a13 * std::pow(sind, 2) +
                 a_const::f3a13 * std::pow(sind, 3) +
                 a_const::f4a13 * std::pow(sind, 4)) +
          std::pow(xi4, 5) * (a_const::f1a144444 * sind +
                              a_const::f2a144444 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * xi5 *
              (a_const::f1a144445 * sind +
               a_const::f2a144445 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * (a_const::f0a14444 + a_const::f1a14444 * sind +
                              a_const::f2a14444 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a144455 * sind +
               a_const::f2a144455 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * xi5 *
              (a_const::f0a14445 + a_const::f1a14445 * sind +
               a_const::f2a14445 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * (a_const::f0a1444 + a_const::f1a1444 * sind +
                              a_const::f2a1444 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a144555 * sind +
               a_const::f2a144555 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a14455 + a_const::f1a14455 * sind +
               a_const::f2a14455 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * xi5 *
              (a_const::f0a1445 + a_const::f1a1445 * sind +
               a_const::f2a1445 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * (a_const::f0a144 + a_const::f1a144 * sind +
                              a_const::f2a144 * std::pow(sind, 2) +
                              a_const::f3a144 * std::pow(sind, 3)) +
          xi4 * std::pow(xi5, 4) *
              (a_const::f1a145555 * sind +
               a_const::f2a145555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 3) *
              (a_const::f0a14555 + a_const::f1a14555 * sind +
               a_const::f2a14555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 2) *
              (a_const::f0a1455 + a_const::f1a1455 * sind +
               a_const::f2a1455 * std::pow(sind, 2)) +
          xi4 * xi5 *
              (a_const::f0a145 + a_const::f1a145 * sind +
               a_const::f2a145 * std::pow(sind, 2) +
               a_const::f3a145 * std::pow(sind, 3)) +
          xi4 * (a_const::f0a14 + a_const::f1a14 * sind +
                 a_const::f2a14 * std::pow(sind, 2) +
                 a_const::f3a14 * std::pow(sind, 3) +
                 a_const::f4a14 * std::pow(sind, 4)) +
          std::pow(xi5, 5) * (a_const::f1a155555 * sind +
                              a_const::f2a155555 * std::pow(sind, 2)) +
          std::pow(xi5, 4) * (a_const::f0a15555 + a_const::f1a15555 * sind +
                              a_const::f2a15555 * std::pow(sind, 2)) +
          std::pow(xi5, 3) * (a_const::f0a1555 + a_const::f1a1555 * sind +
                              a_const::f2a1555 * std::pow(sind, 2)) +
          std::pow(xi5, 2) * (a_const::f0a155 + a_const::f1a155 * sind +
                              a_const::f2a155 * std::pow(sind, 2) +
                              a_const::f3a155 * std::pow(sind, 3)) +
          xi5 * (a_const::f0a15 + a_const::f1a15 * sind +
                 a_const::f2a15 * std::pow(sind, 2) +
                 a_const::f3a15 * std::pow(sind, 3) +
                 a_const::f4a15 * std::pow(sind, 4));
      dvdxi2 =
          a_const::f0a2 + a_const::f1a2 * sind +
          a_const::f2a2 * std::pow(sind, 2) +
          a_const::f3a2 * std::pow(sind, 3) +
          a_const::f4a2 * std::pow(sind, 4) +
          a_const::f5a2 * std::pow(sind, 5) +
          a_const::f6a2 * std::pow(sind, 6) +
          std::pow(xi1, 5) * (a_const::f1a111112 * sind +
                              a_const::f2a111112 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 4) * xi2 *
              (a_const::f1a111122 * sind +
               a_const::f2a111122 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 *
              (a_const::f1a111123 * sind +
               a_const::f2a111123 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi4 *
              (a_const::f1a111124 * sind +
               a_const::f2a111124 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi5 *
              (a_const::f1a111125 * sind +
               a_const::f2a111125 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * (a_const::f0a11112 + a_const::f1a11112 * sind +
                              a_const::f2a11112 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a111222 * sind +
               a_const::f2a111222 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a111223 * sind +
               a_const::f2a111223 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a111224 * sind +
               a_const::f2a111224 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a111225 * sind +
               a_const::f2a111225 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 *
              (a_const::f0a11122 + a_const::f1a11122 * sind +
               a_const::f2a11122 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a111233 * sind +
               a_const::f2a111233 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a111234 * sind +
               a_const::f2a111234 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a111235 * sind +
               a_const::f2a111235 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 *
              (a_const::f0a11123 + a_const::f1a11123 * sind +
               a_const::f2a11123 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a111244 * sind +
               a_const::f2a111244 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a111245 * sind +
               a_const::f2a111245 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 *
              (a_const::f0a11124 + a_const::f1a11124 * sind +
               a_const::f2a11124 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a111255 * sind +
               a_const::f2a111255 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi5 *
              (a_const::f0a11125 + a_const::f1a11125 * sind +
               a_const::f2a11125 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * (a_const::f0a1112 + a_const::f1a1112 * sind +
                              a_const::f2a1112 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a112222 * sind +
               a_const::f2a112222 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a112223 * sind +
               a_const::f2a112223 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a112224 * sind +
               a_const::f2a112224 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a112225 * sind +
               a_const::f2a112225 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a11222 + a_const::f1a11222 * sind +
               a_const::f2a11222 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a112233 * sind +
               a_const::f2a112233 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a112234 * sind +
               a_const::f2a112234 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a112235 * sind +
               a_const::f2a112235 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a11223 + a_const::f1a11223 * sind +
               a_const::f2a11223 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a112244 * sind +
               a_const::f2a112244 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a112245 * sind +
               a_const::f2a112245 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a11224 + a_const::f1a11224 * sind +
               a_const::f2a11224 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a112255 * sind +
               a_const::f2a112255 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a11225 + a_const::f1a11225 * sind +
               a_const::f2a11225 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 *
              (a_const::f0a1122 + a_const::f1a1122 * sind +
               a_const::f2a1122 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a112333 * sind +
               a_const::f2a112333 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a112334 * sind +
               a_const::f2a112334 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a112335 * sind +
               a_const::f2a112335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a11233 + a_const::f1a11233 * sind +
               a_const::f2a11233 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a112344 * sind +
               a_const::f2a112344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a11234 + a_const::f1a11234 * sind +
               a_const::f2a11234 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a112355 * sind +
               a_const::f2a112355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a11235 + a_const::f1a11235 * sind +
               a_const::f2a11235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 *
              (a_const::f0a1123 + a_const::f1a1123 * sind +
               a_const::f2a1123 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a112444 * sind +
               a_const::f2a112444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a112445 * sind +
               a_const::f2a112445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a11244 + a_const::f1a11244 * sind +
               a_const::f2a11244 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a112455 * sind +
               a_const::f2a112455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a11245 + a_const::f1a11245 * sind +
               a_const::f2a11245 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 *
              (a_const::f0a1124 + a_const::f1a1124 * sind +
               a_const::f2a1124 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a112555 * sind +
               a_const::f2a112555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a11255 + a_const::f1a11255 * sind +
               a_const::f2a11255 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi5 *
              (a_const::f0a1125 + a_const::f1a1125 * sind +
               a_const::f2a1125 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * (a_const::f0a112 + a_const::f1a112 * sind +
                              a_const::f2a112 * std::pow(sind, 2) +
                              a_const::f3a112 * std::pow(sind, 3)) +
          5 * xi1 * std::pow(xi2, 4) *
              (a_const::f1a122222 * sind +
               a_const::f2a122222 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a122223 * sind +
               a_const::f2a122223 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a122224 * sind +
               a_const::f2a122224 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a122225 * sind +
               a_const::f2a122225 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi2, 3) *
              (a_const::f0a12222 + a_const::f1a12222 * sind +
               a_const::f2a12222 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a122233 * sind +
               a_const::f2a122233 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a122234 * sind +
               a_const::f2a122234 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a122235 * sind +
               a_const::f2a122235 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a12223 + a_const::f1a12223 * sind +
               a_const::f2a12223 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a122244 * sind +
               a_const::f2a122244 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a122245 * sind +
               a_const::f2a122245 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a12224 + a_const::f1a12224 * sind +
               a_const::f2a12224 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a122255 * sind +
               a_const::f2a122255 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a12225 + a_const::f1a12225 * sind +
               a_const::f2a12225 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) *
              (a_const::f0a1222 + a_const::f1a1222 * sind +
               a_const::f2a1222 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a122333 * sind +
               a_const::f2a122333 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a122334 * sind +
               a_const::f2a122334 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a122335 * sind +
               a_const::f2a122335 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a12233 + a_const::f1a12233 * sind +
               a_const::f2a12233 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a122344 * sind +
               a_const::f2a122344 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a12234 + a_const::f1a12234 * sind +
               a_const::f2a12234 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a122355 * sind +
               a_const::f2a122355 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a12235 + a_const::f1a12235 * sind +
               a_const::f2a12235 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 *
              (a_const::f0a1223 + a_const::f1a1223 * sind +
               a_const::f2a1223 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a122444 * sind +
               a_const::f2a122444 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a122445 * sind +
               a_const::f2a122445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a12244 + a_const::f1a12244 * sind +
               a_const::f2a12244 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a122455 * sind +
               a_const::f2a122455 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a12245 + a_const::f1a12245 * sind +
               a_const::f2a12245 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 *
              (a_const::f0a1224 + a_const::f1a1224 * sind +
               a_const::f2a1224 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a122555 * sind +
               a_const::f2a122555 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a12255 + a_const::f1a12255 * sind +
               a_const::f2a12255 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi5 *
              (a_const::f0a1225 + a_const::f1a1225 * sind +
               a_const::f2a1225 * std::pow(sind, 2)) +
          2 * xi1 * xi2 *
              (a_const::f0a122 + a_const::f1a122 * sind +
               a_const::f2a122 * std::pow(sind, 2) +
               a_const::f3a122 * std::pow(sind, 3)) +
          xi1 * std::pow(xi3, 4) *
              (a_const::f1a123333 * sind +
               a_const::f2a123333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a123334 * sind +
               a_const::f2a123334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a123335 * sind +
               a_const::f2a123335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) *
              (a_const::f0a12333 + a_const::f1a12333 * sind +
               a_const::f2a12333 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a123344 * sind +
               a_const::f2a123344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a12334 + a_const::f1a12334 * sind +
               a_const::f2a12334 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a123355 * sind +
               a_const::f2a123355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a12335 + a_const::f1a12335 * sind +
               a_const::f2a12335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) *
              (a_const::f0a1233 + a_const::f1a1233 * sind +
               a_const::f2a1233 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a123444 * sind +
               a_const::f2a123444 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a12344 + a_const::f1a12344 * sind +
               a_const::f2a12344 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 *
              (a_const::f0a1234 + a_const::f1a1234 * sind +
               a_const::f2a1234 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a123555 * sind +
               a_const::f2a123555 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a12355 + a_const::f1a12355 * sind +
               a_const::f2a12355 * std::pow(sind, 2)) +
          xi1 * xi3 * xi5 *
              (a_const::f0a1235 + a_const::f1a1235 * sind +
               a_const::f2a1235 * std::pow(sind, 2)) +
          xi1 * xi3 *
              (a_const::f0a123 + a_const::f1a123 * sind +
               a_const::f2a123 * std::pow(sind, 2) +
               a_const::f3a123 * std::pow(sind, 3)) +
          xi1 * std::pow(xi4, 4) *
              (a_const::f1a124444 * sind +
               a_const::f2a124444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a124445 * sind +
               a_const::f2a124445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) *
              (a_const::f0a12444 + a_const::f1a12444 * sind +
               a_const::f2a12444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a124455 * sind +
               a_const::f2a124455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a12445 + a_const::f1a12445 * sind +
               a_const::f2a12445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) *
              (a_const::f0a1244 + a_const::f1a1244 * sind +
               a_const::f2a1244 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a124555 * sind +
               a_const::f2a124555 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a12455 + a_const::f1a12455 * sind +
               a_const::f2a12455 * std::pow(sind, 2)) +
          xi1 * xi4 * xi5 *
              (a_const::f0a1245 + a_const::f1a1245 * sind +
               a_const::f2a1245 * std::pow(sind, 2)) +
          xi1 * xi4 *
              (a_const::f0a124 + a_const::f1a124 * sind +
               a_const::f2a124 * std::pow(sind, 2) +
               a_const::f3a124 * std::pow(sind, 3)) +
          xi1 * std::pow(xi5, 4) *
              (a_const::f1a125555 * sind +
               a_const::f2a125555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 3) *
              (a_const::f0a12555 + a_const::f1a12555 * sind +
               a_const::f2a12555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 2) *
              (a_const::f0a1255 + a_const::f1a1255 * sind +
               a_const::f2a1255 * std::pow(sind, 2)) +
          xi1 * xi5 *
              (a_const::f0a125 + a_const::f1a125 * sind +
               a_const::f2a125 * std::pow(sind, 2) +
               a_const::f3a125 * std::pow(sind, 3)) +
          xi1 * (a_const::f0a12 + a_const::f1a12 * sind +
                 a_const::f2a12 * std::pow(sind, 2) +
                 a_const::f3a12 * std::pow(sind, 3) +
                 a_const::f4a12 * std::pow(sind, 4)) +
          6 * std::pow(xi2, 5) *
              (a_const::f1a222222 * sind +
               a_const::f2a222222 * std::pow(sind, 2)) +
          5 * std::pow(xi2, 4) * xi3 *
              (a_const::f1a222223 * sind +
               a_const::f2a222223 * std::pow(sind, 2)) +
          5 * std::pow(xi2, 4) * xi4 *
              (a_const::f1a222224 * sind +
               a_const::f2a222224 * std::pow(sind, 2)) +
          5 * std::pow(xi2, 4) * xi5 *
              (a_const::f1a222225 * sind +
               a_const::f2a222225 * std::pow(sind, 2)) +
          5 * std::pow(xi2, 4) *
              (a_const::f0a22222 + a_const::f1a22222 * sind +
               a_const::f2a22222 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a222233 * sind +
               a_const::f2a222233 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a222234 * sind +
               a_const::f2a222234 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a222235 * sind +
               a_const::f2a222235 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi3 *
              (a_const::f0a22223 + a_const::f1a22223 * sind +
               a_const::f2a22223 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a222244 * sind +
               a_const::f2a222244 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a222245 * sind +
               a_const::f2a222245 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi4 *
              (a_const::f0a22224 + a_const::f1a22224 * sind +
               a_const::f2a22224 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a222255 * sind +
               a_const::f2a222255 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) * xi5 *
              (a_const::f0a22225 + a_const::f1a22225 * sind +
               a_const::f2a22225 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 3) *
              (a_const::f0a2222 + a_const::f1a2222 * sind +
               a_const::f2a2222 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a222333 * sind +
               a_const::f2a222333 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a222334 * sind +
               a_const::f2a222334 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a222335 * sind +
               a_const::f2a222335 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a22233 + a_const::f1a22233 * sind +
               a_const::f2a22233 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a222344 * sind +
               a_const::f2a222344 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a222345 * sind +
               a_const::f2a222345 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a22234 + a_const::f1a22234 * sind +
               a_const::f2a22234 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a222355 * sind +
               a_const::f2a222355 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a22235 + a_const::f1a22235 * sind +
               a_const::f2a22235 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a2223 + a_const::f1a2223 * sind +
               a_const::f2a2223 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a222444 * sind +
               a_const::f2a222444 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a222445 * sind +
               a_const::f2a222445 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a22244 + a_const::f1a22244 * sind +
               a_const::f2a22244 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a222455 * sind +
               a_const::f2a222455 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a22245 + a_const::f1a22245 * sind +
               a_const::f2a22245 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a2224 + a_const::f1a2224 * sind +
               a_const::f2a2224 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a222555 * sind +
               a_const::f2a222555 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a22255 + a_const::f1a22255 * sind +
               a_const::f2a22255 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a2225 + a_const::f1a2225 * sind +
               a_const::f2a2225 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) *
              (a_const::f0a222 + a_const::f1a222 * sind +
               a_const::f2a222 * std::pow(sind, 2) +
               a_const::f3a222 * std::pow(sind, 3)) +
          2 * xi2 * std::pow(xi3, 4) *
              (a_const::f1a223333 * sind +
               a_const::f2a223333 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a223334 * sind +
               a_const::f2a223334 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a223335 * sind +
               a_const::f2a223335 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 3) *
              (a_const::f0a22333 + a_const::f1a22333 * sind +
               a_const::f2a22333 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a223344 * sind +
               a_const::f2a223344 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a223345 * sind +
               a_const::f2a223345 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a22334 + a_const::f1a22334 * sind +
               a_const::f2a22334 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a223355 * sind +
               a_const::f2a223355 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a22335 + a_const::f1a22335 * sind +
               a_const::f2a22335 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a2233 + a_const::f1a2233 * sind +
               a_const::f2a2233 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a223444 * sind +
               a_const::f2a223444 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a223445 * sind +
               a_const::f2a223445 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a22344 + a_const::f1a22344 * sind +
               a_const::f2a22344 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a223455 * sind +
               a_const::f2a223455 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a22345 + a_const::f1a22345 * sind +
               a_const::f2a22345 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 *
              (a_const::f0a2234 + a_const::f1a2234 * sind +
               a_const::f2a2234 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a223555 * sind +
               a_const::f2a223555 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a22355 + a_const::f1a22355 * sind +
               a_const::f2a22355 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi5 *
              (a_const::f0a2235 + a_const::f1a2235 * sind +
               a_const::f2a2235 * std::pow(sind, 2)) +
          2 * xi2 * xi3 *
              (a_const::f0a223 + a_const::f1a223 * sind +
               a_const::f2a223 * std::pow(sind, 2) +
               a_const::f3a223 * std::pow(sind, 3)) +
          2 * xi2 * std::pow(xi4, 4) *
              (a_const::f1a224444 * sind +
               a_const::f2a224444 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a224445 * sind +
               a_const::f2a224445 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 3) *
              (a_const::f0a22444 + a_const::f1a22444 * sind +
               a_const::f2a22444 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a224455 * sind +
               a_const::f2a224455 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a22445 + a_const::f1a22445 * sind +
               a_const::f2a22445 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a2244 + a_const::f1a2244 * sind +
               a_const::f2a2244 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a224555 * sind +
               a_const::f2a224555 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a22455 + a_const::f1a22455 * sind +
               a_const::f2a22455 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * xi5 *
              (a_const::f0a2245 + a_const::f1a2245 * sind +
               a_const::f2a2245 * std::pow(sind, 2)) +
          2 * xi2 * xi4 *
              (a_const::f0a224 + a_const::f1a224 * sind +
               a_const::f2a224 * std::pow(sind, 2) +
               a_const::f3a224 * std::pow(sind, 3)) +
          2 * xi2 * std::pow(xi5, 4) *
              (a_const::f1a225555 * sind +
               a_const::f2a225555 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi5, 3) *
              (a_const::f0a22555 + a_const::f1a22555 * sind +
               a_const::f2a22555 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a2255 + a_const::f1a2255 * sind +
               a_const::f2a2255 * std::pow(sind, 2)) +
          2 * xi2 * xi5 *
              (a_const::f0a225 + a_const::f1a225 * sind +
               a_const::f2a225 * std::pow(sind, 2) +
               a_const::f3a225 * std::pow(sind, 3)) +
          2 * xi2 *
              (a_const::f0a22 + a_const::f1a22 * sind +
               a_const::f2a22 * std::pow(sind, 2) +
               a_const::f3a22 * std::pow(sind, 3) +
               a_const::f4a22 * std::pow(sind, 4)) +
          std::pow(xi3, 5) * (a_const::f1a233333 * sind +
                              a_const::f2a233333 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi4 *
              (a_const::f1a233334 * sind +
               a_const::f2a233334 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi5 *
              (a_const::f1a233335 * sind +
               a_const::f2a233335 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * (a_const::f0a23333 + a_const::f1a23333 * sind +
                              a_const::f2a23333 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a233344 * sind +
               a_const::f2a233344 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a233345 * sind +
               a_const::f2a233345 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 *
              (a_const::f0a23334 + a_const::f1a23334 * sind +
               a_const::f2a23334 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a233355 * sind +
               a_const::f2a233355 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi5 *
              (a_const::f0a23335 + a_const::f1a23335 * sind +
               a_const::f2a23335 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * (a_const::f0a2333 + a_const::f1a2333 * sind +
                              a_const::f2a2333 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a233444 * sind +
               a_const::f2a233444 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a233445 * sind +
               a_const::f2a233445 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a23344 + a_const::f1a23344 * sind +
               a_const::f2a23344 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a233455 * sind +
               a_const::f2a233455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a23345 + a_const::f1a23345 * sind +
               a_const::f2a23345 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 *
              (a_const::f0a2334 + a_const::f1a2334 * sind +
               a_const::f2a2334 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a233555 * sind +
               a_const::f2a233555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a23355 + a_const::f1a23355 * sind +
               a_const::f2a23355 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi5 *
              (a_const::f0a2335 + a_const::f1a2335 * sind +
               a_const::f2a2335 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * (a_const::f0a233 + a_const::f1a233 * sind +
                              a_const::f2a233 * std::pow(sind, 2) +
                              a_const::f3a233 * std::pow(sind, 3)) +
          xi3 * std::pow(xi4, 4) *
              (a_const::f1a234444 * sind +
               a_const::f2a234444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a234445 * sind +
               a_const::f2a234445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) *
              (a_const::f0a23444 + a_const::f1a23444 * sind +
               a_const::f2a23444 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a234455 * sind +
               a_const::f2a234455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a23445 + a_const::f1a23445 * sind +
               a_const::f2a23445 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) *
              (a_const::f0a2344 + a_const::f1a2344 * sind +
               a_const::f2a2344 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a234555 * sind +
               a_const::f2a234555 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a23455 + a_const::f1a23455 * sind +
               a_const::f2a23455 * std::pow(sind, 2)) +
          xi3 * xi4 * xi5 *
              (a_const::f0a2345 + a_const::f1a2345 * sind +
               a_const::f2a2345 * std::pow(sind, 2)) +
          xi3 * xi4 *
              (a_const::f0a234 + a_const::f1a234 * sind +
               a_const::f2a234 * std::pow(sind, 2) +
               a_const::f3a234 * std::pow(sind, 3)) +
          xi3 * std::pow(xi5, 4) *
              (a_const::f1a235555 * sind +
               a_const::f2a235555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 3) *
              (a_const::f0a23555 + a_const::f1a23555 * sind +
               a_const::f2a23555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 2) *
              (a_const::f0a2355 + a_const::f1a2355 * sind +
               a_const::f2a2355 * std::pow(sind, 2)) +
          xi3 * xi5 *
              (a_const::f0a235 + a_const::f1a235 * sind +
               a_const::f2a235 * std::pow(sind, 2) +
               a_const::f3a235 * std::pow(sind, 3)) +
          xi3 * (a_const::f0a23 + a_const::f1a23 * sind +
                 a_const::f2a23 * std::pow(sind, 2) +
                 a_const::f3a23 * std::pow(sind, 3) +
                 a_const::f4a23 * std::pow(sind, 4)) +
          std::pow(xi4, 5) * (a_const::f1a244444 * sind +
                              a_const::f2a244444 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * xi5 *
              (a_const::f1a244445 * sind +
               a_const::f2a244445 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * (a_const::f0a24444 + a_const::f1a24444 * sind +
                              a_const::f2a24444 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a244455 * sind +
               a_const::f2a244455 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * xi5 *
              (a_const::f0a24445 + a_const::f1a24445 * sind +
               a_const::f2a24445 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * (a_const::f0a2444 + a_const::f1a2444 * sind +
                              a_const::f2a2444 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a244555 * sind +
               a_const::f2a244555 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a24455 + a_const::f1a24455 * sind +
               a_const::f2a24455 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * xi5 *
              (a_const::f0a2445 + a_const::f1a2445 * sind +
               a_const::f2a2445 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * (a_const::f0a244 + a_const::f1a244 * sind +
                              a_const::f2a244 * std::pow(sind, 2) +
                              a_const::f3a244 * std::pow(sind, 3)) +
          xi4 * std::pow(xi5, 4) *
              (a_const::f1a245555 * sind +
               a_const::f2a245555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 3) *
              (a_const::f0a24555 + a_const::f1a24555 * sind +
               a_const::f2a24555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 2) *
              (a_const::f0a2455 + a_const::f1a2455 * sind +
               a_const::f2a2455 * std::pow(sind, 2)) +
          xi4 * xi5 *
              (a_const::f0a245 + a_const::f1a245 * sind +
               a_const::f2a245 * std::pow(sind, 2) +
               a_const::f3a245 * std::pow(sind, 3)) +
          xi4 * (a_const::f0a24 + a_const::f1a24 * sind +
                 a_const::f2a24 * std::pow(sind, 2) +
                 a_const::f3a24 * std::pow(sind, 3) +
                 a_const::f4a24 * std::pow(sind, 4)) +
          std::pow(xi5, 5) * (a_const::f1a255555 * sind +
                              a_const::f2a255555 * std::pow(sind, 2)) +
          std::pow(xi5, 4) * (a_const::f0a25555 + a_const::f1a25555 * sind +
                              a_const::f2a25555 * std::pow(sind, 2)) +
          std::pow(xi5, 3) * (a_const::f0a2555 + a_const::f1a2555 * sind +
                              a_const::f2a2555 * std::pow(sind, 2)) +
          std::pow(xi5, 2) * (a_const::f0a255 + a_const::f1a255 * sind +
                              a_const::f2a255 * std::pow(sind, 2) +
                              a_const::f3a255 * std::pow(sind, 3)) +
          xi5 * (a_const::f0a25 + a_const::f1a25 * sind +
                 a_const::f2a25 * std::pow(sind, 2) +
                 a_const::f3a25 * std::pow(sind, 3) +
                 a_const::f4a25 * std::pow(sind, 4));
      dvdxi3 =
          a_const::f0a3 + a_const::f1a3 * sind +
          a_const::f2a3 * std::pow(sind, 2) +
          a_const::f3a3 * std::pow(sind, 3) +
          a_const::f4a3 * std::pow(sind, 4) +
          a_const::f5a3 * std::pow(sind, 5) +
          a_const::f6a3 * std::pow(sind, 6) +
          std::pow(xi1, 5) * (a_const::f1a111113 * sind +
                              a_const::f2a111113 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 *
              (a_const::f1a111123 * sind +
               a_const::f2a111123 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 4) * xi3 *
              (a_const::f1a111133 * sind +
               a_const::f2a111133 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi4 *
              (a_const::f1a111134 * sind +
               a_const::f2a111134 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi5 *
              (a_const::f1a111135 * sind +
               a_const::f2a111135 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * (a_const::f0a11113 + a_const::f1a11113 * sind +
                              a_const::f2a11113 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a111223 * sind +
               a_const::f2a111223 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a111233 * sind +
               a_const::f2a111233 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a111234 * sind +
               a_const::f2a111234 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a111235 * sind +
               a_const::f2a111235 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 *
              (a_const::f0a11123 + a_const::f1a11123 * sind +
               a_const::f2a11123 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a111333 * sind +
               a_const::f2a111333 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a111334 * sind +
               a_const::f2a111334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a111335 * sind +
               a_const::f2a111335 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi3 *
              (a_const::f0a11133 + a_const::f1a11133 * sind +
               a_const::f2a11133 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a111344 * sind +
               a_const::f2a111344 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a111345 * sind +
               a_const::f2a111345 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 *
              (a_const::f0a11134 + a_const::f1a11134 * sind +
               a_const::f2a11134 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a111355 * sind +
               a_const::f2a111355 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi5 *
              (a_const::f0a11135 + a_const::f1a11135 * sind +
               a_const::f2a11135 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * (a_const::f0a1113 + a_const::f1a1113 * sind +
                              a_const::f2a1113 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a112223 * sind +
               a_const::f2a112223 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a112233 * sind +
               a_const::f2a112233 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a112234 * sind +
               a_const::f2a112234 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a112235 * sind +
               a_const::f2a112235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a11223 + a_const::f1a11223 * sind +
               a_const::f2a11223 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a112333 * sind +
               a_const::f2a112333 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a112334 * sind +
               a_const::f2a112334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a112335 * sind +
               a_const::f2a112335 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a11233 + a_const::f1a11233 * sind +
               a_const::f2a11233 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a112344 * sind +
               a_const::f2a112344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a11234 + a_const::f1a11234 * sind +
               a_const::f2a11234 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a112355 * sind +
               a_const::f2a112355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a11235 + a_const::f1a11235 * sind +
               a_const::f2a11235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 *
              (a_const::f0a1123 + a_const::f1a1123 * sind +
               a_const::f2a1123 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a113333 * sind +
               a_const::f2a113333 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a113334 * sind +
               a_const::f2a113334 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a113335 * sind +
               a_const::f2a113335 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a11333 + a_const::f1a11333 * sind +
               a_const::f2a11333 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a113344 * sind +
               a_const::f2a113344 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a113345 * sind +
               a_const::f2a113345 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a11334 + a_const::f1a11334 * sind +
               a_const::f2a11334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a113355 * sind +
               a_const::f2a113355 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a11335 + a_const::f1a11335 * sind +
               a_const::f2a11335 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 *
              (a_const::f0a1133 + a_const::f1a1133 * sind +
               a_const::f2a1133 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a113444 * sind +
               a_const::f2a113444 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a113445 * sind +
               a_const::f2a113445 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a11344 + a_const::f1a11344 * sind +
               a_const::f2a11344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a113455 * sind +
               a_const::f2a113455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a11345 + a_const::f1a11345 * sind +
               a_const::f2a11345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 *
              (a_const::f0a1134 + a_const::f1a1134 * sind +
               a_const::f2a1134 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a113555 * sind +
               a_const::f2a113555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a11355 + a_const::f1a11355 * sind +
               a_const::f2a11355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi5 *
              (a_const::f0a1135 + a_const::f1a1135 * sind +
               a_const::f2a1135 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * (a_const::f0a113 + a_const::f1a113 * sind +
                              a_const::f2a113 * std::pow(sind, 2) +
                              a_const::f3a113 * std::pow(sind, 3)) +
          xi1 * std::pow(xi2, 4) *
              (a_const::f1a122223 * sind +
               a_const::f2a122223 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a122233 * sind +
               a_const::f2a122233 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a122234 * sind +
               a_const::f2a122234 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a122235 * sind +
               a_const::f2a122235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) *
              (a_const::f0a12223 + a_const::f1a12223 * sind +
               a_const::f2a12223 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a122333 * sind +
               a_const::f2a122333 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a122334 * sind +
               a_const::f2a122334 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a122335 * sind +
               a_const::f2a122335 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a12233 + a_const::f1a12233 * sind +
               a_const::f2a12233 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a122344 * sind +
               a_const::f2a122344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a12234 + a_const::f1a12234 * sind +
               a_const::f2a12234 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a122355 * sind +
               a_const::f2a122355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a12235 + a_const::f1a12235 * sind +
               a_const::f2a12235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) *
              (a_const::f0a1223 + a_const::f1a1223 * sind +
               a_const::f2a1223 * std::pow(sind, 2)) +
          4 * xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a123333 * sind +
               a_const::f2a123333 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a123334 * sind +
               a_const::f2a123334 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a123335 * sind +
               a_const::f2a123335 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a12333 + a_const::f1a12333 * sind +
               a_const::f2a12333 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a123344 * sind +
               a_const::f2a123344 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a12334 + a_const::f1a12334 * sind +
               a_const::f2a12334 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a123355 * sind +
               a_const::f2a123355 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a12335 + a_const::f1a12335 * sind +
               a_const::f2a12335 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 *
              (a_const::f0a1233 + a_const::f1a1233 * sind +
               a_const::f2a1233 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a123444 * sind +
               a_const::f2a123444 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a12344 + a_const::f1a12344 * sind +
               a_const::f2a12344 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 *
              (a_const::f0a1234 + a_const::f1a1234 * sind +
               a_const::f2a1234 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a123555 * sind +
               a_const::f2a123555 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a12355 + a_const::f1a12355 * sind +
               a_const::f2a12355 * std::pow(sind, 2)) +
          xi1 * xi2 * xi5 *
              (a_const::f0a1235 + a_const::f1a1235 * sind +
               a_const::f2a1235 * std::pow(sind, 2)) +
          xi1 * xi2 *
              (a_const::f0a123 + a_const::f1a123 * sind +
               a_const::f2a123 * std::pow(sind, 2) +
               a_const::f3a123 * std::pow(sind, 3)) +
          5 * xi1 * std::pow(xi3, 4) *
              (a_const::f1a133333 * sind +
               a_const::f2a133333 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a133334 * sind +
               a_const::f2a133334 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a133335 * sind +
               a_const::f2a133335 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi3, 3) *
              (a_const::f0a13333 + a_const::f1a13333 * sind +
               a_const::f2a13333 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a133344 * sind +
               a_const::f2a133344 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a133345 * sind +
               a_const::f2a133345 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a13334 + a_const::f1a13334 * sind +
               a_const::f2a13334 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a133355 * sind +
               a_const::f2a133355 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a13335 + a_const::f1a13335 * sind +
               a_const::f2a13335 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) *
              (a_const::f0a1333 + a_const::f1a1333 * sind +
               a_const::f2a1333 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a133444 * sind +
               a_const::f2a133444 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a133445 * sind +
               a_const::f2a133445 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a13344 + a_const::f1a13344 * sind +
               a_const::f2a13344 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a133455 * sind +
               a_const::f2a133455 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a13345 + a_const::f1a13345 * sind +
               a_const::f2a13345 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 *
              (a_const::f0a1334 + a_const::f1a1334 * sind +
               a_const::f2a1334 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a133555 * sind +
               a_const::f2a133555 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a13355 + a_const::f1a13355 * sind +
               a_const::f2a13355 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi5 *
              (a_const::f0a1335 + a_const::f1a1335 * sind +
               a_const::f2a1335 * std::pow(sind, 2)) +
          2 * xi1 * xi3 *
              (a_const::f0a133 + a_const::f1a133 * sind +
               a_const::f2a133 * std::pow(sind, 2) +
               a_const::f3a133 * std::pow(sind, 3)) +
          xi1 * std::pow(xi4, 4) *
              (a_const::f1a134444 * sind +
               a_const::f2a134444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a134445 * sind +
               a_const::f2a134445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) *
              (a_const::f0a13444 + a_const::f1a13444 * sind +
               a_const::f2a13444 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a134455 * sind +
               a_const::f2a134455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a13445 + a_const::f1a13445 * sind +
               a_const::f2a13445 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) *
              (a_const::f0a1344 + a_const::f1a1344 * sind +
               a_const::f2a1344 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a134555 * sind +
               a_const::f2a134555 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a13455 + a_const::f1a13455 * sind +
               a_const::f2a13455 * std::pow(sind, 2)) +
          xi1 * xi4 * xi5 *
              (a_const::f0a1345 + a_const::f1a1345 * sind +
               a_const::f2a1345 * std::pow(sind, 2)) +
          xi1 * xi4 *
              (a_const::f0a134 + a_const::f1a134 * sind +
               a_const::f2a134 * std::pow(sind, 2) +
               a_const::f3a134 * std::pow(sind, 3)) +
          xi1 * std::pow(xi5, 4) *
              (a_const::f1a135555 * sind +
               a_const::f2a135555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 3) *
              (a_const::f0a13555 + a_const::f1a13555 * sind +
               a_const::f2a13555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 2) *
              (a_const::f0a1355 + a_const::f1a1355 * sind +
               a_const::f2a1355 * std::pow(sind, 2)) +
          xi1 * xi5 *
              (a_const::f0a135 + a_const::f1a135 * sind +
               a_const::f2a135 * std::pow(sind, 2) +
               a_const::f3a135 * std::pow(sind, 3)) +
          xi1 * (a_const::f0a13 + a_const::f1a13 * sind +
                 a_const::f2a13 * std::pow(sind, 2) +
                 a_const::f3a13 * std::pow(sind, 3) +
                 a_const::f4a13 * std::pow(sind, 4)) +
          std::pow(xi2, 5) * (a_const::f1a222223 * sind +
                              a_const::f2a222223 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 4) * xi3 *
              (a_const::f1a222233 * sind +
               a_const::f2a222233 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi4 *
              (a_const::f1a222234 * sind +
               a_const::f2a222234 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi5 *
              (a_const::f1a222235 * sind +
               a_const::f2a222235 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * (a_const::f0a22223 + a_const::f1a22223 * sind +
                              a_const::f2a22223 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a222333 * sind +
               a_const::f2a222333 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a222334 * sind +
               a_const::f2a222334 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a222335 * sind +
               a_const::f2a222335 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi3 *
              (a_const::f0a22233 + a_const::f1a22233 * sind +
               a_const::f2a22233 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a222344 * sind +
               a_const::f2a222344 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a222345 * sind +
               a_const::f2a222345 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 *
              (a_const::f0a22234 + a_const::f1a22234 * sind +
               a_const::f2a22234 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a222355 * sind +
               a_const::f2a222355 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi5 *
              (a_const::f0a22235 + a_const::f1a22235 * sind +
               a_const::f2a22235 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * (a_const::f0a2223 + a_const::f1a2223 * sind +
                              a_const::f2a2223 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a223333 * sind +
               a_const::f2a223333 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a223334 * sind +
               a_const::f2a223334 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a223335 * sind +
               a_const::f2a223335 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a22333 + a_const::f1a22333 * sind +
               a_const::f2a22333 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a223344 * sind +
               a_const::f2a223344 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a223345 * sind +
               a_const::f2a223345 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a22334 + a_const::f1a22334 * sind +
               a_const::f2a22334 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a223355 * sind +
               a_const::f2a223355 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a22335 + a_const::f1a22335 * sind +
               a_const::f2a22335 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a2233 + a_const::f1a2233 * sind +
               a_const::f2a2233 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a223444 * sind +
               a_const::f2a223444 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a223445 * sind +
               a_const::f2a223445 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a22344 + a_const::f1a22344 * sind +
               a_const::f2a22344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a223455 * sind +
               a_const::f2a223455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a22345 + a_const::f1a22345 * sind +
               a_const::f2a22345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 *
              (a_const::f0a2234 + a_const::f1a2234 * sind +
               a_const::f2a2234 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a223555 * sind +
               a_const::f2a223555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a22355 + a_const::f1a22355 * sind +
               a_const::f2a22355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi5 *
              (a_const::f0a2235 + a_const::f1a2235 * sind +
               a_const::f2a2235 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * (a_const::f0a223 + a_const::f1a223 * sind +
                              a_const::f2a223 * std::pow(sind, 2) +
                              a_const::f3a223 * std::pow(sind, 3)) +
          5 * xi2 * std::pow(xi3, 4) *
              (a_const::f1a233333 * sind +
               a_const::f2a233333 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a233334 * sind +
               a_const::f2a233334 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a233335 * sind +
               a_const::f2a233335 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi3, 3) *
              (a_const::f0a23333 + a_const::f1a23333 * sind +
               a_const::f2a23333 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a233344 * sind +
               a_const::f2a233344 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a233345 * sind +
               a_const::f2a233345 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a23334 + a_const::f1a23334 * sind +
               a_const::f2a23334 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a233355 * sind +
               a_const::f2a233355 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a23335 + a_const::f1a23335 * sind +
               a_const::f2a23335 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a2333 + a_const::f1a2333 * sind +
               a_const::f2a2333 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a233444 * sind +
               a_const::f2a233444 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a233445 * sind +
               a_const::f2a233445 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a23344 + a_const::f1a23344 * sind +
               a_const::f2a23344 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a233455 * sind +
               a_const::f2a233455 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a23345 + a_const::f1a23345 * sind +
               a_const::f2a23345 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 *
              (a_const::f0a2334 + a_const::f1a2334 * sind +
               a_const::f2a2334 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a233555 * sind +
               a_const::f2a233555 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a23355 + a_const::f1a23355 * sind +
               a_const::f2a23355 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi5 *
              (a_const::f0a2335 + a_const::f1a2335 * sind +
               a_const::f2a2335 * std::pow(sind, 2)) +
          2 * xi2 * xi3 *
              (a_const::f0a233 + a_const::f1a233 * sind +
               a_const::f2a233 * std::pow(sind, 2) +
               a_const::f3a233 * std::pow(sind, 3)) +
          xi2 * std::pow(xi4, 4) *
              (a_const::f1a234444 * sind +
               a_const::f2a234444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a234445 * sind +
               a_const::f2a234445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) *
              (a_const::f0a23444 + a_const::f1a23444 * sind +
               a_const::f2a23444 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a234455 * sind +
               a_const::f2a234455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a23445 + a_const::f1a23445 * sind +
               a_const::f2a23445 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) *
              (a_const::f0a2344 + a_const::f1a2344 * sind +
               a_const::f2a2344 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a234555 * sind +
               a_const::f2a234555 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a23455 + a_const::f1a23455 * sind +
               a_const::f2a23455 * std::pow(sind, 2)) +
          xi2 * xi4 * xi5 *
              (a_const::f0a2345 + a_const::f1a2345 * sind +
               a_const::f2a2345 * std::pow(sind, 2)) +
          xi2 * xi4 *
              (a_const::f0a234 + a_const::f1a234 * sind +
               a_const::f2a234 * std::pow(sind, 2) +
               a_const::f3a234 * std::pow(sind, 3)) +
          xi2 * std::pow(xi5, 4) *
              (a_const::f1a235555 * sind +
               a_const::f2a235555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 3) *
              (a_const::f0a23555 + a_const::f1a23555 * sind +
               a_const::f2a23555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 2) *
              (a_const::f0a2355 + a_const::f1a2355 * sind +
               a_const::f2a2355 * std::pow(sind, 2)) +
          xi2 * xi5 *
              (a_const::f0a235 + a_const::f1a235 * sind +
               a_const::f2a235 * std::pow(sind, 2) +
               a_const::f3a235 * std::pow(sind, 3)) +
          xi2 * (a_const::f0a23 + a_const::f1a23 * sind +
                 a_const::f2a23 * std::pow(sind, 2) +
                 a_const::f3a23 * std::pow(sind, 3) +
                 a_const::f4a23 * std::pow(sind, 4)) +
          6 * std::pow(xi3, 5) *
              (a_const::f1a333333 * sind +
               a_const::f2a333333 * std::pow(sind, 2)) +
          5 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a333334 * sind +
               a_const::f2a333334 * std::pow(sind, 2)) +
          5 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a333335 * sind +
               a_const::f2a333335 * std::pow(sind, 2)) +
          5 * std::pow(xi3, 4) *
              (a_const::f0a33333 + a_const::f1a33333 * sind +
               a_const::f2a33333 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a333344 * sind +
               a_const::f2a333344 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a333345 * sind +
               a_const::f2a333345 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) * xi4 *
              (a_const::f0a33334 + a_const::f1a33334 * sind +
               a_const::f2a33334 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a333355 * sind +
               a_const::f2a333355 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) * xi5 *
              (a_const::f0a33335 + a_const::f1a33335 * sind +
               a_const::f2a33335 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 3) *
              (a_const::f0a3333 + a_const::f1a3333 * sind +
               a_const::f2a3333 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a333444 * sind +
               a_const::f2a333444 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a333445 * sind +
               a_const::f2a333445 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a33344 + a_const::f1a33344 * sind +
               a_const::f2a33344 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a333455 * sind +
               a_const::f2a333455 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a33345 + a_const::f1a33345 * sind +
               a_const::f2a33345 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a3334 + a_const::f1a3334 * sind +
               a_const::f2a3334 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a333555 * sind +
               a_const::f2a333555 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a33355 + a_const::f1a33355 * sind +
               a_const::f2a33355 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a3335 + a_const::f1a3335 * sind +
               a_const::f2a3335 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) *
              (a_const::f0a333 + a_const::f1a333 * sind +
               a_const::f2a333 * std::pow(sind, 2) +
               a_const::f3a333 * std::pow(sind, 3)) +
          2 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a334444 * sind +
               a_const::f2a334444 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a334445 * sind +
               a_const::f2a334445 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 3) *
              (a_const::f0a33444 + a_const::f1a33444 * sind +
               a_const::f2a33444 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a334455 * sind +
               a_const::f2a334455 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a33445 + a_const::f1a33445 * sind +
               a_const::f2a33445 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a3344 + a_const::f1a3344 * sind +
               a_const::f2a3344 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a334555 * sind +
               a_const::f2a334555 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a33455 + a_const::f1a33455 * sind +
               a_const::f2a33455 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * xi5 *
              (a_const::f0a3345 + a_const::f1a3345 * sind +
               a_const::f2a3345 * std::pow(sind, 2)) +
          2 * xi3 * xi4 *
              (a_const::f0a334 + a_const::f1a334 * sind +
               a_const::f2a334 * std::pow(sind, 2) +
               a_const::f3a334 * std::pow(sind, 3)) +
          2 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a335555 * sind +
               a_const::f2a335555 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi5, 3) *
              (a_const::f0a33555 + a_const::f1a33555 * sind +
               a_const::f2a33555 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a3355 + a_const::f1a3355 * sind +
               a_const::f2a3355 * std::pow(sind, 2)) +
          2 * xi3 * xi5 *
              (a_const::f0a335 + a_const::f1a335 * sind +
               a_const::f2a335 * std::pow(sind, 2) +
               a_const::f3a335 * std::pow(sind, 3)) +
          2 * xi3 *
              (a_const::f0a33 + a_const::f1a33 * sind +
               a_const::f2a33 * std::pow(sind, 2) +
               a_const::f3a33 * std::pow(sind, 3) +
               a_const::f4a33 * std::pow(sind, 4)) +
          std::pow(xi4, 5) * (a_const::f1a344444 * sind +
                              a_const::f2a344444 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * xi5 *
              (a_const::f1a344445 * sind +
               a_const::f2a344445 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * (a_const::f0a34444 + a_const::f1a34444 * sind +
                              a_const::f2a34444 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a344455 * sind +
               a_const::f2a344455 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * xi5 *
              (a_const::f0a34445 + a_const::f1a34445 * sind +
               a_const::f2a34445 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * (a_const::f0a3444 + a_const::f1a3444 * sind +
                              a_const::f2a3444 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a344555 * sind +
               a_const::f2a344555 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a34455 + a_const::f1a34455 * sind +
               a_const::f2a34455 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * xi5 *
              (a_const::f0a3445 + a_const::f1a3445 * sind +
               a_const::f2a3445 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * (a_const::f0a344 + a_const::f1a344 * sind +
                              a_const::f2a344 * std::pow(sind, 2) +
                              a_const::f3a344 * std::pow(sind, 3)) +
          xi4 * std::pow(xi5, 4) *
              (a_const::f1a345555 * sind +
               a_const::f2a345555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 3) *
              (a_const::f0a34555 + a_const::f1a34555 * sind +
               a_const::f2a34555 * std::pow(sind, 2)) +
          xi4 * std::pow(xi5, 2) *
              (a_const::f0a3455 + a_const::f1a3455 * sind +
               a_const::f2a3455 * std::pow(sind, 2)) +
          xi4 * xi5 *
              (a_const::f0a345 + a_const::f1a345 * sind +
               a_const::f2a345 * std::pow(sind, 2) +
               a_const::f3a345 * std::pow(sind, 3)) +
          xi4 * (a_const::f0a34 + a_const::f1a34 * sind +
                 a_const::f2a34 * std::pow(sind, 2) +
                 a_const::f3a34 * std::pow(sind, 3) +
                 a_const::f4a34 * std::pow(sind, 4)) +
          std::pow(xi5, 5) * (a_const::f1a355555 * sind +
                              a_const::f2a355555 * std::pow(sind, 2)) +
          std::pow(xi5, 4) * (a_const::f0a35555 + a_const::f1a35555 * sind +
                              a_const::f2a35555 * std::pow(sind, 2)) +
          std::pow(xi5, 3) * (a_const::f0a3555 + a_const::f1a3555 * sind +
                              a_const::f2a3555 * std::pow(sind, 2)) +
          std::pow(xi5, 2) * (a_const::f0a355 + a_const::f1a355 * sind +
                              a_const::f2a355 * std::pow(sind, 2) +
                              a_const::f3a355 * std::pow(sind, 3)) +
          xi5 * (a_const::f0a35 + a_const::f1a35 * sind +
                 a_const::f2a35 * std::pow(sind, 2) +
                 a_const::f3a35 * std::pow(sind, 3) +
                 a_const::f4a35 * std::pow(sind, 4));
      dvdxi4 =
          a_const::f0a4 + a_const::f1a4 * sind +
          a_const::f2a4 * std::pow(sind, 2) +
          a_const::f3a4 * std::pow(sind, 3) +
          a_const::f4a4 * std::pow(sind, 4) +
          a_const::f5a4 * std::pow(sind, 5) +
          a_const::f6a4 * std::pow(sind, 6) +
          std::pow(xi1, 5) * (a_const::f1a111114 * sind +
                              a_const::f2a111114 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 *
              (a_const::f1a111124 * sind +
               a_const::f2a111124 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 *
              (a_const::f1a111134 * sind +
               a_const::f2a111134 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 4) * xi4 *
              (a_const::f1a111144 * sind +
               a_const::f2a111144 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi5 *
              (a_const::f1a111145 * sind +
               a_const::f2a111145 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * (a_const::f0a11114 + a_const::f1a11114 * sind +
                              a_const::f2a11114 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a111224 * sind +
               a_const::f2a111224 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a111234 * sind +
               a_const::f2a111234 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a111244 * sind +
               a_const::f2a111244 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a111245 * sind +
               a_const::f2a111245 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 *
              (a_const::f0a11124 + a_const::f1a11124 * sind +
               a_const::f2a11124 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a111334 * sind +
               a_const::f2a111334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a111344 * sind +
               a_const::f2a111344 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a111345 * sind +
               a_const::f2a111345 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 *
              (a_const::f0a11134 + a_const::f1a11134 * sind +
               a_const::f2a11134 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a111444 * sind +
               a_const::f2a111444 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a111445 * sind +
               a_const::f2a111445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi4 *
              (a_const::f0a11144 + a_const::f1a11144 * sind +
               a_const::f2a11144 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a111455 * sind +
               a_const::f2a111455 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi5 *
              (a_const::f0a11145 + a_const::f1a11145 * sind +
               a_const::f2a11145 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * (a_const::f0a1114 + a_const::f1a1114 * sind +
                              a_const::f2a1114 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a112224 * sind +
               a_const::f2a112224 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a112234 * sind +
               a_const::f2a112234 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a112244 * sind +
               a_const::f2a112244 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a112245 * sind +
               a_const::f2a112245 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a11224 + a_const::f1a11224 * sind +
               a_const::f2a11224 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a112334 * sind +
               a_const::f2a112334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a112344 * sind +
               a_const::f2a112344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a11234 + a_const::f1a11234 * sind +
               a_const::f2a11234 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a112444 * sind +
               a_const::f2a112444 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a112445 * sind +
               a_const::f2a112445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a11244 + a_const::f1a11244 * sind +
               a_const::f2a11244 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a112455 * sind +
               a_const::f2a112455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a11245 + a_const::f1a11245 * sind +
               a_const::f2a11245 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 *
              (a_const::f0a1124 + a_const::f1a1124 * sind +
               a_const::f2a1124 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a113334 * sind +
               a_const::f2a113334 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a113344 * sind +
               a_const::f2a113344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a113345 * sind +
               a_const::f2a113345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a11334 + a_const::f1a11334 * sind +
               a_const::f2a11334 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a113444 * sind +
               a_const::f2a113444 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a113445 * sind +
               a_const::f2a113445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a11344 + a_const::f1a11344 * sind +
               a_const::f2a11344 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a113455 * sind +
               a_const::f2a113455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a11345 + a_const::f1a11345 * sind +
               a_const::f2a11345 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 *
              (a_const::f0a1134 + a_const::f1a1134 * sind +
               a_const::f2a1134 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a114444 * sind +
               a_const::f2a114444 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a114445 * sind +
               a_const::f2a114445 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a11444 + a_const::f1a11444 * sind +
               a_const::f2a11444 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a114455 * sind +
               a_const::f2a114455 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a11445 + a_const::f1a11445 * sind +
               a_const::f2a11445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi4 *
              (a_const::f0a1144 + a_const::f1a1144 * sind +
               a_const::f2a1144 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a114555 * sind +
               a_const::f2a114555 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a11455 + a_const::f1a11455 * sind +
               a_const::f2a11455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi5 *
              (a_const::f0a1145 + a_const::f1a1145 * sind +
               a_const::f2a1145 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * (a_const::f0a114 + a_const::f1a114 * sind +
                              a_const::f2a114 * std::pow(sind, 2) +
                              a_const::f3a114 * std::pow(sind, 3)) +
          xi1 * std::pow(xi2, 4) *
              (a_const::f1a122224 * sind +
               a_const::f2a122224 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a122234 * sind +
               a_const::f2a122234 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a122244 * sind +
               a_const::f2a122244 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a122245 * sind +
               a_const::f2a122245 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) *
              (a_const::f0a12224 + a_const::f1a12224 * sind +
               a_const::f2a12224 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a122334 * sind +
               a_const::f2a122334 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a122344 * sind +
               a_const::f2a122344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a12234 + a_const::f1a12234 * sind +
               a_const::f2a12234 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a122444 * sind +
               a_const::f2a122444 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a122445 * sind +
               a_const::f2a122445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a12244 + a_const::f1a12244 * sind +
               a_const::f2a12244 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a122455 * sind +
               a_const::f2a122455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a12245 + a_const::f1a12245 * sind +
               a_const::f2a12245 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) *
              (a_const::f0a1224 + a_const::f1a1224 * sind +
               a_const::f2a1224 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a123334 * sind +
               a_const::f2a123334 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a123344 * sind +
               a_const::f2a123344 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a12334 + a_const::f1a12334 * sind +
               a_const::f2a12334 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a123444 * sind +
               a_const::f2a123444 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a12344 + a_const::f1a12344 * sind +
               a_const::f2a12344 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 *
              (a_const::f0a1234 + a_const::f1a1234 * sind +
               a_const::f2a1234 * std::pow(sind, 2)) +
          4 * xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a124444 * sind +
               a_const::f2a124444 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a124445 * sind +
               a_const::f2a124445 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a12444 + a_const::f1a12444 * sind +
               a_const::f2a12444 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a124455 * sind +
               a_const::f2a124455 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a12445 + a_const::f1a12445 * sind +
               a_const::f2a12445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 *
              (a_const::f0a1244 + a_const::f1a1244 * sind +
               a_const::f2a1244 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a124555 * sind +
               a_const::f2a124555 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a12455 + a_const::f1a12455 * sind +
               a_const::f2a12455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi5 *
              (a_const::f0a1245 + a_const::f1a1245 * sind +
               a_const::f2a1245 * std::pow(sind, 2)) +
          xi1 * xi2 *
              (a_const::f0a124 + a_const::f1a124 * sind +
               a_const::f2a124 * std::pow(sind, 2) +
               a_const::f3a124 * std::pow(sind, 3)) +
          xi1 * std::pow(xi3, 4) *
              (a_const::f1a133334 * sind +
               a_const::f2a133334 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a133344 * sind +
               a_const::f2a133344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a133345 * sind +
               a_const::f2a133345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) *
              (a_const::f0a13334 + a_const::f1a13334 * sind +
               a_const::f2a13334 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a133444 * sind +
               a_const::f2a133444 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a133445 * sind +
               a_const::f2a133445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a13344 + a_const::f1a13344 * sind +
               a_const::f2a13344 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a133455 * sind +
               a_const::f2a133455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a13345 + a_const::f1a13345 * sind +
               a_const::f2a13345 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) *
              (a_const::f0a1334 + a_const::f1a1334 * sind +
               a_const::f2a1334 * std::pow(sind, 2)) +
          4 * xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a134444 * sind +
               a_const::f2a134444 * std::pow(sind, 2)) +
          3 * xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a134445 * sind +
               a_const::f2a134445 * std::pow(sind, 2)) +
          3 * xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a13444 + a_const::f1a13444 * sind +
               a_const::f2a13444 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a134455 * sind +
               a_const::f2a134455 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a13445 + a_const::f1a13445 * sind +
               a_const::f2a13445 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 *
              (a_const::f0a1344 + a_const::f1a1344 * sind +
               a_const::f2a1344 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a134555 * sind +
               a_const::f2a134555 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a13455 + a_const::f1a13455 * sind +
               a_const::f2a13455 * std::pow(sind, 2)) +
          xi1 * xi3 * xi5 *
              (a_const::f0a1345 + a_const::f1a1345 * sind +
               a_const::f2a1345 * std::pow(sind, 2)) +
          xi1 * xi3 *
              (a_const::f0a134 + a_const::f1a134 * sind +
               a_const::f2a134 * std::pow(sind, 2) +
               a_const::f3a134 * std::pow(sind, 3)) +
          5 * xi1 * std::pow(xi4, 4) *
              (a_const::f1a144444 * sind +
               a_const::f2a144444 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a144445 * sind +
               a_const::f2a144445 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi4, 3) *
              (a_const::f0a14444 + a_const::f1a14444 * sind +
               a_const::f2a14444 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a144455 * sind +
               a_const::f2a144455 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a14445 + a_const::f1a14445 * sind +
               a_const::f2a14445 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi4, 2) *
              (a_const::f0a1444 + a_const::f1a1444 * sind +
               a_const::f2a1444 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a144555 * sind +
               a_const::f2a144555 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a14455 + a_const::f1a14455 * sind +
               a_const::f2a14455 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * xi5 *
              (a_const::f0a1445 + a_const::f1a1445 * sind +
               a_const::f2a1445 * std::pow(sind, 2)) +
          2 * xi1 * xi4 *
              (a_const::f0a144 + a_const::f1a144 * sind +
               a_const::f2a144 * std::pow(sind, 2) +
               a_const::f3a144 * std::pow(sind, 3)) +
          xi1 * std::pow(xi5, 4) *
              (a_const::f1a145555 * sind +
               a_const::f2a145555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 3) *
              (a_const::f0a14555 + a_const::f1a14555 * sind +
               a_const::f2a14555 * std::pow(sind, 2)) +
          xi1 * std::pow(xi5, 2) *
              (a_const::f0a1455 + a_const::f1a1455 * sind +
               a_const::f2a1455 * std::pow(sind, 2)) +
          xi1 * xi5 *
              (a_const::f0a145 + a_const::f1a145 * sind +
               a_const::f2a145 * std::pow(sind, 2) +
               a_const::f3a145 * std::pow(sind, 3)) +
          xi1 * (a_const::f0a14 + a_const::f1a14 * sind +
                 a_const::f2a14 * std::pow(sind, 2) +
                 a_const::f3a14 * std::pow(sind, 3) +
                 a_const::f4a14 * std::pow(sind, 4)) +
          std::pow(xi2, 5) * (a_const::f1a222224 * sind +
                              a_const::f2a222224 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 *
              (a_const::f1a222234 * sind +
               a_const::f2a222234 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 4) * xi4 *
              (a_const::f1a222244 * sind +
               a_const::f2a222244 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi5 *
              (a_const::f1a222245 * sind +
               a_const::f2a222245 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * (a_const::f0a22224 + a_const::f1a22224 * sind +
                              a_const::f2a22224 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a222334 * sind +
               a_const::f2a222334 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a222344 * sind +
               a_const::f2a222344 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a222345 * sind +
               a_const::f2a222345 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 *
              (a_const::f0a22234 + a_const::f1a22234 * sind +
               a_const::f2a22234 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a222444 * sind +
               a_const::f2a222444 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a222445 * sind +
               a_const::f2a222445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi4 *
              (a_const::f0a22244 + a_const::f1a22244 * sind +
               a_const::f2a22244 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a222455 * sind +
               a_const::f2a222455 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi5 *
              (a_const::f0a22245 + a_const::f1a22245 * sind +
               a_const::f2a22245 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * (a_const::f0a2224 + a_const::f1a2224 * sind +
                              a_const::f2a2224 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a223334 * sind +
               a_const::f2a223334 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a223344 * sind +
               a_const::f2a223344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a223345 * sind +
               a_const::f2a223345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a22334 + a_const::f1a22334 * sind +
               a_const::f2a22334 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a223444 * sind +
               a_const::f2a223444 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a223445 * sind +
               a_const::f2a223445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a22344 + a_const::f1a22344 * sind +
               a_const::f2a22344 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a223455 * sind +
               a_const::f2a223455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a22345 + a_const::f1a22345 * sind +
               a_const::f2a22345 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 *
              (a_const::f0a2234 + a_const::f1a2234 * sind +
               a_const::f2a2234 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a224444 * sind +
               a_const::f2a224444 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a224445 * sind +
               a_const::f2a224445 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a22444 + a_const::f1a22444 * sind +
               a_const::f2a22444 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a224455 * sind +
               a_const::f2a224455 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a22445 + a_const::f1a22445 * sind +
               a_const::f2a22445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a2244 + a_const::f1a2244 * sind +
               a_const::f2a2244 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a224555 * sind +
               a_const::f2a224555 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a22455 + a_const::f1a22455 * sind +
               a_const::f2a22455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi5 *
              (a_const::f0a2245 + a_const::f1a2245 * sind +
               a_const::f2a2245 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * (a_const::f0a224 + a_const::f1a224 * sind +
                              a_const::f2a224 * std::pow(sind, 2) +
                              a_const::f3a224 * std::pow(sind, 3)) +
          xi2 * std::pow(xi3, 4) *
              (a_const::f1a233334 * sind +
               a_const::f2a233334 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a233344 * sind +
               a_const::f2a233344 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a233345 * sind +
               a_const::f2a233345 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) *
              (a_const::f0a23334 + a_const::f1a23334 * sind +
               a_const::f2a23334 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a233444 * sind +
               a_const::f2a233444 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a233445 * sind +
               a_const::f2a233445 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a23344 + a_const::f1a23344 * sind +
               a_const::f2a23344 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a233455 * sind +
               a_const::f2a233455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a23345 + a_const::f1a23345 * sind +
               a_const::f2a23345 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) *
              (a_const::f0a2334 + a_const::f1a2334 * sind +
               a_const::f2a2334 * std::pow(sind, 2)) +
          4 * xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a234444 * sind +
               a_const::f2a234444 * std::pow(sind, 2)) +
          3 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a234445 * sind +
               a_const::f2a234445 * std::pow(sind, 2)) +
          3 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a23444 + a_const::f1a23444 * sind +
               a_const::f2a23444 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a234455 * sind +
               a_const::f2a234455 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a23445 + a_const::f1a23445 * sind +
               a_const::f2a23445 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 *
              (a_const::f0a2344 + a_const::f1a2344 * sind +
               a_const::f2a2344 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a234555 * sind +
               a_const::f2a234555 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a23455 + a_const::f1a23455 * sind +
               a_const::f2a23455 * std::pow(sind, 2)) +
          xi2 * xi3 * xi5 *
              (a_const::f0a2345 + a_const::f1a2345 * sind +
               a_const::f2a2345 * std::pow(sind, 2)) +
          xi2 * xi3 *
              (a_const::f0a234 + a_const::f1a234 * sind +
               a_const::f2a234 * std::pow(sind, 2) +
               a_const::f3a234 * std::pow(sind, 3)) +
          5 * xi2 * std::pow(xi4, 4) *
              (a_const::f1a244444 * sind +
               a_const::f2a244444 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a244445 * sind +
               a_const::f2a244445 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi4, 3) *
              (a_const::f0a24444 + a_const::f1a24444 * sind +
               a_const::f2a24444 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a244455 * sind +
               a_const::f2a244455 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a24445 + a_const::f1a24445 * sind +
               a_const::f2a24445 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a2444 + a_const::f1a2444 * sind +
               a_const::f2a2444 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a244555 * sind +
               a_const::f2a244555 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a24455 + a_const::f1a24455 * sind +
               a_const::f2a24455 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * xi5 *
              (a_const::f0a2445 + a_const::f1a2445 * sind +
               a_const::f2a2445 * std::pow(sind, 2)) +
          2 * xi2 * xi4 *
              (a_const::f0a244 + a_const::f1a244 * sind +
               a_const::f2a244 * std::pow(sind, 2) +
               a_const::f3a244 * std::pow(sind, 3)) +
          xi2 * std::pow(xi5, 4) *
              (a_const::f1a245555 * sind +
               a_const::f2a245555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 3) *
              (a_const::f0a24555 + a_const::f1a24555 * sind +
               a_const::f2a24555 * std::pow(sind, 2)) +
          xi2 * std::pow(xi5, 2) *
              (a_const::f0a2455 + a_const::f1a2455 * sind +
               a_const::f2a2455 * std::pow(sind, 2)) +
          xi2 * xi5 *
              (a_const::f0a245 + a_const::f1a245 * sind +
               a_const::f2a245 * std::pow(sind, 2) +
               a_const::f3a245 * std::pow(sind, 3)) +
          xi2 * (a_const::f0a24 + a_const::f1a24 * sind +
                 a_const::f2a24 * std::pow(sind, 2) +
                 a_const::f3a24 * std::pow(sind, 3) +
                 a_const::f4a24 * std::pow(sind, 4)) +
          std::pow(xi3, 5) * (a_const::f1a333334 * sind +
                              a_const::f2a333334 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a333344 * sind +
               a_const::f2a333344 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi5 *
              (a_const::f1a333345 * sind +
               a_const::f2a333345 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * (a_const::f0a33334 + a_const::f1a33334 * sind +
                              a_const::f2a33334 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a333444 * sind +
               a_const::f2a333444 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a333445 * sind +
               a_const::f2a333445 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 3) * xi4 *
              (a_const::f0a33344 + a_const::f1a33344 * sind +
               a_const::f2a33344 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a333455 * sind +
               a_const::f2a333455 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi5 *
              (a_const::f0a33345 + a_const::f1a33345 * sind +
               a_const::f2a33345 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * (a_const::f0a3334 + a_const::f1a3334 * sind +
                              a_const::f2a3334 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a334444 * sind +
               a_const::f2a334444 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a334445 * sind +
               a_const::f2a334445 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a33444 + a_const::f1a33444 * sind +
               a_const::f2a33444 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a334455 * sind +
               a_const::f2a334455 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a33445 + a_const::f1a33445 * sind +
               a_const::f2a33445 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a3344 + a_const::f1a3344 * sind +
               a_const::f2a3344 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a334555 * sind +
               a_const::f2a334555 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a33455 + a_const::f1a33455 * sind +
               a_const::f2a33455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi5 *
              (a_const::f0a3345 + a_const::f1a3345 * sind +
               a_const::f2a3345 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * (a_const::f0a334 + a_const::f1a334 * sind +
                              a_const::f2a334 * std::pow(sind, 2) +
                              a_const::f3a334 * std::pow(sind, 3)) +
          5 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a344444 * sind +
               a_const::f2a344444 * std::pow(sind, 2)) +
          4 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a344445 * sind +
               a_const::f2a344445 * std::pow(sind, 2)) +
          4 * xi3 * std::pow(xi4, 3) *
              (a_const::f0a34444 + a_const::f1a34444 * sind +
               a_const::f2a34444 * std::pow(sind, 2)) +
          3 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a344455 * sind +
               a_const::f2a344455 * std::pow(sind, 2)) +
          3 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a34445 + a_const::f1a34445 * sind +
               a_const::f2a34445 * std::pow(sind, 2)) +
          3 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a3444 + a_const::f1a3444 * sind +
               a_const::f2a3444 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a344555 * sind +
               a_const::f2a344555 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a34455 + a_const::f1a34455 * sind +
               a_const::f2a34455 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * xi5 *
              (a_const::f0a3445 + a_const::f1a3445 * sind +
               a_const::f2a3445 * std::pow(sind, 2)) +
          2 * xi3 * xi4 *
              (a_const::f0a344 + a_const::f1a344 * sind +
               a_const::f2a344 * std::pow(sind, 2) +
               a_const::f3a344 * std::pow(sind, 3)) +
          xi3 * std::pow(xi5, 4) *
              (a_const::f1a345555 * sind +
               a_const::f2a345555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 3) *
              (a_const::f0a34555 + a_const::f1a34555 * sind +
               a_const::f2a34555 * std::pow(sind, 2)) +
          xi3 * std::pow(xi5, 2) *
              (a_const::f0a3455 + a_const::f1a3455 * sind +
               a_const::f2a3455 * std::pow(sind, 2)) +
          xi3 * xi5 *
              (a_const::f0a345 + a_const::f1a345 * sind +
               a_const::f2a345 * std::pow(sind, 2) +
               a_const::f3a345 * std::pow(sind, 3)) +
          xi3 * (a_const::f0a34 + a_const::f1a34 * sind +
                 a_const::f2a34 * std::pow(sind, 2) +
                 a_const::f3a34 * std::pow(sind, 3) +
                 a_const::f4a34 * std::pow(sind, 4)) +
          6 * std::pow(xi4, 5) *
              (a_const::f1a444444 * sind +
               a_const::f2a444444 * std::pow(sind, 2)) +
          5 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a444445 * sind +
               a_const::f2a444445 * std::pow(sind, 2)) +
          5 * std::pow(xi4, 4) *
              (a_const::f0a44444 + a_const::f1a44444 * sind +
               a_const::f2a44444 * std::pow(sind, 2)) +
          4 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a444455 * sind +
               a_const::f2a444455 * std::pow(sind, 2)) +
          4 * std::pow(xi4, 3) * xi5 *
              (a_const::f0a44445 + a_const::f1a44445 * sind +
               a_const::f2a44445 * std::pow(sind, 2)) +
          4 * std::pow(xi4, 3) *
              (a_const::f0a4444 + a_const::f1a4444 * sind +
               a_const::f2a4444 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a444555 * sind +
               a_const::f2a444555 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a44455 + a_const::f1a44455 * sind +
               a_const::f2a44455 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a4445 + a_const::f1a4445 * sind +
               a_const::f2a4445 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 2) *
              (a_const::f0a444 + a_const::f1a444 * sind +
               a_const::f2a444 * std::pow(sind, 2) +
               a_const::f3a444 * std::pow(sind, 3)) +
          2 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a445555 * sind +
               a_const::f2a445555 * std::pow(sind, 2)) +
          2 * xi4 * std::pow(xi5, 3) *
              (a_const::f0a44555 + a_const::f1a44555 * sind +
               a_const::f2a44555 * std::pow(sind, 2)) +
          2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a4455 + a_const::f1a4455 * sind +
               a_const::f2a4455 * std::pow(sind, 2)) +
          2 * xi4 * xi5 *
              (a_const::f0a445 + a_const::f1a445 * sind +
               a_const::f2a445 * std::pow(sind, 2) +
               a_const::f3a445 * std::pow(sind, 3)) +
          2 * xi4 *
              (a_const::f0a44 + a_const::f1a44 * sind +
               a_const::f2a44 * std::pow(sind, 2) +
               a_const::f3a44 * std::pow(sind, 3) +
               a_const::f4a44 * std::pow(sind, 4)) +
          std::pow(xi5, 5) * (a_const::f1a455555 * sind +
                              a_const::f2a455555 * std::pow(sind, 2)) +
          std::pow(xi5, 4) * (a_const::f0a45555 + a_const::f1a45555 * sind +
                              a_const::f2a45555 * std::pow(sind, 2)) +
          std::pow(xi5, 3) * (a_const::f0a4555 + a_const::f1a4555 * sind +
                              a_const::f2a4555 * std::pow(sind, 2)) +
          std::pow(xi5, 2) * (a_const::f0a455 + a_const::f1a455 * sind +
                              a_const::f2a455 * std::pow(sind, 2) +
                              a_const::f3a455 * std::pow(sind, 3)) +
          xi5 * (a_const::f0a45 + a_const::f1a45 * sind +
                 a_const::f2a45 * std::pow(sind, 2) +
                 a_const::f3a45 * std::pow(sind, 3) +
                 a_const::f4a45 * std::pow(sind, 4));
      dvdxi5 =
          a_const::f0a5 + a_const::f1a5 * sind +
          a_const::f2a5 * std::pow(sind, 2) +
          a_const::f3a5 * std::pow(sind, 3) +
          a_const::f4a5 * std::pow(sind, 4) +
          a_const::f5a5 * std::pow(sind, 5) +
          a_const::f6a5 * std::pow(sind, 6) +
          std::pow(xi1, 5) * (a_const::f1a111115 * sind +
                              a_const::f2a111115 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi2 *
              (a_const::f1a111125 * sind +
               a_const::f2a111125 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi3 *
              (a_const::f1a111135 * sind +
               a_const::f2a111135 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * xi4 *
              (a_const::f1a111145 * sind +
               a_const::f2a111145 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 4) * xi5 *
              (a_const::f1a111155 * sind +
               a_const::f2a111155 * std::pow(sind, 2)) +
          std::pow(xi1, 4) * (a_const::f0a11115 + a_const::f1a11115 * sind +
                              a_const::f2a11115 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a111225 * sind +
               a_const::f2a111225 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a111235 * sind +
               a_const::f2a111235 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a111245 * sind +
               a_const::f2a111245 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a111255 * sind +
               a_const::f2a111255 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi2 *
              (a_const::f0a11125 + a_const::f1a11125 * sind +
               a_const::f2a11125 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a111335 * sind +
               a_const::f2a111335 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a111345 * sind +
               a_const::f2a111345 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a111355 * sind +
               a_const::f2a111355 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi3 *
              (a_const::f0a11135 + a_const::f1a11135 * sind +
               a_const::f2a11135 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a111445 * sind +
               a_const::f2a111445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a111455 * sind +
               a_const::f2a111455 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * xi4 *
              (a_const::f0a11145 + a_const::f1a11145 * sind +
               a_const::f2a11145 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a111555 * sind +
               a_const::f2a111555 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 3) * xi5 *
              (a_const::f0a11155 + a_const::f1a11155 * sind +
               a_const::f2a11155 * std::pow(sind, 2)) +
          std::pow(xi1, 3) * (a_const::f0a1115 + a_const::f1a1115 * sind +
                              a_const::f2a1115 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a112225 * sind +
               a_const::f2a112225 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a112235 * sind +
               a_const::f2a112235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a112245 * sind +
               a_const::f2a112245 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a112255 * sind +
               a_const::f2a112255 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f0a11225 + a_const::f1a11225 * sind +
               a_const::f2a11225 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a112335 * sind +
               a_const::f2a112335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a112345 * sind +
               a_const::f2a112345 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a112355 * sind +
               a_const::f2a112355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f0a11235 + a_const::f1a11235 * sind +
               a_const::f2a11235 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a112445 * sind +
               a_const::f2a112445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a112455 * sind +
               a_const::f2a112455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f0a11245 + a_const::f1a11245 * sind +
               a_const::f2a11245 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a112555 * sind +
               a_const::f2a112555 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f0a11255 + a_const::f1a11255 * sind +
               a_const::f2a11255 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi2 *
              (a_const::f0a1125 + a_const::f1a1125 * sind +
               a_const::f2a1125 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a113335 * sind +
               a_const::f2a113335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a113345 * sind +
               a_const::f2a113345 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a113355 * sind +
               a_const::f2a113355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f0a11335 + a_const::f1a11335 * sind +
               a_const::f2a11335 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a113445 * sind +
               a_const::f2a113445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a113455 * sind +
               a_const::f2a113455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f0a11345 + a_const::f1a11345 * sind +
               a_const::f2a11345 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a113555 * sind +
               a_const::f2a113555 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f0a11355 + a_const::f1a11355 * sind +
               a_const::f2a11355 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi3 *
              (a_const::f0a1135 + a_const::f1a1135 * sind +
               a_const::f2a1135 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a114445 * sind +
               a_const::f2a114445 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a114455 * sind +
               a_const::f2a114455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f0a11445 + a_const::f1a11445 * sind +
               a_const::f2a11445 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a114555 * sind +
               a_const::f2a114555 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f0a11455 + a_const::f1a11455 * sind +
               a_const::f2a11455 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * xi4 *
              (a_const::f0a1145 + a_const::f1a1145 * sind +
               a_const::f2a1145 * std::pow(sind, 2)) +
          4 * std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a115555 * sind +
               a_const::f2a115555 * std::pow(sind, 2)) +
          3 * std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f0a11555 + a_const::f1a11555 * sind +
               a_const::f2a11555 * std::pow(sind, 2)) +
          2 * std::pow(xi1, 2) * xi5 *
              (a_const::f0a1155 + a_const::f1a1155 * sind +
               a_const::f2a1155 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * (a_const::f0a115 + a_const::f1a115 * sind +
                              a_const::f2a115 * std::pow(sind, 2) +
                              a_const::f3a115 * std::pow(sind, 3)) +
          xi1 * std::pow(xi2, 4) *
              (a_const::f1a122225 * sind +
               a_const::f2a122225 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a122235 * sind +
               a_const::f2a122235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a122245 * sind +
               a_const::f2a122245 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a122255 * sind +
               a_const::f2a122255 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 3) *
              (a_const::f0a12225 + a_const::f1a12225 * sind +
               a_const::f2a12225 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a122335 * sind +
               a_const::f2a122335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a122345 * sind +
               a_const::f2a122345 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a122355 * sind +
               a_const::f2a122355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f0a12235 + a_const::f1a12235 * sind +
               a_const::f2a12235 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a122445 * sind +
               a_const::f2a122445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a122455 * sind +
               a_const::f2a122455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f0a12245 + a_const::f1a12245 * sind +
               a_const::f2a12245 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a122555 * sind +
               a_const::f2a122555 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a12255 + a_const::f1a12255 * sind +
               a_const::f2a12255 * std::pow(sind, 2)) +
          xi1 * std::pow(xi2, 2) *
              (a_const::f0a1225 + a_const::f1a1225 * sind +
               a_const::f2a1225 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a123335 * sind +
               a_const::f2a123335 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a123345 * sind +
               a_const::f2a123345 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a123355 * sind +
               a_const::f2a123355 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f0a12335 + a_const::f1a12335 * sind +
               a_const::f2a12335 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a123445 * sind +
               a_const::f2a123445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a123455 * sind +
               a_const::f2a123455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 * xi4 *
              (a_const::f0a12345 + a_const::f1a12345 * sind +
               a_const::f2a12345 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a123555 * sind +
               a_const::f2a123555 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi3 * xi5 *
              (a_const::f0a12355 + a_const::f1a12355 * sind +
               a_const::f2a12355 * std::pow(sind, 2)) +
          xi1 * xi2 * xi3 *
              (a_const::f0a1235 + a_const::f1a1235 * sind +
               a_const::f2a1235 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a124445 * sind +
               a_const::f2a124445 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a124455 * sind +
               a_const::f2a124455 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f0a12445 + a_const::f1a12445 * sind +
               a_const::f2a12445 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a124555 * sind +
               a_const::f2a124555 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi4 * xi5 *
              (a_const::f0a12455 + a_const::f1a12455 * sind +
               a_const::f2a12455 * std::pow(sind, 2)) +
          xi1 * xi2 * xi4 *
              (a_const::f0a1245 + a_const::f1a1245 * sind +
               a_const::f2a1245 * std::pow(sind, 2)) +
          4 * xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a125555 * sind +
               a_const::f2a125555 * std::pow(sind, 2)) +
          3 * xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a12555 + a_const::f1a12555 * sind +
               a_const::f2a12555 * std::pow(sind, 2)) +
          2 * xi1 * xi2 * xi5 *
              (a_const::f0a1255 + a_const::f1a1255 * sind +
               a_const::f2a1255 * std::pow(sind, 2)) +
          xi1 * xi2 *
              (a_const::f0a125 + a_const::f1a125 * sind +
               a_const::f2a125 * std::pow(sind, 2) +
               a_const::f3a125 * std::pow(sind, 3)) +
          xi1 * std::pow(xi3, 4) *
              (a_const::f1a133335 * sind +
               a_const::f2a133335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a133345 * sind +
               a_const::f2a133345 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a133355 * sind +
               a_const::f2a133355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 3) *
              (a_const::f0a13335 + a_const::f1a13335 * sind +
               a_const::f2a13335 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a133445 * sind +
               a_const::f2a133445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a133455 * sind +
               a_const::f2a133455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a13345 + a_const::f1a13345 * sind +
               a_const::f2a13345 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a133555 * sind +
               a_const::f2a133555 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a13355 + a_const::f1a13355 * sind +
               a_const::f2a13355 * std::pow(sind, 2)) +
          xi1 * std::pow(xi3, 2) *
              (a_const::f0a1335 + a_const::f1a1335 * sind +
               a_const::f2a1335 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a134445 * sind +
               a_const::f2a134445 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a134455 * sind +
               a_const::f2a134455 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a13445 + a_const::f1a13445 * sind +
               a_const::f2a13445 * std::pow(sind, 2)) +
          3 * xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a134555 * sind +
               a_const::f2a134555 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi4 * xi5 *
              (a_const::f0a13455 + a_const::f1a13455 * sind +
               a_const::f2a13455 * std::pow(sind, 2)) +
          xi1 * xi3 * xi4 *
              (a_const::f0a1345 + a_const::f1a1345 * sind +
               a_const::f2a1345 * std::pow(sind, 2)) +
          4 * xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a135555 * sind +
               a_const::f2a135555 * std::pow(sind, 2)) +
          3 * xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a13555 + a_const::f1a13555 * sind +
               a_const::f2a13555 * std::pow(sind, 2)) +
          2 * xi1 * xi3 * xi5 *
              (a_const::f0a1355 + a_const::f1a1355 * sind +
               a_const::f2a1355 * std::pow(sind, 2)) +
          xi1 * xi3 *
              (a_const::f0a135 + a_const::f1a135 * sind +
               a_const::f2a135 * std::pow(sind, 2) +
               a_const::f3a135 * std::pow(sind, 3)) +
          xi1 * std::pow(xi4, 4) *
              (a_const::f1a144445 * sind +
               a_const::f2a144445 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a144455 * sind +
               a_const::f2a144455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 3) *
              (a_const::f0a14445 + a_const::f1a14445 * sind +
               a_const::f2a14445 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a144555 * sind +
               a_const::f2a144555 * std::pow(sind, 2)) +
          2 * xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a14455 + a_const::f1a14455 * sind +
               a_const::f2a14455 * std::pow(sind, 2)) +
          xi1 * std::pow(xi4, 2) *
              (a_const::f0a1445 + a_const::f1a1445 * sind +
               a_const::f2a1445 * std::pow(sind, 2)) +
          4 * xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a145555 * sind +
               a_const::f2a145555 * std::pow(sind, 2)) +
          3 * xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a14555 + a_const::f1a14555 * sind +
               a_const::f2a14555 * std::pow(sind, 2)) +
          2 * xi1 * xi4 * xi5 *
              (a_const::f0a1455 + a_const::f1a1455 * sind +
               a_const::f2a1455 * std::pow(sind, 2)) +
          xi1 * xi4 *
              (a_const::f0a145 + a_const::f1a145 * sind +
               a_const::f2a145 * std::pow(sind, 2) +
               a_const::f3a145 * std::pow(sind, 3)) +
          5 * xi1 * std::pow(xi5, 4) *
              (a_const::f1a155555 * sind +
               a_const::f2a155555 * std::pow(sind, 2)) +
          4 * xi1 * std::pow(xi5, 3) *
              (a_const::f0a15555 + a_const::f1a15555 * sind +
               a_const::f2a15555 * std::pow(sind, 2)) +
          3 * xi1 * std::pow(xi5, 2) *
              (a_const::f0a1555 + a_const::f1a1555 * sind +
               a_const::f2a1555 * std::pow(sind, 2)) +
          2 * xi1 * xi5 *
              (a_const::f0a155 + a_const::f1a155 * sind +
               a_const::f2a155 * std::pow(sind, 2) +
               a_const::f3a155 * std::pow(sind, 3)) +
          xi1 * (a_const::f0a15 + a_const::f1a15 * sind +
                 a_const::f2a15 * std::pow(sind, 2) +
                 a_const::f3a15 * std::pow(sind, 3) +
                 a_const::f4a15 * std::pow(sind, 4)) +
          std::pow(xi2, 5) * (a_const::f1a222225 * sind +
                              a_const::f2a222225 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi3 *
              (a_const::f1a222235 * sind +
               a_const::f2a222235 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * xi4 *
              (a_const::f1a222245 * sind +
               a_const::f2a222245 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 4) * xi5 *
              (a_const::f1a222255 * sind +
               a_const::f2a222255 * std::pow(sind, 2)) +
          std::pow(xi2, 4) * (a_const::f0a22225 + a_const::f1a22225 * sind +
                              a_const::f2a22225 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a222335 * sind +
               a_const::f2a222335 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a222345 * sind +
               a_const::f2a222345 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a222355 * sind +
               a_const::f2a222355 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi3 *
              (a_const::f0a22235 + a_const::f1a22235 * sind +
               a_const::f2a22235 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a222445 * sind +
               a_const::f2a222445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a222455 * sind +
               a_const::f2a222455 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * xi4 *
              (a_const::f0a22245 + a_const::f1a22245 * sind +
               a_const::f2a22245 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a222555 * sind +
               a_const::f2a222555 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 3) * xi5 *
              (a_const::f0a22255 + a_const::f1a22255 * sind +
               a_const::f2a22255 * std::pow(sind, 2)) +
          std::pow(xi2, 3) * (a_const::f0a2225 + a_const::f1a2225 * sind +
                              a_const::f2a2225 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a223335 * sind +
               a_const::f2a223335 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a223345 * sind +
               a_const::f2a223345 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a223355 * sind +
               a_const::f2a223355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f0a22335 + a_const::f1a22335 * sind +
               a_const::f2a22335 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a223445 * sind +
               a_const::f2a223445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a223455 * sind +
               a_const::f2a223455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f0a22345 + a_const::f1a22345 * sind +
               a_const::f2a22345 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a223555 * sind +
               a_const::f2a223555 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f0a22355 + a_const::f1a22355 * sind +
               a_const::f2a22355 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi3 *
              (a_const::f0a2235 + a_const::f1a2235 * sind +
               a_const::f2a2235 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a224445 * sind +
               a_const::f2a224445 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a224455 * sind +
               a_const::f2a224455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f0a22445 + a_const::f1a22445 * sind +
               a_const::f2a22445 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a224555 * sind +
               a_const::f2a224555 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f0a22455 + a_const::f1a22455 * sind +
               a_const::f2a22455 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * xi4 *
              (a_const::f0a2245 + a_const::f1a2245 * sind +
               a_const::f2a2245 * std::pow(sind, 2)) +
          4 * std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a225555 * sind +
               a_const::f2a225555 * std::pow(sind, 2)) +
          3 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f0a22555 + a_const::f1a22555 * sind +
               a_const::f2a22555 * std::pow(sind, 2)) +
          2 * std::pow(xi2, 2) * xi5 *
              (a_const::f0a2255 + a_const::f1a2255 * sind +
               a_const::f2a2255 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * (a_const::f0a225 + a_const::f1a225 * sind +
                              a_const::f2a225 * std::pow(sind, 2) +
                              a_const::f3a225 * std::pow(sind, 3)) +
          xi2 * std::pow(xi3, 4) *
              (a_const::f1a233335 * sind +
               a_const::f2a233335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a233345 * sind +
               a_const::f2a233345 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a233355 * sind +
               a_const::f2a233355 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 3) *
              (a_const::f0a23335 + a_const::f1a23335 * sind +
               a_const::f2a23335 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a233445 * sind +
               a_const::f2a233445 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a233455 * sind +
               a_const::f2a233455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f0a23345 + a_const::f1a23345 * sind +
               a_const::f2a23345 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a233555 * sind +
               a_const::f2a233555 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a23355 + a_const::f1a23355 * sind +
               a_const::f2a23355 * std::pow(sind, 2)) +
          xi2 * std::pow(xi3, 2) *
              (a_const::f0a2335 + a_const::f1a2335 * sind +
               a_const::f2a2335 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a234445 * sind +
               a_const::f2a234445 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a234455 * sind +
               a_const::f2a234455 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f0a23445 + a_const::f1a23445 * sind +
               a_const::f2a23445 * std::pow(sind, 2)) +
          3 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a234555 * sind +
               a_const::f2a234555 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f0a23455 + a_const::f1a23455 * sind +
               a_const::f2a23455 * std::pow(sind, 2)) +
          xi2 * xi3 * xi4 *
              (a_const::f0a2345 + a_const::f1a2345 * sind +
               a_const::f2a2345 * std::pow(sind, 2)) +
          4 * xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a235555 * sind +
               a_const::f2a235555 * std::pow(sind, 2)) +
          3 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a23555 + a_const::f1a23555 * sind +
               a_const::f2a23555 * std::pow(sind, 2)) +
          2 * xi2 * xi3 * xi5 *
              (a_const::f0a2355 + a_const::f1a2355 * sind +
               a_const::f2a2355 * std::pow(sind, 2)) +
          xi2 * xi3 *
              (a_const::f0a235 + a_const::f1a235 * sind +
               a_const::f2a235 * std::pow(sind, 2) +
               a_const::f3a235 * std::pow(sind, 3)) +
          xi2 * std::pow(xi4, 4) *
              (a_const::f1a244445 * sind +
               a_const::f2a244445 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a244455 * sind +
               a_const::f2a244455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 3) *
              (a_const::f0a24445 + a_const::f1a24445 * sind +
               a_const::f2a24445 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a244555 * sind +
               a_const::f2a244555 * std::pow(sind, 2)) +
          2 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a24455 + a_const::f1a24455 * sind +
               a_const::f2a24455 * std::pow(sind, 2)) +
          xi2 * std::pow(xi4, 2) *
              (a_const::f0a2445 + a_const::f1a2445 * sind +
               a_const::f2a2445 * std::pow(sind, 2)) +
          4 * xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a245555 * sind +
               a_const::f2a245555 * std::pow(sind, 2)) +
          3 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a24555 + a_const::f1a24555 * sind +
               a_const::f2a24555 * std::pow(sind, 2)) +
          2 * xi2 * xi4 * xi5 *
              (a_const::f0a2455 + a_const::f1a2455 * sind +
               a_const::f2a2455 * std::pow(sind, 2)) +
          xi2 * xi4 *
              (a_const::f0a245 + a_const::f1a245 * sind +
               a_const::f2a245 * std::pow(sind, 2) +
               a_const::f3a245 * std::pow(sind, 3)) +
          5 * xi2 * std::pow(xi5, 4) *
              (a_const::f1a255555 * sind +
               a_const::f2a255555 * std::pow(sind, 2)) +
          4 * xi2 * std::pow(xi5, 3) *
              (a_const::f0a25555 + a_const::f1a25555 * sind +
               a_const::f2a25555 * std::pow(sind, 2)) +
          3 * xi2 * std::pow(xi5, 2) *
              (a_const::f0a2555 + a_const::f1a2555 * sind +
               a_const::f2a2555 * std::pow(sind, 2)) +
          2 * xi2 * xi5 *
              (a_const::f0a255 + a_const::f1a255 * sind +
               a_const::f2a255 * std::pow(sind, 2) +
               a_const::f3a255 * std::pow(sind, 3)) +
          xi2 * (a_const::f0a25 + a_const::f1a25 * sind +
                 a_const::f2a25 * std::pow(sind, 2) +
                 a_const::f3a25 * std::pow(sind, 3) +
                 a_const::f4a25 * std::pow(sind, 4)) +
          std::pow(xi3, 5) * (a_const::f1a333335 * sind +
                              a_const::f2a333335 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * xi4 *
              (a_const::f1a333345 * sind +
               a_const::f2a333345 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a333355 * sind +
               a_const::f2a333355 * std::pow(sind, 2)) +
          std::pow(xi3, 4) * (a_const::f0a33335 + a_const::f1a33335 * sind +
                              a_const::f2a33335 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a333445 * sind +
               a_const::f2a333445 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a333455 * sind +
               a_const::f2a333455 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * xi4 *
              (a_const::f0a33345 + a_const::f1a33345 * sind +
               a_const::f2a33345 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a333555 * sind +
               a_const::f2a333555 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 3) * xi5 *
              (a_const::f0a33355 + a_const::f1a33355 * sind +
               a_const::f2a33355 * std::pow(sind, 2)) +
          std::pow(xi3, 3) * (a_const::f0a3335 + a_const::f1a3335 * sind +
                              a_const::f2a3335 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a334445 * sind +
               a_const::f2a334445 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a334455 * sind +
               a_const::f2a334455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f0a33445 + a_const::f1a33445 * sind +
               a_const::f2a33445 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a334555 * sind +
               a_const::f2a334555 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f0a33455 + a_const::f1a33455 * sind +
               a_const::f2a33455 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * xi4 *
              (a_const::f0a3345 + a_const::f1a3345 * sind +
               a_const::f2a3345 * std::pow(sind, 2)) +
          4 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a335555 * sind +
               a_const::f2a335555 * std::pow(sind, 2)) +
          3 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f0a33555 + a_const::f1a33555 * sind +
               a_const::f2a33555 * std::pow(sind, 2)) +
          2 * std::pow(xi3, 2) * xi5 *
              (a_const::f0a3355 + a_const::f1a3355 * sind +
               a_const::f2a3355 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * (a_const::f0a335 + a_const::f1a335 * sind +
                              a_const::f2a335 * std::pow(sind, 2) +
                              a_const::f3a335 * std::pow(sind, 3)) +
          xi3 * std::pow(xi4, 4) *
              (a_const::f1a344445 * sind +
               a_const::f2a344445 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a344455 * sind +
               a_const::f2a344455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 3) *
              (a_const::f0a34445 + a_const::f1a34445 * sind +
               a_const::f2a34445 * std::pow(sind, 2)) +
          3 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a344555 * sind +
               a_const::f2a344555 * std::pow(sind, 2)) +
          2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a34455 + a_const::f1a34455 * sind +
               a_const::f2a34455 * std::pow(sind, 2)) +
          xi3 * std::pow(xi4, 2) *
              (a_const::f0a3445 + a_const::f1a3445 * sind +
               a_const::f2a3445 * std::pow(sind, 2)) +
          4 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a345555 * sind +
               a_const::f2a345555 * std::pow(sind, 2)) +
          3 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a34555 + a_const::f1a34555 * sind +
               a_const::f2a34555 * std::pow(sind, 2)) +
          2 * xi3 * xi4 * xi5 *
              (a_const::f0a3455 + a_const::f1a3455 * sind +
               a_const::f2a3455 * std::pow(sind, 2)) +
          xi3 * xi4 *
              (a_const::f0a345 + a_const::f1a345 * sind +
               a_const::f2a345 * std::pow(sind, 2) +
               a_const::f3a345 * std::pow(sind, 3)) +
          5 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a355555 * sind +
               a_const::f2a355555 * std::pow(sind, 2)) +
          4 * xi3 * std::pow(xi5, 3) *
              (a_const::f0a35555 + a_const::f1a35555 * sind +
               a_const::f2a35555 * std::pow(sind, 2)) +
          3 * xi3 * std::pow(xi5, 2) *
              (a_const::f0a3555 + a_const::f1a3555 * sind +
               a_const::f2a3555 * std::pow(sind, 2)) +
          2 * xi3 * xi5 *
              (a_const::f0a355 + a_const::f1a355 * sind +
               a_const::f2a355 * std::pow(sind, 2) +
               a_const::f3a355 * std::pow(sind, 3)) +
          xi3 * (a_const::f0a35 + a_const::f1a35 * sind +
                 a_const::f2a35 * std::pow(sind, 2) +
                 a_const::f3a35 * std::pow(sind, 3) +
                 a_const::f4a35 * std::pow(sind, 4)) +
          std::pow(xi4, 5) * (a_const::f1a444445 * sind +
                              a_const::f2a444445 * std::pow(sind, 2)) +
          2 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a444455 * sind +
               a_const::f2a444455 * std::pow(sind, 2)) +
          std::pow(xi4, 4) * (a_const::f0a44445 + a_const::f1a44445 * sind +
                              a_const::f2a44445 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a444555 * sind +
               a_const::f2a444555 * std::pow(sind, 2)) +
          2 * std::pow(xi4, 3) * xi5 *
              (a_const::f0a44455 + a_const::f1a44455 * sind +
               a_const::f2a44455 * std::pow(sind, 2)) +
          std::pow(xi4, 3) * (a_const::f0a4445 + a_const::f1a4445 * sind +
                              a_const::f2a4445 * std::pow(sind, 2)) +
          4 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a445555 * sind +
               a_const::f2a445555 * std::pow(sind, 2)) +
          3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f0a44555 + a_const::f1a44555 * sind +
               a_const::f2a44555 * std::pow(sind, 2)) +
          2 * std::pow(xi4, 2) * xi5 *
              (a_const::f0a4455 + a_const::f1a4455 * sind +
               a_const::f2a4455 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * (a_const::f0a445 + a_const::f1a445 * sind +
                              a_const::f2a445 * std::pow(sind, 2) +
                              a_const::f3a445 * std::pow(sind, 3)) +
          5 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a455555 * sind +
               a_const::f2a455555 * std::pow(sind, 2)) +
          4 * xi4 * std::pow(xi5, 3) *
              (a_const::f0a45555 + a_const::f1a45555 * sind +
               a_const::f2a45555 * std::pow(sind, 2)) +
          3 * xi4 * std::pow(xi5, 2) *
              (a_const::f0a4555 + a_const::f1a4555 * sind +
               a_const::f2a4555 * std::pow(sind, 2)) +
          2 * xi4 * xi5 *
              (a_const::f0a455 + a_const::f1a455 * sind +
               a_const::f2a455 * std::pow(sind, 2) +
               a_const::f3a455 * std::pow(sind, 3)) +
          xi4 * (a_const::f0a45 + a_const::f1a45 * sind +
                 a_const::f2a45 * std::pow(sind, 2) +
                 a_const::f3a45 * std::pow(sind, 3) +
                 a_const::f4a45 * std::pow(sind, 4)) +
          6 * std::pow(xi5, 5) *
              (a_const::f1a555555 * sind +
               a_const::f2a555555 * std::pow(sind, 2)) +
          5 * std::pow(xi5, 4) *
              (a_const::f0a55555 + a_const::f1a55555 * sind +
               a_const::f2a55555 * std::pow(sind, 2)) +
          4 * std::pow(xi5, 3) *
              (a_const::f0a5555 + a_const::f1a5555 * sind +
               a_const::f2a5555 * std::pow(sind, 2)) +
          3 * std::pow(xi5, 2) *
              (a_const::f0a555 + a_const::f1a555 * sind +
               a_const::f2a555 * std::pow(sind, 2) +
               a_const::f3a555 * std::pow(sind, 3)) +
          2 * xi5 *
              (a_const::f0a55 + a_const::f1a55 * sind +
               a_const::f2a55 * std::pow(sind, 2) +
               a_const::f3a55 * std::pow(sind, 3) +
               a_const::f4a55 * std::pow(sind, 4));
      dvdsind =
          a_const::f1a + 2 * a_const::f2a * sind +
          3 * a_const::f3a * std::pow(sind, 2) +
          4 * a_const::f4a * std::pow(sind, 3) +
          5 * a_const::f5a * std::pow(sind, 4) +
          6 * a_const::f6a * std::pow(sind, 5) +
          7 * a_const::f7a * std::pow(sind, 6) +
          8 * a_const::f8a * std::pow(sind, 7) +
          std::pow(xi1, 6) *
              (a_const::f1a111111 + 2 * a_const::f2a111111 * sind) +
          std::pow(xi1, 5) * xi2 *
              (a_const::f1a111112 + 2 * a_const::f2a111112 * sind) +
          std::pow(xi1, 5) * xi3 *
              (a_const::f1a111113 + 2 * a_const::f2a111113 * sind) +
          std::pow(xi1, 5) * xi4 *
              (a_const::f1a111114 + 2 * a_const::f2a111114 * sind) +
          std::pow(xi1, 5) * xi5 *
              (a_const::f1a111115 + 2 * a_const::f2a111115 * sind) +
          std::pow(xi1, 5) *
              (a_const::f1a11111 + 2 * a_const::f2a11111 * sind) +
          std::pow(xi1, 4) * std::pow(xi2, 2) *
              (a_const::f1a111122 + 2 * a_const::f2a111122 * sind) +
          std::pow(xi1, 4) * xi2 * xi3 *
              (a_const::f1a111123 + 2 * a_const::f2a111123 * sind) +
          std::pow(xi1, 4) * xi2 * xi4 *
              (a_const::f1a111124 + 2 * a_const::f2a111124 * sind) +
          std::pow(xi1, 4) * xi2 * xi5 *
              (a_const::f1a111125 + 2 * a_const::f2a111125 * sind) +
          std::pow(xi1, 4) * xi2 *
              (a_const::f1a11112 + 2 * a_const::f2a11112 * sind) +
          std::pow(xi1, 4) * std::pow(xi3, 2) *
              (a_const::f1a111133 + 2 * a_const::f2a111133 * sind) +
          std::pow(xi1, 4) * xi3 * xi4 *
              (a_const::f1a111134 + 2 * a_const::f2a111134 * sind) +
          std::pow(xi1, 4) * xi3 * xi5 *
              (a_const::f1a111135 + 2 * a_const::f2a111135 * sind) +
          std::pow(xi1, 4) * xi3 *
              (a_const::f1a11113 + 2 * a_const::f2a11113 * sind) +
          std::pow(xi1, 4) * std::pow(xi4, 2) *
              (a_const::f1a111144 + 2 * a_const::f2a111144 * sind) +
          std::pow(xi1, 4) * xi4 * xi5 *
              (a_const::f1a111145 + 2 * a_const::f2a111145 * sind) +
          std::pow(xi1, 4) * xi4 *
              (a_const::f1a11114 + 2 * a_const::f2a11114 * sind) +
          std::pow(xi1, 4) * std::pow(xi5, 2) *
              (a_const::f1a111155 + 2 * a_const::f2a111155 * sind) +
          std::pow(xi1, 4) * xi5 *
              (a_const::f1a11115 + 2 * a_const::f2a11115 * sind) +
          std::pow(xi1, 4) * (a_const::f1a1111 + 2 * a_const::f2a1111 * sind) +
          std::pow(xi1, 3) * std::pow(xi2, 3) *
              (a_const::f1a111222 + 2 * a_const::f2a111222 * sind) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a111223 + 2 * a_const::f2a111223 * sind) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a111224 + 2 * a_const::f2a111224 * sind) +
          std::pow(xi1, 3) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a111225 + 2 * a_const::f2a111225 * sind) +
          std::pow(xi1, 3) * std::pow(xi2, 2) *
              (a_const::f1a11122 + 2 * a_const::f2a11122 * sind) +
          std::pow(xi1, 3) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a111233 + 2 * a_const::f2a111233 * sind) +
          std::pow(xi1, 3) * xi2 * xi3 * xi4 *
              (a_const::f1a111234 + 2 * a_const::f2a111234 * sind) +
          std::pow(xi1, 3) * xi2 * xi3 * xi5 *
              (a_const::f1a111235 + 2 * a_const::f2a111235 * sind) +
          std::pow(xi1, 3) * xi2 * xi3 *
              (a_const::f1a11123 + 2 * a_const::f2a11123 * sind) +
          std::pow(xi1, 3) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a111244 + 2 * a_const::f2a111244 * sind) +
          std::pow(xi1, 3) * xi2 * xi4 * xi5 *
              (a_const::f1a111245 + 2 * a_const::f2a111245 * sind) +
          std::pow(xi1, 3) * xi2 * xi4 *
              (a_const::f1a11124 + 2 * a_const::f2a11124 * sind) +
          std::pow(xi1, 3) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a111255 + 2 * a_const::f2a111255 * sind) +
          std::pow(xi1, 3) * xi2 * xi5 *
              (a_const::f1a11125 + 2 * a_const::f2a11125 * sind) +
          std::pow(xi1, 3) * xi2 *
              (a_const::f1a1112 + 2 * a_const::f2a1112 * sind) +
          std::pow(xi1, 3) * std::pow(xi3, 3) *
              (a_const::f1a111333 + 2 * a_const::f2a111333 * sind) +
          std::pow(xi1, 3) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a111334 + 2 * a_const::f2a111334 * sind) +
          std::pow(xi1, 3) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a111335 + 2 * a_const::f2a111335 * sind) +
          std::pow(xi1, 3) * std::pow(xi3, 2) *
              (a_const::f1a11133 + 2 * a_const::f2a11133 * sind) +
          std::pow(xi1, 3) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a111344 + 2 * a_const::f2a111344 * sind) +
          std::pow(xi1, 3) * xi3 * xi4 * xi5 *
              (a_const::f1a111345 + 2 * a_const::f2a111345 * sind) +
          std::pow(xi1, 3) * xi3 * xi4 *
              (a_const::f1a11134 + 2 * a_const::f2a11134 * sind) +
          std::pow(xi1, 3) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a111355 + 2 * a_const::f2a111355 * sind) +
          std::pow(xi1, 3) * xi3 * xi5 *
              (a_const::f1a11135 + 2 * a_const::f2a11135 * sind) +
          std::pow(xi1, 3) * xi3 *
              (a_const::f1a1113 + 2 * a_const::f2a1113 * sind) +
          std::pow(xi1, 3) * std::pow(xi4, 3) *
              (a_const::f1a111444 + 2 * a_const::f2a111444 * sind) +
          std::pow(xi1, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a111445 + 2 * a_const::f2a111445 * sind) +
          std::pow(xi1, 3) * std::pow(xi4, 2) *
              (a_const::f1a11144 + 2 * a_const::f2a11144 * sind) +
          std::pow(xi1, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a111455 + 2 * a_const::f2a111455 * sind) +
          std::pow(xi1, 3) * xi4 * xi5 *
              (a_const::f1a11145 + 2 * a_const::f2a11145 * sind) +
          std::pow(xi1, 3) * xi4 *
              (a_const::f1a1114 + 2 * a_const::f2a1114 * sind) +
          std::pow(xi1, 3) * std::pow(xi5, 3) *
              (a_const::f1a111555 + 2 * a_const::f2a111555 * sind) +
          std::pow(xi1, 3) * std::pow(xi5, 2) *
              (a_const::f1a11155 + 2 * a_const::f2a11155 * sind) +
          std::pow(xi1, 3) * xi5 *
              (a_const::f1a1115 + 2 * a_const::f2a1115 * sind) +
          std::pow(xi1, 3) * (a_const::f1a111 + 2 * a_const::f2a111 * sind +
                              3 * a_const::f3a111 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi2, 4) *
              (a_const::f1a112222 + 2 * a_const::f2a112222 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi3 *
              (a_const::f1a112223 + 2 * a_const::f2a112223 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi4 *
              (a_const::f1a112224 + 2 * a_const::f2a112224 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 3) * xi5 *
              (a_const::f1a112225 + 2 * a_const::f2a112225 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 3) *
              (a_const::f1a11222 + 2 * a_const::f2a11222 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a112233 + 2 * a_const::f2a112233 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a112234 + 2 * a_const::f2a112234 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a112235 + 2 * a_const::f2a112235 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi3 *
              (a_const::f1a11223 + 2 * a_const::f2a11223 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a112244 + 2 * a_const::f2a112244 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a112245 + 2 * a_const::f2a112245 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi4 *
              (a_const::f1a11224 + 2 * a_const::f2a11224 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a112255 + 2 * a_const::f2a112255 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) * xi5 *
              (a_const::f1a11225 + 2 * a_const::f2a11225 * sind) +
          std::pow(xi1, 2) * std::pow(xi2, 2) *
              (a_const::f1a1122 + 2 * a_const::f2a1122 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 3) *
              (a_const::f1a112333 + 2 * a_const::f2a112333 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a112334 + 2 * a_const::f2a112334 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a112335 + 2 * a_const::f2a112335 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi3, 2) *
              (a_const::f1a11233 + 2 * a_const::f2a11233 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a112344 + 2 * a_const::f2a112344 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a112345 + 2 * a_const::f2a112345 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 * xi4 *
              (a_const::f1a11234 + 2 * a_const::f2a11234 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a112355 + 2 * a_const::f2a112355 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 * xi5 *
              (a_const::f1a11235 + 2 * a_const::f2a11235 * sind) +
          std::pow(xi1, 2) * xi2 * xi3 *
              (a_const::f1a1123 + 2 * a_const::f2a1123 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 3) *
              (a_const::f1a112444 + 2 * a_const::f2a112444 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a112445 + 2 * a_const::f2a112445 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi4, 2) *
              (a_const::f1a11244 + 2 * a_const::f2a11244 * sind) +
          std::pow(xi1, 2) * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a112455 + 2 * a_const::f2a112455 * sind) +
          std::pow(xi1, 2) * xi2 * xi4 * xi5 *
              (a_const::f1a11245 + 2 * a_const::f2a11245 * sind) +
          std::pow(xi1, 2) * xi2 * xi4 *
              (a_const::f1a1124 + 2 * a_const::f2a1124 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 3) *
              (a_const::f1a112555 + 2 * a_const::f2a112555 * sind) +
          std::pow(xi1, 2) * xi2 * std::pow(xi5, 2) *
              (a_const::f1a11255 + 2 * a_const::f2a11255 * sind) +
          std::pow(xi1, 2) * xi2 * xi5 *
              (a_const::f1a1125 + 2 * a_const::f2a1125 * sind) +
          std::pow(xi1, 2) * xi2 *
              (a_const::f1a112 + 2 * a_const::f2a112 * sind +
               3 * a_const::f3a112 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi3, 4) *
              (a_const::f1a113333 + 2 * a_const::f2a113333 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 3) * xi4 *
              (a_const::f1a113334 + 2 * a_const::f2a113334 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 3) * xi5 *
              (a_const::f1a113335 + 2 * a_const::f2a113335 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 3) *
              (a_const::f1a11333 + 2 * a_const::f2a11333 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a113344 + 2 * a_const::f2a113344 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a113345 + 2 * a_const::f2a113345 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a11334 + 2 * a_const::f2a11334 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a113355 + 2 * a_const::f2a113355 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a11335 + 2 * a_const::f2a11335 * sind) +
          std::pow(xi1, 2) * std::pow(xi3, 2) *
              (a_const::f1a1133 + 2 * a_const::f2a1133 * sind) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 3) *
              (a_const::f1a113444 + 2 * a_const::f2a113444 * sind) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a113445 + 2 * a_const::f2a113445 * sind) +
          std::pow(xi1, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a11344 + 2 * a_const::f2a11344 * sind) +
          std::pow(xi1, 2) * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a113455 + 2 * a_const::f2a113455 * sind) +
          std::pow(xi1, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a11345 + 2 * a_const::f2a11345 * sind) +
          std::pow(xi1, 2) * xi3 * xi4 *
              (a_const::f1a1134 + 2 * a_const::f2a1134 * sind) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 3) *
              (a_const::f1a113555 + 2 * a_const::f2a113555 * sind) +
          std::pow(xi1, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a11355 + 2 * a_const::f2a11355 * sind) +
          std::pow(xi1, 2) * xi3 * xi5 *
              (a_const::f1a1135 + 2 * a_const::f2a1135 * sind) +
          std::pow(xi1, 2) * xi3 *
              (a_const::f1a113 + 2 * a_const::f2a113 * sind +
               3 * a_const::f3a113 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi4, 4) *
              (a_const::f1a114444 + 2 * a_const::f2a114444 * sind) +
          std::pow(xi1, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a114445 + 2 * a_const::f2a114445 * sind) +
          std::pow(xi1, 2) * std::pow(xi4, 3) *
              (a_const::f1a11444 + 2 * a_const::f2a11444 * sind) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a114455 + 2 * a_const::f2a114455 * sind) +
          std::pow(xi1, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a11445 + 2 * a_const::f2a11445 * sind) +
          std::pow(xi1, 2) * std::pow(xi4, 2) *
              (a_const::f1a1144 + 2 * a_const::f2a1144 * sind) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a114555 + 2 * a_const::f2a114555 * sind) +
          std::pow(xi1, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a11455 + 2 * a_const::f2a11455 * sind) +
          std::pow(xi1, 2) * xi4 * xi5 *
              (a_const::f1a1145 + 2 * a_const::f2a1145 * sind) +
          std::pow(xi1, 2) * xi4 *
              (a_const::f1a114 + 2 * a_const::f2a114 * sind +
               3 * a_const::f3a114 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * std::pow(xi5, 4) *
              (a_const::f1a115555 + 2 * a_const::f2a115555 * sind) +
          std::pow(xi1, 2) * std::pow(xi5, 3) *
              (a_const::f1a11555 + 2 * a_const::f2a11555 * sind) +
          std::pow(xi1, 2) * std::pow(xi5, 2) *
              (a_const::f1a1155 + 2 * a_const::f2a1155 * sind) +
          std::pow(xi1, 2) * xi5 *
              (a_const::f1a115 + 2 * a_const::f2a115 * sind +
               3 * a_const::f3a115 * std::pow(sind, 2)) +
          std::pow(xi1, 2) * (a_const::f1a11 + 2 * a_const::f2a11 * sind +
                              3 * a_const::f3a11 * std::pow(sind, 2) +
                              4 * a_const::f4a11 * std::pow(sind, 3)) +
          xi1 * std::pow(xi2, 5) *
              (a_const::f1a122222 + 2 * a_const::f2a122222 * sind) +
          xi1 * std::pow(xi2, 4) * xi3 *
              (a_const::f1a122223 + 2 * a_const::f2a122223 * sind) +
          xi1 * std::pow(xi2, 4) * xi4 *
              (a_const::f1a122224 + 2 * a_const::f2a122224 * sind) +
          xi1 * std::pow(xi2, 4) * xi5 *
              (a_const::f1a122225 + 2 * a_const::f2a122225 * sind) +
          xi1 * std::pow(xi2, 4) *
              (a_const::f1a12222 + 2 * a_const::f2a12222 * sind) +
          xi1 * std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a122233 + 2 * a_const::f2a122233 * sind) +
          xi1 * std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a122234 + 2 * a_const::f2a122234 * sind) +
          xi1 * std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a122235 + 2 * a_const::f2a122235 * sind) +
          xi1 * std::pow(xi2, 3) * xi3 *
              (a_const::f1a12223 + 2 * a_const::f2a12223 * sind) +
          xi1 * std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a122244 + 2 * a_const::f2a122244 * sind) +
          xi1 * std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a122245 + 2 * a_const::f2a122245 * sind) +
          xi1 * std::pow(xi2, 3) * xi4 *
              (a_const::f1a12224 + 2 * a_const::f2a12224 * sind) +
          xi1 * std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a122255 + 2 * a_const::f2a122255 * sind) +
          xi1 * std::pow(xi2, 3) * xi5 *
              (a_const::f1a12225 + 2 * a_const::f2a12225 * sind) +
          xi1 * std::pow(xi2, 3) *
              (a_const::f1a1222 + 2 * a_const::f2a1222 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a122333 + 2 * a_const::f2a122333 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a122334 + 2 * a_const::f2a122334 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a122335 + 2 * a_const::f2a122335 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a12233 + 2 * a_const::f2a12233 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a122344 + 2 * a_const::f2a122344 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a122345 + 2 * a_const::f2a122345 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a12234 + 2 * a_const::f2a12234 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a122355 + 2 * a_const::f2a122355 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a12235 + 2 * a_const::f2a12235 * sind) +
          xi1 * std::pow(xi2, 2) * xi3 *
              (a_const::f1a1223 + 2 * a_const::f2a1223 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a122444 + 2 * a_const::f2a122444 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a122445 + 2 * a_const::f2a122445 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a12244 + 2 * a_const::f2a12244 * sind) +
          xi1 * std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a122455 + 2 * a_const::f2a122455 * sind) +
          xi1 * std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a12245 + 2 * a_const::f2a12245 * sind) +
          xi1 * std::pow(xi2, 2) * xi4 *
              (a_const::f1a1224 + 2 * a_const::f2a1224 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a122555 + 2 * a_const::f2a122555 * sind) +
          xi1 * std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a12255 + 2 * a_const::f2a12255 * sind) +
          xi1 * std::pow(xi2, 2) * xi5 *
              (a_const::f1a1225 + 2 * a_const::f2a1225 * sind) +
          xi1 * std::pow(xi2, 2) *
              (a_const::f1a122 + 2 * a_const::f2a122 * sind +
               3 * a_const::f3a122 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi3, 4) *
              (a_const::f1a123333 + 2 * a_const::f2a123333 * sind) +
          xi1 * xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a123334 + 2 * a_const::f2a123334 * sind) +
          xi1 * xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a123335 + 2 * a_const::f2a123335 * sind) +
          xi1 * xi2 * std::pow(xi3, 3) *
              (a_const::f1a12333 + 2 * a_const::f2a12333 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a123344 + 2 * a_const::f2a123344 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a123345 + 2 * a_const::f2a123345 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a12334 + 2 * a_const::f2a12334 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a123355 + 2 * a_const::f2a123355 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a12335 + 2 * a_const::f2a12335 * sind) +
          xi1 * xi2 * std::pow(xi3, 2) *
              (a_const::f1a1233 + 2 * a_const::f2a1233 * sind) +
          xi1 * xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a123444 + 2 * a_const::f2a123444 * sind) +
          xi1 * xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a123445 + 2 * a_const::f2a123445 * sind) +
          xi1 * xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a12344 + 2 * a_const::f2a12344 * sind) +
          xi1 * xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a123455 + 2 * a_const::f2a123455 * sind) +
          xi1 * xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a12345 + 2 * a_const::f2a12345 * sind) +
          xi1 * xi2 * xi3 * xi4 *
              (a_const::f1a1234 + 2 * a_const::f2a1234 * sind) +
          xi1 * xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a123555 + 2 * a_const::f2a123555 * sind) +
          xi1 * xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a12355 + 2 * a_const::f2a12355 * sind) +
          xi1 * xi2 * xi3 * xi5 *
              (a_const::f1a1235 + 2 * a_const::f2a1235 * sind) +
          xi1 * xi2 * xi3 *
              (a_const::f1a123 + 2 * a_const::f2a123 * sind +
               3 * a_const::f3a123 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi4, 4) *
              (a_const::f1a124444 + 2 * a_const::f2a124444 * sind) +
          xi1 * xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a124445 + 2 * a_const::f2a124445 * sind) +
          xi1 * xi2 * std::pow(xi4, 3) *
              (a_const::f1a12444 + 2 * a_const::f2a12444 * sind) +
          xi1 * xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a124455 + 2 * a_const::f2a124455 * sind) +
          xi1 * xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a12445 + 2 * a_const::f2a12445 * sind) +
          xi1 * xi2 * std::pow(xi4, 2) *
              (a_const::f1a1244 + 2 * a_const::f2a1244 * sind) +
          xi1 * xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a124555 + 2 * a_const::f2a124555 * sind) +
          xi1 * xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a12455 + 2 * a_const::f2a12455 * sind) +
          xi1 * xi2 * xi4 * xi5 *
              (a_const::f1a1245 + 2 * a_const::f2a1245 * sind) +
          xi1 * xi2 * xi4 *
              (a_const::f1a124 + 2 * a_const::f2a124 * sind +
               3 * a_const::f3a124 * std::pow(sind, 2)) +
          xi1 * xi2 * std::pow(xi5, 4) *
              (a_const::f1a125555 + 2 * a_const::f2a125555 * sind) +
          xi1 * xi2 * std::pow(xi5, 3) *
              (a_const::f1a12555 + 2 * a_const::f2a12555 * sind) +
          xi1 * xi2 * std::pow(xi5, 2) *
              (a_const::f1a1255 + 2 * a_const::f2a1255 * sind) +
          xi1 * xi2 * xi5 *
              (a_const::f1a125 + 2 * a_const::f2a125 * sind +
               3 * a_const::f3a125 * std::pow(sind, 2)) +
          xi1 * xi2 *
              (a_const::f1a12 + 2 * a_const::f2a12 * sind +
               3 * a_const::f3a12 * std::pow(sind, 2) +
               4 * a_const::f4a12 * std::pow(sind, 3)) +
          xi1 * std::pow(xi3, 5) *
              (a_const::f1a133333 + 2 * a_const::f2a133333 * sind) +
          xi1 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a133334 + 2 * a_const::f2a133334 * sind) +
          xi1 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a133335 + 2 * a_const::f2a133335 * sind) +
          xi1 * std::pow(xi3, 4) *
              (a_const::f1a13333 + 2 * a_const::f2a13333 * sind) +
          xi1 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a133344 + 2 * a_const::f2a133344 * sind) +
          xi1 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a133345 + 2 * a_const::f2a133345 * sind) +
          xi1 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a13334 + 2 * a_const::f2a13334 * sind) +
          xi1 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a133355 + 2 * a_const::f2a133355 * sind) +
          xi1 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a13335 + 2 * a_const::f2a13335 * sind) +
          xi1 * std::pow(xi3, 3) *
              (a_const::f1a1333 + 2 * a_const::f2a1333 * sind) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a133444 + 2 * a_const::f2a133444 * sind) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a133445 + 2 * a_const::f2a133445 * sind) +
          xi1 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a13344 + 2 * a_const::f2a13344 * sind) +
          xi1 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a133455 + 2 * a_const::f2a133455 * sind) +
          xi1 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a13345 + 2 * a_const::f2a13345 * sind) +
          xi1 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a1334 + 2 * a_const::f2a1334 * sind) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a133555 + 2 * a_const::f2a133555 * sind) +
          xi1 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a13355 + 2 * a_const::f2a13355 * sind) +
          xi1 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a1335 + 2 * a_const::f2a1335 * sind) +
          xi1 * std::pow(xi3, 2) *
              (a_const::f1a133 + 2 * a_const::f2a133 * sind +
               3 * a_const::f3a133 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a134444 + 2 * a_const::f2a134444 * sind) +
          xi1 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a134445 + 2 * a_const::f2a134445 * sind) +
          xi1 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a13444 + 2 * a_const::f2a13444 * sind) +
          xi1 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a134455 + 2 * a_const::f2a134455 * sind) +
          xi1 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a13445 + 2 * a_const::f2a13445 * sind) +
          xi1 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a1344 + 2 * a_const::f2a1344 * sind) +
          xi1 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a134555 + 2 * a_const::f2a134555 * sind) +
          xi1 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a13455 + 2 * a_const::f2a13455 * sind) +
          xi1 * xi3 * xi4 * xi5 *
              (a_const::f1a1345 + 2 * a_const::f2a1345 * sind) +
          xi1 * xi3 * xi4 *
              (a_const::f1a134 + 2 * a_const::f2a134 * sind +
               3 * a_const::f3a134 * std::pow(sind, 2)) +
          xi1 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a135555 + 2 * a_const::f2a135555 * sind) +
          xi1 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a13555 + 2 * a_const::f2a13555 * sind) +
          xi1 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a1355 + 2 * a_const::f2a1355 * sind) +
          xi1 * xi3 * xi5 *
              (a_const::f1a135 + 2 * a_const::f2a135 * sind +
               3 * a_const::f3a135 * std::pow(sind, 2)) +
          xi1 * xi3 *
              (a_const::f1a13 + 2 * a_const::f2a13 * sind +
               3 * a_const::f3a13 * std::pow(sind, 2) +
               4 * a_const::f4a13 * std::pow(sind, 3)) +
          xi1 * std::pow(xi4, 5) *
              (a_const::f1a144444 + 2 * a_const::f2a144444 * sind) +
          xi1 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a144445 + 2 * a_const::f2a144445 * sind) +
          xi1 * std::pow(xi4, 4) *
              (a_const::f1a14444 + 2 * a_const::f2a14444 * sind) +
          xi1 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a144455 + 2 * a_const::f2a144455 * sind) +
          xi1 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a14445 + 2 * a_const::f2a14445 * sind) +
          xi1 * std::pow(xi4, 3) *
              (a_const::f1a1444 + 2 * a_const::f2a1444 * sind) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a144555 + 2 * a_const::f2a144555 * sind) +
          xi1 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a14455 + 2 * a_const::f2a14455 * sind) +
          xi1 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a1445 + 2 * a_const::f2a1445 * sind) +
          xi1 * std::pow(xi4, 2) *
              (a_const::f1a144 + 2 * a_const::f2a144 * sind +
               3 * a_const::f3a144 * std::pow(sind, 2)) +
          xi1 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a145555 + 2 * a_const::f2a145555 * sind) +
          xi1 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a14555 + 2 * a_const::f2a14555 * sind) +
          xi1 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a1455 + 2 * a_const::f2a1455 * sind) +
          xi1 * xi4 * xi5 *
              (a_const::f1a145 + 2 * a_const::f2a145 * sind +
               3 * a_const::f3a145 * std::pow(sind, 2)) +
          xi1 * xi4 *
              (a_const::f1a14 + 2 * a_const::f2a14 * sind +
               3 * a_const::f3a14 * std::pow(sind, 2) +
               4 * a_const::f4a14 * std::pow(sind, 3)) +
          xi1 * std::pow(xi5, 5) *
              (a_const::f1a155555 + 2 * a_const::f2a155555 * sind) +
          xi1 * std::pow(xi5, 4) *
              (a_const::f1a15555 + 2 * a_const::f2a15555 * sind) +
          xi1 * std::pow(xi5, 3) *
              (a_const::f1a1555 + 2 * a_const::f2a1555 * sind) +
          xi1 * std::pow(xi5, 2) *
              (a_const::f1a155 + 2 * a_const::f2a155 * sind +
               3 * a_const::f3a155 * std::pow(sind, 2)) +
          xi1 * xi5 *
              (a_const::f1a15 + 2 * a_const::f2a15 * sind +
               3 * a_const::f3a15 * std::pow(sind, 2) +
               4 * a_const::f4a15 * std::pow(sind, 3)) +
          xi1 * (a_const::f1a1 + 2 * a_const::f2a1 * sind +
                 3 * a_const::f3a1 * std::pow(sind, 2) +
                 4 * a_const::f4a1 * std::pow(sind, 3) +
                 5 * a_const::f5a1 * std::pow(sind, 4) +
                 6 * a_const::f6a1 * std::pow(sind, 5)) +
          std::pow(xi2, 6) *
              (a_const::f1a222222 + 2 * a_const::f2a222222 * sind) +
          std::pow(xi2, 5) * xi3 *
              (a_const::f1a222223 + 2 * a_const::f2a222223 * sind) +
          std::pow(xi2, 5) * xi4 *
              (a_const::f1a222224 + 2 * a_const::f2a222224 * sind) +
          std::pow(xi2, 5) * xi5 *
              (a_const::f1a222225 + 2 * a_const::f2a222225 * sind) +
          std::pow(xi2, 5) *
              (a_const::f1a22222 + 2 * a_const::f2a22222 * sind) +
          std::pow(xi2, 4) * std::pow(xi3, 2) *
              (a_const::f1a222233 + 2 * a_const::f2a222233 * sind) +
          std::pow(xi2, 4) * xi3 * xi4 *
              (a_const::f1a222234 + 2 * a_const::f2a222234 * sind) +
          std::pow(xi2, 4) * xi3 * xi5 *
              (a_const::f1a222235 + 2 * a_const::f2a222235 * sind) +
          std::pow(xi2, 4) * xi3 *
              (a_const::f1a22223 + 2 * a_const::f2a22223 * sind) +
          std::pow(xi2, 4) * std::pow(xi4, 2) *
              (a_const::f1a222244 + 2 * a_const::f2a222244 * sind) +
          std::pow(xi2, 4) * xi4 * xi5 *
              (a_const::f1a222245 + 2 * a_const::f2a222245 * sind) +
          std::pow(xi2, 4) * xi4 *
              (a_const::f1a22224 + 2 * a_const::f2a22224 * sind) +
          std::pow(xi2, 4) * std::pow(xi5, 2) *
              (a_const::f1a222255 + 2 * a_const::f2a222255 * sind) +
          std::pow(xi2, 4) * xi5 *
              (a_const::f1a22225 + 2 * a_const::f2a22225 * sind) +
          std::pow(xi2, 4) * (a_const::f1a2222 + 2 * a_const::f2a2222 * sind) +
          std::pow(xi2, 3) * std::pow(xi3, 3) *
              (a_const::f1a222333 + 2 * a_const::f2a222333 * sind) +
          std::pow(xi2, 3) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a222334 + 2 * a_const::f2a222334 * sind) +
          std::pow(xi2, 3) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a222335 + 2 * a_const::f2a222335 * sind) +
          std::pow(xi2, 3) * std::pow(xi3, 2) *
              (a_const::f1a22233 + 2 * a_const::f2a22233 * sind) +
          std::pow(xi2, 3) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a222344 + 2 * a_const::f2a222344 * sind) +
          std::pow(xi2, 3) * xi3 * xi4 * xi5 *
              (a_const::f1a222345 + 2 * a_const::f2a222345 * sind) +
          std::pow(xi2, 3) * xi3 * xi4 *
              (a_const::f1a22234 + 2 * a_const::f2a22234 * sind) +
          std::pow(xi2, 3) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a222355 + 2 * a_const::f2a222355 * sind) +
          std::pow(xi2, 3) * xi3 * xi5 *
              (a_const::f1a22235 + 2 * a_const::f2a22235 * sind) +
          std::pow(xi2, 3) * xi3 *
              (a_const::f1a2223 + 2 * a_const::f2a2223 * sind) +
          std::pow(xi2, 3) * std::pow(xi4, 3) *
              (a_const::f1a222444 + 2 * a_const::f2a222444 * sind) +
          std::pow(xi2, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a222445 + 2 * a_const::f2a222445 * sind) +
          std::pow(xi2, 3) * std::pow(xi4, 2) *
              (a_const::f1a22244 + 2 * a_const::f2a22244 * sind) +
          std::pow(xi2, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a222455 + 2 * a_const::f2a222455 * sind) +
          std::pow(xi2, 3) * xi4 * xi5 *
              (a_const::f1a22245 + 2 * a_const::f2a22245 * sind) +
          std::pow(xi2, 3) * xi4 *
              (a_const::f1a2224 + 2 * a_const::f2a2224 * sind) +
          std::pow(xi2, 3) * std::pow(xi5, 3) *
              (a_const::f1a222555 + 2 * a_const::f2a222555 * sind) +
          std::pow(xi2, 3) * std::pow(xi5, 2) *
              (a_const::f1a22255 + 2 * a_const::f2a22255 * sind) +
          std::pow(xi2, 3) * xi5 *
              (a_const::f1a2225 + 2 * a_const::f2a2225 * sind) +
          std::pow(xi2, 3) * (a_const::f1a222 + 2 * a_const::f2a222 * sind +
                              3 * a_const::f3a222 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi3, 4) *
              (a_const::f1a223333 + 2 * a_const::f2a223333 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 3) * xi4 *
              (a_const::f1a223334 + 2 * a_const::f2a223334 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 3) * xi5 *
              (a_const::f1a223335 + 2 * a_const::f2a223335 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 3) *
              (a_const::f1a22333 + 2 * a_const::f2a22333 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a223344 + 2 * a_const::f2a223344 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a223345 + 2 * a_const::f2a223345 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi4 *
              (a_const::f1a22334 + 2 * a_const::f2a22334 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a223355 + 2 * a_const::f2a223355 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) * xi5 *
              (a_const::f1a22335 + 2 * a_const::f2a22335 * sind) +
          std::pow(xi2, 2) * std::pow(xi3, 2) *
              (a_const::f1a2233 + 2 * a_const::f2a2233 * sind) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 3) *
              (a_const::f1a223444 + 2 * a_const::f2a223444 * sind) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a223445 + 2 * a_const::f2a223445 * sind) +
          std::pow(xi2, 2) * xi3 * std::pow(xi4, 2) *
              (a_const::f1a22344 + 2 * a_const::f2a22344 * sind) +
          std::pow(xi2, 2) * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a223455 + 2 * a_const::f2a223455 * sind) +
          std::pow(xi2, 2) * xi3 * xi4 * xi5 *
              (a_const::f1a22345 + 2 * a_const::f2a22345 * sind) +
          std::pow(xi2, 2) * xi3 * xi4 *
              (a_const::f1a2234 + 2 * a_const::f2a2234 * sind) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 3) *
              (a_const::f1a223555 + 2 * a_const::f2a223555 * sind) +
          std::pow(xi2, 2) * xi3 * std::pow(xi5, 2) *
              (a_const::f1a22355 + 2 * a_const::f2a22355 * sind) +
          std::pow(xi2, 2) * xi3 * xi5 *
              (a_const::f1a2235 + 2 * a_const::f2a2235 * sind) +
          std::pow(xi2, 2) * xi3 *
              (a_const::f1a223 + 2 * a_const::f2a223 * sind +
               3 * a_const::f3a223 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi4, 4) *
              (a_const::f1a224444 + 2 * a_const::f2a224444 * sind) +
          std::pow(xi2, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a224445 + 2 * a_const::f2a224445 * sind) +
          std::pow(xi2, 2) * std::pow(xi4, 3) *
              (a_const::f1a22444 + 2 * a_const::f2a22444 * sind) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a224455 + 2 * a_const::f2a224455 * sind) +
          std::pow(xi2, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a22445 + 2 * a_const::f2a22445 * sind) +
          std::pow(xi2, 2) * std::pow(xi4, 2) *
              (a_const::f1a2244 + 2 * a_const::f2a2244 * sind) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a224555 + 2 * a_const::f2a224555 * sind) +
          std::pow(xi2, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a22455 + 2 * a_const::f2a22455 * sind) +
          std::pow(xi2, 2) * xi4 * xi5 *
              (a_const::f1a2245 + 2 * a_const::f2a2245 * sind) +
          std::pow(xi2, 2) * xi4 *
              (a_const::f1a224 + 2 * a_const::f2a224 * sind +
               3 * a_const::f3a224 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * std::pow(xi5, 4) *
              (a_const::f1a225555 + 2 * a_const::f2a225555 * sind) +
          std::pow(xi2, 2) * std::pow(xi5, 3) *
              (a_const::f1a22555 + 2 * a_const::f2a22555 * sind) +
          std::pow(xi2, 2) * std::pow(xi5, 2) *
              (a_const::f1a2255 + 2 * a_const::f2a2255 * sind) +
          std::pow(xi2, 2) * xi5 *
              (a_const::f1a225 + 2 * a_const::f2a225 * sind +
               3 * a_const::f3a225 * std::pow(sind, 2)) +
          std::pow(xi2, 2) * (a_const::f1a22 + 2 * a_const::f2a22 * sind +
                              3 * a_const::f3a22 * std::pow(sind, 2) +
                              4 * a_const::f4a22 * std::pow(sind, 3)) +
          xi2 * std::pow(xi3, 5) *
              (a_const::f1a233333 + 2 * a_const::f2a233333 * sind) +
          xi2 * std::pow(xi3, 4) * xi4 *
              (a_const::f1a233334 + 2 * a_const::f2a233334 * sind) +
          xi2 * std::pow(xi3, 4) * xi5 *
              (a_const::f1a233335 + 2 * a_const::f2a233335 * sind) +
          xi2 * std::pow(xi3, 4) *
              (a_const::f1a23333 + 2 * a_const::f2a23333 * sind) +
          xi2 * std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a233344 + 2 * a_const::f2a233344 * sind) +
          xi2 * std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a233345 + 2 * a_const::f2a233345 * sind) +
          xi2 * std::pow(xi3, 3) * xi4 *
              (a_const::f1a23334 + 2 * a_const::f2a23334 * sind) +
          xi2 * std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a233355 + 2 * a_const::f2a233355 * sind) +
          xi2 * std::pow(xi3, 3) * xi5 *
              (a_const::f1a23335 + 2 * a_const::f2a23335 * sind) +
          xi2 * std::pow(xi3, 3) *
              (a_const::f1a2333 + 2 * a_const::f2a2333 * sind) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a233444 + 2 * a_const::f2a233444 * sind) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a233445 + 2 * a_const::f2a233445 * sind) +
          xi2 * std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a23344 + 2 * a_const::f2a23344 * sind) +
          xi2 * std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a233455 + 2 * a_const::f2a233455 * sind) +
          xi2 * std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a23345 + 2 * a_const::f2a23345 * sind) +
          xi2 * std::pow(xi3, 2) * xi4 *
              (a_const::f1a2334 + 2 * a_const::f2a2334 * sind) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a233555 + 2 * a_const::f2a233555 * sind) +
          xi2 * std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a23355 + 2 * a_const::f2a23355 * sind) +
          xi2 * std::pow(xi3, 2) * xi5 *
              (a_const::f1a2335 + 2 * a_const::f2a2335 * sind) +
          xi2 * std::pow(xi3, 2) *
              (a_const::f1a233 + 2 * a_const::f2a233 * sind +
               3 * a_const::f3a233 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi4, 4) *
              (a_const::f1a234444 + 2 * a_const::f2a234444 * sind) +
          xi2 * xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a234445 + 2 * a_const::f2a234445 * sind) +
          xi2 * xi3 * std::pow(xi4, 3) *
              (a_const::f1a23444 + 2 * a_const::f2a23444 * sind) +
          xi2 * xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a234455 + 2 * a_const::f2a234455 * sind) +
          xi2 * xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a23445 + 2 * a_const::f2a23445 * sind) +
          xi2 * xi3 * std::pow(xi4, 2) *
              (a_const::f1a2344 + 2 * a_const::f2a2344 * sind) +
          xi2 * xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a234555 + 2 * a_const::f2a234555 * sind) +
          xi2 * xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a23455 + 2 * a_const::f2a23455 * sind) +
          xi2 * xi3 * xi4 * xi5 *
              (a_const::f1a2345 + 2 * a_const::f2a2345 * sind) +
          xi2 * xi3 * xi4 *
              (a_const::f1a234 + 2 * a_const::f2a234 * sind +
               3 * a_const::f3a234 * std::pow(sind, 2)) +
          xi2 * xi3 * std::pow(xi5, 4) *
              (a_const::f1a235555 + 2 * a_const::f2a235555 * sind) +
          xi2 * xi3 * std::pow(xi5, 3) *
              (a_const::f1a23555 + 2 * a_const::f2a23555 * sind) +
          xi2 * xi3 * std::pow(xi5, 2) *
              (a_const::f1a2355 + 2 * a_const::f2a2355 * sind) +
          xi2 * xi3 * xi5 *
              (a_const::f1a235 + 2 * a_const::f2a235 * sind +
               3 * a_const::f3a235 * std::pow(sind, 2)) +
          xi2 * xi3 *
              (a_const::f1a23 + 2 * a_const::f2a23 * sind +
               3 * a_const::f3a23 * std::pow(sind, 2) +
               4 * a_const::f4a23 * std::pow(sind, 3)) +
          xi2 * std::pow(xi4, 5) *
              (a_const::f1a244444 + 2 * a_const::f2a244444 * sind) +
          xi2 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a244445 + 2 * a_const::f2a244445 * sind) +
          xi2 * std::pow(xi4, 4) *
              (a_const::f1a24444 + 2 * a_const::f2a24444 * sind) +
          xi2 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a244455 + 2 * a_const::f2a244455 * sind) +
          xi2 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a24445 + 2 * a_const::f2a24445 * sind) +
          xi2 * std::pow(xi4, 3) *
              (a_const::f1a2444 + 2 * a_const::f2a2444 * sind) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a244555 + 2 * a_const::f2a244555 * sind) +
          xi2 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a24455 + 2 * a_const::f2a24455 * sind) +
          xi2 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a2445 + 2 * a_const::f2a2445 * sind) +
          xi2 * std::pow(xi4, 2) *
              (a_const::f1a244 + 2 * a_const::f2a244 * sind +
               3 * a_const::f3a244 * std::pow(sind, 2)) +
          xi2 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a245555 + 2 * a_const::f2a245555 * sind) +
          xi2 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a24555 + 2 * a_const::f2a24555 * sind) +
          xi2 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a2455 + 2 * a_const::f2a2455 * sind) +
          xi2 * xi4 * xi5 *
              (a_const::f1a245 + 2 * a_const::f2a245 * sind +
               3 * a_const::f3a245 * std::pow(sind, 2)) +
          xi2 * xi4 *
              (a_const::f1a24 + 2 * a_const::f2a24 * sind +
               3 * a_const::f3a24 * std::pow(sind, 2) +
               4 * a_const::f4a24 * std::pow(sind, 3)) +
          xi2 * std::pow(xi5, 5) *
              (a_const::f1a255555 + 2 * a_const::f2a255555 * sind) +
          xi2 * std::pow(xi5, 4) *
              (a_const::f1a25555 + 2 * a_const::f2a25555 * sind) +
          xi2 * std::pow(xi5, 3) *
              (a_const::f1a2555 + 2 * a_const::f2a2555 * sind) +
          xi2 * std::pow(xi5, 2) *
              (a_const::f1a255 + 2 * a_const::f2a255 * sind +
               3 * a_const::f3a255 * std::pow(sind, 2)) +
          xi2 * xi5 *
              (a_const::f1a25 + 2 * a_const::f2a25 * sind +
               3 * a_const::f3a25 * std::pow(sind, 2) +
               4 * a_const::f4a25 * std::pow(sind, 3)) +
          xi2 * (a_const::f1a2 + 2 * a_const::f2a2 * sind +
                 3 * a_const::f3a2 * std::pow(sind, 2) +
                 4 * a_const::f4a2 * std::pow(sind, 3) +
                 5 * a_const::f5a2 * std::pow(sind, 4) +
                 6 * a_const::f6a2 * std::pow(sind, 5)) +
          std::pow(xi3, 6) *
              (a_const::f1a333333 + 2 * a_const::f2a333333 * sind) +
          std::pow(xi3, 5) * xi4 *
              (a_const::f1a333334 + 2 * a_const::f2a333334 * sind) +
          std::pow(xi3, 5) * xi5 *
              (a_const::f1a333335 + 2 * a_const::f2a333335 * sind) +
          std::pow(xi3, 5) *
              (a_const::f1a33333 + 2 * a_const::f2a33333 * sind) +
          std::pow(xi3, 4) * std::pow(xi4, 2) *
              (a_const::f1a333344 + 2 * a_const::f2a333344 * sind) +
          std::pow(xi3, 4) * xi4 * xi5 *
              (a_const::f1a333345 + 2 * a_const::f2a333345 * sind) +
          std::pow(xi3, 4) * xi4 *
              (a_const::f1a33334 + 2 * a_const::f2a33334 * sind) +
          std::pow(xi3, 4) * std::pow(xi5, 2) *
              (a_const::f1a333355 + 2 * a_const::f2a333355 * sind) +
          std::pow(xi3, 4) * xi5 *
              (a_const::f1a33335 + 2 * a_const::f2a33335 * sind) +
          std::pow(xi3, 4) * (a_const::f1a3333 + 2 * a_const::f2a3333 * sind) +
          std::pow(xi3, 3) * std::pow(xi4, 3) *
              (a_const::f1a333444 + 2 * a_const::f2a333444 * sind) +
          std::pow(xi3, 3) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a333445 + 2 * a_const::f2a333445 * sind) +
          std::pow(xi3, 3) * std::pow(xi4, 2) *
              (a_const::f1a33344 + 2 * a_const::f2a33344 * sind) +
          std::pow(xi3, 3) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a333455 + 2 * a_const::f2a333455 * sind) +
          std::pow(xi3, 3) * xi4 * xi5 *
              (a_const::f1a33345 + 2 * a_const::f2a33345 * sind) +
          std::pow(xi3, 3) * xi4 *
              (a_const::f1a3334 + 2 * a_const::f2a3334 * sind) +
          std::pow(xi3, 3) * std::pow(xi5, 3) *
              (a_const::f1a333555 + 2 * a_const::f2a333555 * sind) +
          std::pow(xi3, 3) * std::pow(xi5, 2) *
              (a_const::f1a33355 + 2 * a_const::f2a33355 * sind) +
          std::pow(xi3, 3) * xi5 *
              (a_const::f1a3335 + 2 * a_const::f2a3335 * sind) +
          std::pow(xi3, 3) * (a_const::f1a333 + 2 * a_const::f2a333 * sind +
                              3 * a_const::f3a333 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi4, 4) *
              (a_const::f1a334444 + 2 * a_const::f2a334444 * sind) +
          std::pow(xi3, 2) * std::pow(xi4, 3) * xi5 *
              (a_const::f1a334445 + 2 * a_const::f2a334445 * sind) +
          std::pow(xi3, 2) * std::pow(xi4, 3) *
              (a_const::f1a33444 + 2 * a_const::f2a33444 * sind) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a334455 + 2 * a_const::f2a334455 * sind) +
          std::pow(xi3, 2) * std::pow(xi4, 2) * xi5 *
              (a_const::f1a33445 + 2 * a_const::f2a33445 * sind) +
          std::pow(xi3, 2) * std::pow(xi4, 2) *
              (a_const::f1a3344 + 2 * a_const::f2a3344 * sind) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 3) *
              (a_const::f1a334555 + 2 * a_const::f2a334555 * sind) +
          std::pow(xi3, 2) * xi4 * std::pow(xi5, 2) *
              (a_const::f1a33455 + 2 * a_const::f2a33455 * sind) +
          std::pow(xi3, 2) * xi4 * xi5 *
              (a_const::f1a3345 + 2 * a_const::f2a3345 * sind) +
          std::pow(xi3, 2) * xi4 *
              (a_const::f1a334 + 2 * a_const::f2a334 * sind +
               3 * a_const::f3a334 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * std::pow(xi5, 4) *
              (a_const::f1a335555 + 2 * a_const::f2a335555 * sind) +
          std::pow(xi3, 2) * std::pow(xi5, 3) *
              (a_const::f1a33555 + 2 * a_const::f2a33555 * sind) +
          std::pow(xi3, 2) * std::pow(xi5, 2) *
              (a_const::f1a3355 + 2 * a_const::f2a3355 * sind) +
          std::pow(xi3, 2) * xi5 *
              (a_const::f1a335 + 2 * a_const::f2a335 * sind +
               3 * a_const::f3a335 * std::pow(sind, 2)) +
          std::pow(xi3, 2) * (a_const::f1a33 + 2 * a_const::f2a33 * sind +
                              3 * a_const::f3a33 * std::pow(sind, 2) +
                              4 * a_const::f4a33 * std::pow(sind, 3)) +
          xi3 * std::pow(xi4, 5) *
              (a_const::f1a344444 + 2 * a_const::f2a344444 * sind) +
          xi3 * std::pow(xi4, 4) * xi5 *
              (a_const::f1a344445 + 2 * a_const::f2a344445 * sind) +
          xi3 * std::pow(xi4, 4) *
              (a_const::f1a34444 + 2 * a_const::f2a34444 * sind) +
          xi3 * std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a344455 + 2 * a_const::f2a344455 * sind) +
          xi3 * std::pow(xi4, 3) * xi5 *
              (a_const::f1a34445 + 2 * a_const::f2a34445 * sind) +
          xi3 * std::pow(xi4, 3) *
              (a_const::f1a3444 + 2 * a_const::f2a3444 * sind) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a344555 + 2 * a_const::f2a344555 * sind) +
          xi3 * std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a34455 + 2 * a_const::f2a34455 * sind) +
          xi3 * std::pow(xi4, 2) * xi5 *
              (a_const::f1a3445 + 2 * a_const::f2a3445 * sind) +
          xi3 * std::pow(xi4, 2) *
              (a_const::f1a344 + 2 * a_const::f2a344 * sind +
               3 * a_const::f3a344 * std::pow(sind, 2)) +
          xi3 * xi4 * std::pow(xi5, 4) *
              (a_const::f1a345555 + 2 * a_const::f2a345555 * sind) +
          xi3 * xi4 * std::pow(xi5, 3) *
              (a_const::f1a34555 + 2 * a_const::f2a34555 * sind) +
          xi3 * xi4 * std::pow(xi5, 2) *
              (a_const::f1a3455 + 2 * a_const::f2a3455 * sind) +
          xi3 * xi4 * xi5 *
              (a_const::f1a345 + 2 * a_const::f2a345 * sind +
               3 * a_const::f3a345 * std::pow(sind, 2)) +
          xi3 * xi4 *
              (a_const::f1a34 + 2 * a_const::f2a34 * sind +
               3 * a_const::f3a34 * std::pow(sind, 2) +
               4 * a_const::f4a34 * std::pow(sind, 3)) +
          xi3 * std::pow(xi5, 5) *
              (a_const::f1a355555 + 2 * a_const::f2a355555 * sind) +
          xi3 * std::pow(xi5, 4) *
              (a_const::f1a35555 + 2 * a_const::f2a35555 * sind) +
          xi3 * std::pow(xi5, 3) *
              (a_const::f1a3555 + 2 * a_const::f2a3555 * sind) +
          xi3 * std::pow(xi5, 2) *
              (a_const::f1a355 + 2 * a_const::f2a355 * sind +
               3 * a_const::f3a355 * std::pow(sind, 2)) +
          xi3 * xi5 *
              (a_const::f1a35 + 2 * a_const::f2a35 * sind +
               3 * a_const::f3a35 * std::pow(sind, 2) +
               4 * a_const::f4a35 * std::pow(sind, 3)) +
          xi3 * (a_const::f1a3 + 2 * a_const::f2a3 * sind +
                 3 * a_const::f3a3 * std::pow(sind, 2) +
                 4 * a_const::f4a3 * std::pow(sind, 3) +
                 5 * a_const::f5a3 * std::pow(sind, 4) +
                 6 * a_const::f6a3 * std::pow(sind, 5)) +
          std::pow(xi4, 6) *
              (a_const::f1a444444 + 2 * a_const::f2a444444 * sind) +
          std::pow(xi4, 5) * xi5 *
              (a_const::f1a444445 + 2 * a_const::f2a444445 * sind) +
          std::pow(xi4, 5) *
              (a_const::f1a44444 + 2 * a_const::f2a44444 * sind) +
          std::pow(xi4, 4) * std::pow(xi5, 2) *
              (a_const::f1a444455 + 2 * a_const::f2a444455 * sind) +
          std::pow(xi4, 4) * xi5 *
              (a_const::f1a44445 + 2 * a_const::f2a44445 * sind) +
          std::pow(xi4, 4) * (a_const::f1a4444 + 2 * a_const::f2a4444 * sind) +
          std::pow(xi4, 3) * std::pow(xi5, 3) *
              (a_const::f1a444555 + 2 * a_const::f2a444555 * sind) +
          std::pow(xi4, 3) * std::pow(xi5, 2) *
              (a_const::f1a44455 + 2 * a_const::f2a44455 * sind) +
          std::pow(xi4, 3) * xi5 *
              (a_const::f1a4445 + 2 * a_const::f2a4445 * sind) +
          std::pow(xi4, 3) * (a_const::f1a444 + 2 * a_const::f2a444 * sind +
                              3 * a_const::f3a444 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * std::pow(xi5, 4) *
              (a_const::f1a445555 + 2 * a_const::f2a445555 * sind) +
          std::pow(xi4, 2) * std::pow(xi5, 3) *
              (a_const::f1a44555 + 2 * a_const::f2a44555 * sind) +
          std::pow(xi4, 2) * std::pow(xi5, 2) *
              (a_const::f1a4455 + 2 * a_const::f2a4455 * sind) +
          std::pow(xi4, 2) * xi5 *
              (a_const::f1a445 + 2 * a_const::f2a445 * sind +
               3 * a_const::f3a445 * std::pow(sind, 2)) +
          std::pow(xi4, 2) * (a_const::f1a44 + 2 * a_const::f2a44 * sind +
                              3 * a_const::f3a44 * std::pow(sind, 2) +
                              4 * a_const::f4a44 * std::pow(sind, 3)) +
          xi4 * std::pow(xi5, 5) *
              (a_const::f1a455555 + 2 * a_const::f2a455555 * sind) +
          xi4 * std::pow(xi5, 4) *
              (a_const::f1a45555 + 2 * a_const::f2a45555 * sind) +
          xi4 * std::pow(xi5, 3) *
              (a_const::f1a4555 + 2 * a_const::f2a4555 * sind) +
          xi4 * std::pow(xi5, 2) *
              (a_const::f1a455 + 2 * a_const::f2a455 * sind +
               3 * a_const::f3a455 * std::pow(sind, 2)) +
          xi4 * xi5 *
              (a_const::f1a45 + 2 * a_const::f2a45 * sind +
               3 * a_const::f3a45 * std::pow(sind, 2) +
               4 * a_const::f4a45 * std::pow(sind, 3)) +
          xi4 * (a_const::f1a4 + 2 * a_const::f2a4 * sind +
                 3 * a_const::f3a4 * std::pow(sind, 2) +
                 4 * a_const::f4a4 * std::pow(sind, 3) +
                 5 * a_const::f5a4 * std::pow(sind, 4) +
                 6 * a_const::f6a4 * std::pow(sind, 5)) +
          std::pow(xi5, 6) *
              (a_const::f1a555555 + 2 * a_const::f2a555555 * sind) +
          std::pow(xi5, 5) *
              (a_const::f1a55555 + 2 * a_const::f2a55555 * sind) +
          std::pow(xi5, 4) * (a_const::f1a5555 + 2 * a_const::f2a5555 * sind) +
          std::pow(xi5, 3) * (a_const::f1a555 + 2 * a_const::f2a555 * sind +
                              3 * a_const::f3a555 * std::pow(sind, 2)) +
          std::pow(xi5, 2) * (a_const::f1a55 + 2 * a_const::f2a55 * sind +
                              3 * a_const::f3a55 * std::pow(sind, 2) +
                              4 * a_const::f4a55 * std::pow(sind, 3)) +
          xi5 * (a_const::f1a5 + 2 * a_const::f2a5 * sind +
                 3 * a_const::f3a5 * std::pow(sind, 2) +
                 4 * a_const::f4a5 * std::pow(sind, 3) +
                 5 * a_const::f5a5 * std::pow(sind, 4) +
                 6 * a_const::f6a5 * std::pow(sind, 5));
      f_xn = -dsinddxn * dvdsind - dvdxi1 * dxi1dxn - dvdxi2 * dxi2dxn -
             dvdxi3 * dxi3dxn - dvdxi4 * dxi4dxn - dvdxi5 * dxi5dxn;
      f_yn = -dsinddyn * dvdsind - dvdxi1 * dxi1dyn - dvdxi2 * dxi2dyn -
             dvdxi3 * dxi3dyn - dvdxi4 * dxi4dyn - dvdxi5 * dxi5dyn;
      f_zn = -dsinddzn * dvdsind - dvdxi1 * dxi1dzn - dvdxi2 * dxi2dzn -
             dvdxi3 * dxi3dzn - dvdxi4 * dxi4dzn - dvdxi5 * dxi5dzn;
      f_xh1 = -dsinddxh1 * dvdsind - dvdxi1 * dxi1dxh1 - dvdxi2 * dxi2dxh1 -
              dvdxi3 * dxi3dxh1 - dvdxi4 * dxi4dxh1 - dvdxi5 * dxi5dxh1;
      f_yh1 = -dsinddyh1 * dvdsind - dvdxi1 * dxi1dyh1 - dvdxi2 * dxi2dyh1 -
              dvdxi3 * dxi3dyh1 - dvdxi4 * dxi4dyh1 - dvdxi5 * dxi5dyh1;
      f_zh1 = -dsinddzh1 * dvdsind - dvdxi1 * dxi1dzh1 - dvdxi2 * dxi2dzh1 -
              dvdxi3 * dxi3dzh1 - dvdxi4 * dxi4dzh1 - dvdxi5 * dxi5dzh1;
      f_xh2 = -dsinddxh2 * dvdsind - dvdxi1 * dxi1dxh2 - dvdxi2 * dxi2dxh2 -
              dvdxi3 * dxi3dxh2 - dvdxi4 * dxi4dxh2 - dvdxi5 * dxi5dxh2;
      f_yh2 = -dsinddyh2 * dvdsind - dvdxi1 * dxi1dyh2 - dvdxi2 * dxi2dyh2 -
              dvdxi3 * dxi3dyh2 - dvdxi4 * dxi4dyh2 - dvdxi5 * dxi5dyh2;
      f_zh2 = -dsinddzh2 * dvdsind - dvdxi1 * dxi1dzh2 - dvdxi2 * dxi2dzh2 -
              dvdxi3 * dxi3dzh2 - dvdxi4 * dxi4dzh2 - dvdxi5 * dxi5dzh2;
      f_xh3 = -dsinddxh3 * dvdsind - dvdxi1 * dxi1dxh3 - dvdxi2 * dxi2dxh3 -
              dvdxi3 * dxi3dxh3 - dvdxi4 * dxi4dxh3 - dvdxi5 * dxi5dxh3;
      f_yh3 = -dsinddyh3 * dvdsind - dvdxi1 * dxi1dyh3 - dvdxi2 * dxi2dyh3 -
              dvdxi3 * dxi3dyh3 - dvdxi4 * dxi4dyh3 - dvdxi5 * dxi5dyh3;
      f_zh3 = -dsinddzh3 * dvdsind - dvdxi1 * dxi1dzh3 - dvdxi2 * dxi2dzh3 -
              dvdxi3 * dxi3dzh3 - dvdxi4 * dxi4dzh3 - dvdxi5 * dxi5dzh3;
      f(j, 0, i) = f_xh1;
      f(j, 1, i) = f_yh1;
      f(j, 2, i) = f_zh1;
      f(j, 0, i + 1) = f_xh2;
      f(j, 1, i + 1) = f_yh2;
      f(j, 2, i + 1) = f_zh2;
      f(j, 0, i + 2) = f_xh3;
      f(j, 1, i + 2) = f_yh3;
      f(j, 2, i + 2) = f_zh3;
      f(j, 0, i + 3) = f_xn;
      f(j, 1, i + 3) = f_yn;
      f(j, 2, i + 3) = f_zn;
    }
  }
  assert(arma::is_finite(f));
  return f;
}
