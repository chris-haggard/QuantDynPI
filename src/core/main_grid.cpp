#include <mpi.h>
#include <bspline.h>
#include <bsplinebuilder.h>
#include <datatable.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../util/IO_utils.hpp"
#include "../util/RingPolymerUtils.hpp"
#include "../util/rand_utils.hpp"
#include "../util/unit_conversion.hpp"
#include "Input.hpp"
#include "Simulation.hpp"
#include "recorder/TCF.hpp"
#include "recorder/TrajectoryRecorder.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  int rank, nproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  JsonInputLoader Paramters;
  std::string input_file = argv[1];
  Paramters.load(input_file);
  InputHandler ParameterHandler(Paramters.tree);

  MPI_Barrier(MPI_COMM_WORLD);
  auto original_seed = ParameterHandler.ThermostatParameters.seed;
  cout << "original_seed = " << original_seed << endl;
  std::vector<double> seed_vector(nproc);
  if (rank == 0) {
    seed_vector = SeedVector(ParameterHandler.ThermostatParameters.seed, nproc);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Scatter(
      &seed_vector[0], 1, MPI_DOUBLE,
      &ParameterHandler.ThermostatParameters.seed, 1, MPI_DOUBLE, 0,
      MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  arma::arma_rng::set_seed(ParameterHandler.ThermostatParameters.seed);

  Ensemble Ens(ParameterHandler.EnsembleParameters);
  Simulation Sim(ParameterHandler, Ens);
  if (rank == 0) {
    cout << "Max freq = " << Sim.RP->freq_scaled.max() << "\n";
    cout << "Timestep required = " << (1.0 / Sim.RP->freq_scaled.max()) / 20.0
         << " au \n";
    if (Sim.Dyn->dt >= (1.0 / Sim.RP->freq_scaled.max()) / 20.0) {
      cout << "\n *** WARNING: dt = " << Sim.Dyn->dt
           << " au greater than max allowed  = "
           << (1.0 / Sim.RP->freq_scaled.max()) / 20.0 << " au ***\n";
    }
  }

  //******************** Spline setup *******************************
  SPLINTER::DataTable samples;
  arma::vec r_grid = arma::linspace(1.5, 2.5, 32);
  arma::vec theta_grid = arma::linspace(85, 130, 32) * arma::datum::pi / 180.0;
  SPLINTER::DataTable R1_data, R2_data, Theta_data;
  std::vector<double> temp_spline_data_storage(3);
  arma::field<arma::cube> force_storage(3);  // f_R1,f_R2,f_Th
  force_storage(0) = arma::cube(
      r_grid.n_rows, r_grid.n_rows, theta_grid.n_rows,
      arma::fill::zeros);  // cube with dim r1,r2,th
  force_storage(1) = force_storage(0);
  force_storage(2) = force_storage(0);
  //***************** End Spline set up ****************************

  const unsigned int therm_steps_grid = 100;
  const unsigned int run_steps_grid = 500;

  for (arma::uword r1_idx = 0; r1_idx < r_grid.size(); r1_idx++) {
    for (arma::uword r2_idx = 0; r2_idx < r_grid.size(); r2_idx++) {
      for (arma::uword th_idx = 0; th_idx < theta_grid.size(); th_idx++) {
        WaterGeometry(
            Sim.RP->position_cart, r_grid(r1_idx), r_grid(r2_idx),
            theta_grid(th_idx));
        Sim.RP->NMConv.CartToNM(Sim.RP->position_cart, Sim.RP->position_nm);
        if (ParameterHandler.SimulationParameters.sim_type == "QCMD") {
          Sim.RP->PlaceQuasicentroid();
        }

        for (size_t i = 0; i < therm_steps_grid; i++) {
#ifdef SKIP_THERMAL
          break;
#endif
#ifdef PRINT_STEPS
          cout << "grid therm step = " << i << " / " << therm_steps_grid
               << endl;
#endif
          Sim.Dyn->ThermStep();
        }

        for (size_t i = 0; i < run_steps_grid; i++) {
#ifdef PRINT_STEPS
          cout << "grid run step = " << i << " / " << run_steps_grid << endl;
#endif
          Sim.Dyn->Step();
          for (arma::uword z = 0; z < Sim.RP->n_total_atoms; z += 3) {
            arma::mat OH_1 = BondMat(Sim.RP->position_cart, z, z + 1);
            arma::mat OH_2 = BondMat(Sim.RP->position_cart, z, z + 2);
            arma::vec r_1 = BondLength(OH_1);
            arma::vec r_2 = BondLength(OH_2);
            // f_R_1
            force_storage(0)(r1_idx, r2_idx, th_idx) +=
                arma::as_scalar(arma::mean(
                    arma::sum(OH_1 % Sim.RP->force_cart.slice(z + 1), 1) / r_1,
                    0));
            // f_R_2
            force_storage(1)(r2_idx, r2_idx, th_idx) +=
                arma::as_scalar(arma::mean(
                    arma::sum(OH_2 % Sim.RP->force_cart.slice(z + 2), 1) / r_2,
                    0));
            // f_Theta
            arma::vec cos_theta = CosThetaVec(OH_1, OH_2);

            OH_1.each_col() %= cos_theta;
            OH_1.each_col() /= r_1;
            OH_2.each_col() /= r_2;

            // negative of force is gradient
            force_storage(2)(r1_idx, r2_idx, th_idx) +=
                arma::as_scalar(arma::mean(
                    (r_1 / SinThetaVec(cos_theta)) %
                        arma::sum(
                            (OH_2 - (OH_1)) % -Sim.RP->force_cart.slice(z + 1),
                            1),
                    0));
          }
        }
      }
    }
  }
  force_storage(0) /= (Sim.RP->n_molec * run_steps_grid);
  force_storage(1) /= (Sim.RP->n_molec * run_steps_grid);
  force_storage(2) /= (Sim.RP->n_molec * run_steps_grid);

  MPI_Barrier(MPI_COMM_WORLD);
  // get forces on all procs via Allreduce
  for (arma::uword s = 0; s < force_storage.n_elem; s++) {
    MPI_Allreduce(
        MPI_IN_PLACE, force_storage(s).memptr(), force_storage(0).n_rows,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    force_storage(s) /= nproc;
  }
  // transfer
  for (arma::uword i = 0; i < r_grid.n_rows; i++) {
    for (arma::uword j = 0; j < r_grid.n_rows; j++) {
      for (arma::uword k = 0; k < theta_grid.n_rows; k++) {
        temp_spline_data_storage = {r_grid(i), r_grid(j), theta_grid(k)};
        R1_data.addSample(temp_spline_data_storage, force_storage(0)(i, j, k));
        R2_data.addSample(temp_spline_data_storage, force_storage(1)(i, j, k));
        Theta_data.addSample(
            temp_spline_data_storage, force_storage(2)(i, j, k));
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // make the force splines
  SPLINTER::BSpline R1_spline =
      SPLINTER::BSpline::Builder(R1_data).degree(3).build();
  SPLINTER::BSpline R2_spline =
      SPLINTER::BSpline::Builder(R2_data).degree(3).build();
  SPLINTER::BSpline Theta_spline =
      SPLINTER::BSpline::Builder(Theta_data).degree(3).build();
  if (rank == 0) {
    std::string fname =
        ParameterHandler.SimulationParameters.sim_type + "_spline.exe";
    R1_spline.save("R1_" + fname);
    R2_spline.save("R2_" + fname);
    Theta_spline.save("Theta_" + fname);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
