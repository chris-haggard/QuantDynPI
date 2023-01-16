#include <mpi.h>

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
  // provide a different seed to each proccess
  std::vector<double> seed_vector(nproc);
  if (rank == 0) {
    seed_vector = SeedVector(ParameterHandler.ThermostatParameters.seed, nproc);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Scatter(
      &seed_vector[0], 1, MPI_DOUBLE,
      &ParameterHandler.ThermostatParameters.seed, 1, MPI_DOUBLE, 0,
      MPI_COMM_WORLD);

  arma::arma_rng::set_seed(ParameterHandler.ThermostatParameters.seed);

  Ensemble Ens(ParameterHandler.EnsembleParameters);
  Simulation Sim(ParameterHandler, Ens);
  if (rank == 0 && Sim.RP->freq_scaled.max() != 0) {
    cout << "Max freq = " << Sim.RP->freq_scaled.max() << "\n";
    cout << "Timestep required = " << (1.0 / Sim.RP->freq_scaled.max()) / 20.0
         << " au \n";
    if (Sim.Dyn->dt >= (1.0 / Sim.RP->freq_scaled.max()) / 20.0) {
      cout << "\n *** WARNING: dt = " << Sim.Dyn->dt
           << " au greater than max allowed  = "
           << (1.0 / Sim.RP->freq_scaled.max()) / 20.0 << " au ***\n";
    }
  }

  const unsigned int initial_therm_steps =
      (int)std::round(Sim.initial_therm_length / Sim.Dyn->dt);
  const unsigned int therm_steps =
      (int)std::round(Sim.therm_length / Sim.Dyn->dt);
  const unsigned int run_steps = (int)std::round(Sim.run_length / Sim.Dyn->dt);

  //******************* set up recorder *****************
  TCF RecordTCF(
      Sim.RP->DynamicalVariable(), run_steps, Sim.stride, Sim.RP->n_molec,
      nproc);

  std::string base_filename = BaseFileNamer(
      ParameterHandler.SimulationParameters.sim_type,
      ParameterHandler.PotentialParameters.potential_type,
      ParameterHandler.EnsembleParameters.T,
      ParameterHandler.RingPolymerParameters.n_molec,
      ParameterHandler.RingPolymerParameters.n_beads, original_seed,
      ParameterHandler.RingPolymerParameters.gamma);

  std::string TCF_file = OutputFileNamer(base_filename, "TCF");

#ifdef SAVE_TRAJECTORY
  TrajectoryRecorder Traj_bead;
  TrajectoryRecorder Traj_quasi;
  TrajectoryRecorder Traj_force;
  Traj_bead.trajectory_file = OutputFileNamer(base_filename, "bead_traj");
  Traj_quasi.trajectory_file = OutputFileNamer(base_filename, "quasi_traj");
#endif
#ifdef VIRIAL_ESTIMATOR
  arma::mat virial_estimators(initial_therm_steps, 4);
  std::string Virial_file = OutputFileNamer(base_filename, "Virial");
#endif

  for (size_t i = 0; i < initial_therm_steps; i++) {
#ifdef SKIP_THERMAL
    break;
#endif
#ifdef PRINT_STEPS
    cout << "initial therm step = " << i << " / " << initial_therm_steps
         << endl;
#endif
    Sim.Dyn->ThermStep();
#ifdef VIRIAL_ESTIMATOR
    if (rank == 0) {
      virial_estimators(i, 0) = Sim.Dyn->dt * i;
      virial_estimators(i, 1) =
          Ens.V_estimator(Sim.Dyn->PES->Pot(Sim.RP->position_cart));
      virial_estimators(i, 2) =
          Ens.Virial_estimator(Sim.RP->position_cart, Sim.RP->force_cart);
      virial_estimators(i, 3) =
          virial_estimators(i, 1) + virial_estimators(i, 2);
    }
#endif
  }
#ifdef VIRIAL_ESTIMATOR
  if (rank == 0) {
    WriteFile(virial_estimators, Virial_file);
  }
#endif

  for (size_t j = 0; j < Sim.n_traj; j++) {
    for (size_t i = 0; i < therm_steps; i++) {
#ifdef SKIP_THERMAL
      break;
#endif
#ifdef PRINT_STEPS
      cout << "therm step = " << i << " / " << therm_steps << endl;
#endif
      Sim.Dyn->ThermStep();
    }

    for (size_t i = 0; i < run_steps; i++) {
#ifdef PRINT_STEPS
      cout << "run step = " << i << " / " << run_steps << endl;
#endif
      
      RecordTCF.Record(
          arma::mean(Sim.Dyn->PES->Dipole(Sim.RP->DynamicalVariable()), 0), i);

#ifdef SAVE_TRAJECTORY
      if (rank == 0 && j == 0) {
        Traj_bead.AppendFrame(
            Sim.RP->position_cart, Sim.RP->labels, au_fs(i * Sim.Dyn->dt));
        Traj_quasi.AppendFrame(
            Sim.RP->TrajectoryVariable(), Sim.RP->labels,
            au_fs(i * Sim.Dyn->dt));
      }
#endif

      Sim.Dyn->Step();
    }
    RecordTCF.CorrelateTCFs();
  }
  RecordTCF.NormaliseTCFs(Sim.n_traj);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(
      RecordTCF.tcf.memptr(), RecordTCF.tcf.n_rows, MPI_DOUBLE,
      RecordTCF.out_buffer.colptr(rank), RecordTCF.tcf.n_rows, MPI_DOUBLE, 0,
      MPI_COMM_WORLD);

  if (rank == 0) {
    arma::mat temp(RecordTCF.out_buffer.n_rows, 3);
    temp.col(1) = arma::mean(RecordTCF.out_buffer, 1);
    // std error the mean
    temp.col(2) = arma::stddev(RecordTCF.out_buffer, 0, 1) / std::sqrt(nproc);
    for (arma::uword i = 0; i < RecordTCF.out_buffer.n_rows; i++) {
      temp(i, 0) = Sim.Dyn->dt * i * Sim.stride;
    }
    WriteFile(temp, TCF_file);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
