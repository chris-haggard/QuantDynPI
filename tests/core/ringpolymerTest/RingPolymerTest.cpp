#include <gtest/gtest.h>

#include <memory>

#include "../../../src/core/Ensemble.hpp"
#include "../../../src/core/Input.hpp"
#include "../../../src/core/ringpolymer/RingPolymer.hpp"

class RingPolymerTest : public ::testing::Test {
 protected:
  RingPolymerTest() {
    JsonInputLoader Paramters;
    std::string filename = "test_input_file.json";
    Paramters.load(filename);
    // ParameterHandler holds paramets in nested structs
    InputHandler ParameterHandler(Paramters.tree);
    Ens = std::make_unique<Ensemble>(ParameterHandler.EnsembleParameters);
    RP = std::make_unique<RingPolymer>(
        ParameterHandler.RingPolymerParameters, *Ens);
  }

  ~RingPolymerTest() {
  }

  std::unique_ptr<Ensemble> Ens;
  std::unique_ptr<RingPolymer> RP;
};

TEST_F(RingPolymerTest, BasicTest) {
  ASSERT_EQ(RP->n_beads, RP->position_cart.n_rows);
  ASSERT_EQ(RP->n_total_atoms, RP->position_cart.n_slices);
}
