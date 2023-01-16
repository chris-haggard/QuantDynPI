#pragma once

#include <unordered_map>

/**
 * @brief Struct to hold atom masses map. All members static so no object needs
 * to be created.
 */
struct AtomData {
 private:
  static std::unordered_map<std::string, const double> atom_mass;

 public:
  static double GetAtomMass(const std::string);
};

