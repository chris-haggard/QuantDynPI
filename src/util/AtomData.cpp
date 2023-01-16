#include "AtomData.hpp"

#include <unordered_map>

/**
 * @brief Map to hold atomic masses in Hartree atomic units. Static struct so
 * can be called without an object created. NIST values, no isotopes.
 */
std::unordered_map<std::string, const double> AtomData::atom_mass = {
    {"H", 1837.15264737},   // H1
    {"H2", 3674.30529474},  // 2 * H1
    {"O", 29156.945698},    // O16
    {"N", 25526.0423743}};  // N14

double AtomData::GetAtomMass(const std::string s) {
  return atom_mass.at(s);
}
