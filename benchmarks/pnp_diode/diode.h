#ifndef __DIODE_H
#define __DIODE_H

#include <dolfin.h>

/**
 * Define coefficients for PNP diode problem
 * along with dimensional analysis
 */

// Constants
const double HALF_PI = 1.57079632679;
const double ELEMENTARY_CHARGE = 1.60217662e-19; // C
const double BOLTZMANN = 1.38064852e-23; // J / K
const double VACUUM_PERMITTIVITY = 8.854187817e-12; // C / V*m

// device
const double temperature = 3e+2; // K
const double relative_permittivity = 1.17e+1; // for silicon

// maj * min = intrinsic carrier density = 1e+16 / m^3
// min - maj = doping level
const double majority_carrier = 1.5e+22; // 1 / m^3
const double minority_carrier = 6.66667e+9; // 1 / m^3

// const double majority_carrier = 5.0e+20;
// const double minority_carrier = 2.0e+10;

// const double majority_carrier = 1.0e+19;
// const double minority_carrier = 1.0e+12;


// n-doped Si (phospherus)
const double n_doping_level = majority_carrier - minority_carrier; // 1 / m^3
const double n_hole_diffusivity = 10.9e-4; // m^2 / s
const double n_electron_diffusivity = 28.74e-4; // m^2 / s
const double n_hole_boundary_value = minority_carrier;
const double n_electron_boundary_value = majority_carrier;


// pi-doped Si (boron)
const double p_doping_level = majority_carrier - minority_carrier; // 1 / m^3
const double p_hole_diffusivity = 10.68e-4; // m^2 / s
const double p_electron_diffusivity = 26.93e-4; // m^2 / s
const double p_hole_boundary_value = majority_carrier;
const double p_electron_boundary_value = minority_carrier;

// reference values
const double reference_length = 1e-5; // m
const double reference_potential = temperature * BOLTZMANN / ELEMENTARY_CHARGE; // J/C = V
const double reference_density = std::max(
  p_hole_boundary_value,
  n_electron_boundary_value
); // 1 / m^3
const double reference_diffusivity = std::max(
  p_hole_diffusivity,
  std::max(
    n_hole_diffusivity,
    std::max(p_electron_diffusivity, n_electron_diffusivity)
  )
); // m^2 / s
const double reference_permittivity = reference_density * reference_length * reference_length *
  ELEMENTARY_CHARGE / reference_potential; // F / m

double scale_density (double density) { return density / reference_density; };
double scale_potential (double phi) { return phi / reference_potential; };
double scale_diffusivity (double diff) { return diff / reference_diffusivity; };
double scale_permittivity (double perm) { return perm / reference_permittivity; };

double material_property (double x, double left_val, double right_val) {
  const double transition = 0.05;
  if (fabs(x) < transition) {
    const double difference = right_val - left_val;
    const double sum = right_val + left_val;
    return 0.5 * (sum + difference * std::sin(x * HALF_PI / transition));
  }

  return x < 0.0 ? left_val : right_val;
};

/// define coefficients
std::vector<double> valencies = {
  0.0,
  1.0,
  -1.0
}; // e_c: potential "valency" is at valencies[0] and should be zero
std::vector<double> reactions (double x) { return { 0.0, 0.0, 0.0 }; };
std::vector<double> diffusivities (double x) {
  return {
    0.0,
    scale_diffusivity( material_property(x, n_hole_diffusivity, p_hole_diffusivity) ), // cm^2 / s
    scale_diffusivity( material_property(x, n_electron_diffusivity, p_electron_diffusivity) ) // cm^2 / s
  };
};
double permittivity (double x) {
  return relative_permittivity * scale_permittivity(VACUUM_PERMITTIVITY);
};
double fixed_charge (double x) {
  return scale_density( material_property(x, n_doping_level, -p_doping_level) );
};

/**
 * boundary conditions
 */
std::vector<double> left_contact (double voltage_drop) {
  return {
    0.5 * scale_potential(voltage_drop), // V
    scale_density(n_hole_boundary_value), // 1 / m^3
    scale_density(n_electron_boundary_value) // 1 / m^3
  };
};
std::vector<double> right_contact (double voltage_drop) {
  return {
    -0.5 * scale_potential(voltage_drop), // V
    scale_density(p_hole_boundary_value), // 1 / m^3
    scale_density(p_electron_boundary_value) // 1 / m^3
  };
};

/**
 * define expressions from coefficients
 */
class Initial_Guess : public dolfin::Expression {
  public:
    Initial_Guess (double voltage_drop) : dolfin::Expression(3), volt(voltage_drop) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> left( left_contact(volt) );
      std::vector<double> right( right_contact(volt) );

      values[0] = material_property(x[0], left[0], right[0]);
      values[1] = std::log( material_property(x[0], left[1], right[1]) );
      values[2] = std::log( material_property(x[0], left[2], right[2]) );
    }
  private:
    double volt;
};

class Permittivity_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = permittivity(x[0]);
    }
};

class Poisson_Scale_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = 1.0;
    }
};

class Fixed_Charged_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = fixed_charge(x[0]);
    }
};

class Diffusivity_Expression : public dolfin::Expression {
  public:
    Diffusivity_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> diff(diffusivities(x[0]));
      values[0] = diff[0];
      values[1] = diff[1];
      values[2] = diff[2];
    }
};

class Reaction_Expression : public dolfin::Expression {
  public:
    Reaction_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> reac(reactions(x[0]));
      values[0] = reac[0];
      values[1] = reac[1];
      values[2] = reac[2];
    }
};

class Valency_Expression : public dolfin::Expression {
  public:
    Valency_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = valencies[0]; // potential valency is not used and should be zero
      values[1] = valencies[1];
      values[2] = valencies[2];
    }
};

#endif
