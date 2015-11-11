/*! \file boundary_conditions.cpp
 *  \brief contains the functions for each class of boundary conditions
 */

#include "boundary_conditions.h"

using namespace dolfin;

/**
 * Public Functions for SubDomains
 */
// constructor
XBoundaries::XBoundaries(double lower, double upper)
{
  _lower = lower;
  _upper = upper;
}
// Return 1 if on the x-boundary
bool XBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (
    std::fabs(x[0] - _lower) < 5.0*DOLFIN_EPS
    || std::fabs(x[0] - _upper) < 5.0*DOLFIN_EPS
  );
}
// constructor
YBoundaries::YBoundaries(double lower, double upper)
{
  _lower = lower;
  _upper = upper;
}
// Return 1 if on the y-boundary
bool YBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (
    std::fabs(x[1] - _lower) < 5.0*DOLFIN_EPS
    || std::fabs(x[1] - _upper) < 5.0*DOLFIN_EPS
  );
}
// constructor
ZBoundaries::ZBoundaries(double lower, double upper)
{
  _lower = lower;
  _upper = upper;
}
// Return 1 if on the z-boundary
bool ZBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (
    std::fabs(x[2] - _lower) < 5.0*DOLFIN_EPS
    || std::fabs(x[2] - _upper) < 5.0*DOLFIN_EPS
  );
}
// constructor
SymmBoundaries::SymmBoundaries(unsigned int coord, double lower, double upper)
{
  _coord = coord;
  _lower = lower;
  _upper = upper;
}
// Return 1 if on the z-boundary
bool SymmBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (
    std::fabs(x[_coord] - _lower) < 5.0*DOLFIN_EPS
    || std::fabs(x[_coord] - _upper) < 5.0*DOLFIN_EPS
  );
}

/**
 * Initial expressions for functions satisfying
 * boundary conditions
 */
//  constructor
LogCharge::LogCharge(double lower_val, double upper_val,
  double lower, double upper, int bc_coord) : Expression()
{
  _lower_val = lower_val;
  _upper_val = upper_val;
  _lower = lower;
  _upper = upper;
  _bc_coord = bc_coord;
}
// evaluate LogCharge Expression
void LogCharge::eval(Array<double>& values, const Array<double>& x) const
{
  values[0]  = std::log(_lower_val) * (_upper - x[_bc_coord]) / (_upper - _lower);
  values[0] += std::log(_upper_val) * (x[_bc_coord] - _lower) / (_upper - _lower);
}
//  constructor
Voltage::Voltage(double lower_val, double upper_val,
  double lower, double upper, int bc_coord) : Expression()
{
  _lower_val = lower_val;
  _upper_val = upper_val;
  _lower = lower;
  _upper = upper;
  _bc_coord = bc_coord;
}
// evaluate Voltage Expression
void Voltage::eval(Array<double>& values, const Array<double>& x) const
{
  values[0]  = _lower_val * (_upper - x[_bc_coord]) / (_upper - _lower);
  values[0] += _upper_val * (x[_bc_coord] - _lower) / (_upper - _lower);
}
