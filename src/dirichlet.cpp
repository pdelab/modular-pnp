#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "dirichlet.h"

using namespace std;

//--------------------------------------
Dirichlet_Subdomain::Dirichlet_Subdomain (
  std::vector<std::size_t> coordinates,
  std::vector<double> mesh_min,
  std::vector<double> mesh_max,
  double epsilon
) {
  _coordinates.swap(coordinates);
  _mesh_min.swap(mesh_min);
  _mesh_max.swap(mesh_max);
  _epsilon = epsilon;
}
//--------------------------------------
Dirichlet_Subdomain::~Dirichlet_Subdomain () {}
//--------------------------------------
bool Dirichlet_Subdomain::inside(
  const dolfin::Array<double>& x,
  bool on_boundary
) const {
  std::size_t d;
  bool dirichlet = false;
  for (std::size_t i = 0; i < _coordinates.size(); i++) {
    d = _coordinates[i];
    dirichlet = dirichlet or (
      x[d] < _mesh_min[d] + _epsilon or x[d] > _mesh_max[d] - _epsilon
    );
  }
  return on_boundary && dirichlet;
}
//--------------------------------------
Linear_Function::Linear_Function (
  std::size_t coordinate,
  double mesh_min,
  double mesh_max,
  double lower_value,
  double upper_value
) {
  _coordinate = coordinate;
  _mesh_min = mesh_min;
  _mesh_max = mesh_max;
  _lower_value = lower_value;
  _upper_value = upper_value;
  _distance = 1.0 / (_mesh_max - _mesh_min);
}
//--------------------------------------
Linear_Function::~Linear_Function () {}
//--------------------------------------
void Linear_Function::eval (
  dolfin::Array<double>& values,
  const dolfin::Array<double>& x
) const {
  values[0] = _lower_value * (_mesh_max - x[_coordinate]) * _distance;
  values[0] += _upper_value * (x[_coordinate] - _mesh_min) * _distance;
}
//--------------------------------------
