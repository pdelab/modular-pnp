#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "domain.h"
#include "dirichlet.h"
#include "poisson.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "poisson_forms.h"

using namespace std;

//--------------------------------------
Poisson::Poisson (
  const std::shared_ptr<const dolfin::Mesh> mesh,
  const domain_param &domain,
  const std::map<std::string, double> coefficients,
  const itsolver_param &itsolver,
  const AMG_param &amg
) {
  Poisson::update_mesh(mesh);

  _function_space.reset(
    new poisson_forms::FunctionSpace(*_mesh)
  );

  _bilinear_form.reset(
    new poisson_forms::Form_a(_function_space, _function_space)
  );
  _linear_form.reset(
    new poisson_forms::Form_L(_function_space)
  );

  Poisson::set_solution_to_constant(0.0);

  Poisson::_construct_coefficients();
  Poisson::set_coefficients(coefficients);

  _itsolver = itsolver;
  _amg = amg;

}
//--------------------------------------
Poisson::~Poisson() {}
//--------------------------------------
void Poisson::update_mesh (
  const std::shared_ptr<const dolfin::Mesh> mesh
) {
  _mesh.reset(new dolfin::Mesh(*mesh));

  _mesh_dim = _mesh->topology().dim();
  _mesh_epsilon = 0.1 * _mesh->rmin();
  for (std::size_t d = 0; d < _mesh_dim; d++) {
    _mesh_max.push_back(-1E+20);
    _mesh_min.push_back(+1E+20);
  }

  double coord_value = 0.0;
  for (std::size_t vert = 0; vert < _mesh->num_vertices(); ++vert) {
    for (std::size_t d = 0; d < _mesh_dim; d++) {
      coord_value = _mesh->coordinates()[vert * _mesh_dim + d];
      _mesh_max[d] = coord_value > _mesh_max[d] ? coord_value : _mesh_max[d];
      _mesh_min[d] = coord_value < _mesh_min[d] ? coord_value : _mesh_min[d];
    }
  }
}
//--------------------------------------
dolfin::Mesh Poisson::get_mesh () {
  return *(_mesh);
}
//--------------------------------------
void Poisson::use_quasi_newton () {
  _quasi_newton = true;
}
//--------------------------------------
void Poisson::use_exact_newton () {
  _quasi_newton = false;
}
//--------------------------------------
void Poisson::set_solution_to_constant (
  double value
) {
  _solution_function.reset( new dolfin::Function(_function_space) );

  std::shared_ptr<dolfin::Constant> constant_fn;
  if (_solution_function->value_rank() == 0) {
    constant_fn.reset(new dolfin::Constant(value));
  }
  else {
    std::vector<double> constant_value(_solution_function->value_rank(), value);
    constant_fn.reset(new dolfin::Constant(constant_value));
  }
  _solution_function->interpolate(*constant_fn);

  _linear_form->uu = *(_solution_function);
}
//--------------------------------------
void Poisson::set_solution (
  Linear_Function::Linear_Function expression
) {
  _solution_function.reset( new dolfin::Function(_function_space) );
  _solution_function->interpolate(expression);
  _linear_form->uu = *(_solution_function);
}
//--------------------------------------
dolfin::Function Poisson::get_solution () {
  return *(_solution_function);
}
//--------------------------------------
void Poisson::_construct_coefficients () {
  std::string diffusivity("diffusivity");
  std::string reactivity("reactivity");
  std::string source("source");

  std::shared_ptr<const dolfin::Constant> diffusivity_fn;
  std::shared_ptr<const dolfin::Constant> reactivity_fn;
  std::shared_ptr<const dolfin::Constant> source_fn;

  _bilinear_coefficient.clear();
  _bilinear_coefficient[diffusivity] = diffusivity_fn;
  _bilinear_coefficient[reactivity] = reactivity_fn;

  _linear_coefficient.clear();
  _linear_coefficient[diffusivity] = diffusivity_fn;
  _linear_coefficient[reactivity] = reactivity_fn;
  _linear_coefficient[source] = source_fn;
}
//--------------------------------------
void Poisson::print_coefficients () {

  printf("The coefficients of the bilinear form are:\n");
  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator bc;
  for (bc = _bilinear_coefficient.begin(); bc != _bilinear_coefficient.end(); ++bc) {
    printf("\t%s\n", bc->first.c_str());
  }

  printf("The coefficients of the linear form are:\n");
  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator lc;
  for (lc = _linear_coefficient.begin(); lc != _linear_coefficient.end(); ++lc) {
    printf("\t%s\n", lc->first.c_str());
  }
}
//--------------------------------------
void Poisson::set_coefficients (
  std::map<std::string, double> values
) {
  std::shared_ptr<dolfin::Constant> constant_fn;

  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator bc;
  for (bc = _bilinear_coefficient.begin(); bc != _bilinear_coefficient.end(); ++bc) {
    constant_fn.reset( new dolfin::Constant(values.find(bc->first)->second) );
    _bilinear_form->set_coefficient(bc->first, constant_fn);
  }

  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator lc;
  for (lc = _linear_coefficient.begin(); lc != _linear_coefficient.end(); ++lc) {
    constant_fn.reset( new dolfin::Constant(values.find(lc->first)->second) );
    _linear_form->set_coefficient(lc->first, constant_fn);
  }
}
//--------------------------------------
void Poisson::remove_Dirichlet_dof (
  std::vector<std::size_t> coordinates
) {
  _dirichlet_SubDomain.reset(
    new Dirichlet_Subdomain::Dirichlet_Subdomain(coordinates, _mesh_min, _mesh_max, _mesh_epsilon)
  );

  std::shared_ptr<dolfin::Constant> zero_constant;
  if (_solution_function->value_rank() == 0) {
    zero_constant.reset(new dolfin::Constant(0.0));
  }
  else {
    std::vector<double> zero_vector(_solution_function->value_rank(), 0.0);
    zero_constant.reset(new dolfin::Constant(zero_vector));
  }

  _dirichletBC.reset(
    new dolfin::DirichletBC(_function_space, zero_constant, _dirichlet_SubDomain)
  );
}
//--------------------------------------
void Poisson::set_DirichletBC (
  std::size_t coordinate,
  double lower_value,
  double upper_value
) {
  Poisson::remove_Dirichlet_dof( {coordinate} );

  Linear_Function::Linear_Function linear_interpolant(
   coordinate,
   _mesh_min[coordinate],
   _mesh_max[coordinate],
   lower_value,
   upper_value
  );

  Poisson::set_solution(linear_interpolant);
}
//--------------------------------------
dolfin::Function Poisson::dolfin_solve () {
  dolfin::Function solution(_function_space);
  dolfin::Equation equation(_bilinear_form, _linear_form);
  dolfin::solve(equation, solution, *_dirichletBC);

  *(solution.vector()) += *(_solution_function->vector());
  return solution;
}
//--------------------------------------
dolfin::SubDomain Poisson::get_Dirichlet_SubDomain () {
  return *(_dirichlet_SubDomain);
}
//--------------------------------------
