#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "domain.h"
#include "dirichlet.h"
#include "vector_poisson.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_poisson_forms.h"

using namespace std;

//--------------------------------------
Vector_Poisson::Vector_Poisson (
  const std::shared_ptr<const dolfin::Mesh> mesh,
  const domain_param &domain,
  const std::map<std::string, std::vector<double>> coefficients,
  const itsolver_param &itsolver,
  const AMG_param &amg
) {
  Vector_Poisson::update_mesh(mesh);

  _function_space.reset(
    new vector_poisson_forms::FunctionSpace(*_mesh)
  );

  _bilinear_form.reset(
    new vector_poisson_forms::Form_a(_function_space, _function_space)
  );
  _linear_form.reset(
    new vector_poisson_forms::Form_L(_function_space)
  );

  printf("set solution\n"); fflush(stdout);
  Vector_Poisson::set_solution(0.0);

  printf("construct coeffs\n"); fflush(stdout);
  Vector_Poisson::_construct_coefficients();

  printf("set coeffs\n"); fflush(stdout);
  Vector_Poisson::set_coefficients(coefficients);

  _itsolver = itsolver;
  _amg = amg;

}
//--------------------------------------
Vector_Poisson::~Vector_Poisson() {}
//--------------------------------------
void Vector_Poisson::update_mesh (
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
dolfin::Mesh Vector_Poisson::get_mesh () {
  return *(_mesh);
}
//--------------------------------------
void Vector_Poisson::use_quasi_newton () {
  _quasi_newton = true;
}
//--------------------------------------
void Vector_Poisson::use_exact_newton () {
  _quasi_newton = false;
}
//--------------------------------------
std::size_t Vector_Poisson::_get_solution_dimension() {
  return _solution_function->function_space()
    ->element()
    ->num_sub_elements();
}
//--------------------------------------
void Vector_Poisson::set_solution (
  double value
) {
  std::shared_ptr<dolfin::Constant> constant_fn;
  _solution_function.reset( new dolfin::Function(_function_space) );
  std::size_t dimension = Vector_Poisson::_get_solution_dimension();

  if (dimension == 0) {
    constant_fn.reset( new dolfin::Constant(value) );
    _solution_function->interpolate(*constant_fn);
  }
  else {
    std::vector<double> values(dimension, value);
    constant_fn.reset( new dolfin::Constant(values) );
    _solution_function->interpolate(*constant_fn);
  }

  _linear_form->uu = *(_solution_function);
}
//--------------------------------------
void Vector_Poisson::set_solution (
  std::vector<double> value
) {
  _solution_function.reset( new dolfin::Function(_function_space) );
  std::shared_ptr<dolfin::Constant> constant_fn;
  std::size_t dimension = Vector_Poisson::_get_solution_dimension();

  if (dimension == value.size()) {
    constant_fn.reset( new dolfin::Constant(value) );
    _solution_function->interpolate(*constant_fn);
  }
  else {
    printf("Dimension mismatch!!\n");
    printf("\tsetting solution to zeros %lu \n", dimension); fflush(stdout);
    constant_fn.reset( new dolfin::Constant(dimension, 0.0) );
    _solution_function->interpolate(*constant_fn);
  }

  _linear_form->uu = *(_solution_function);
}
//--------------------------------------
void Vector_Poisson::set_solution (
  const std::vector<Linear_Function::Linear_Function> expression
) {

  std::size_t dimension = Vector_Poisson::_get_solution_dimension();
  if (dimension == expression.size()) {

    std::vector<std::shared_ptr<const dolfin::FunctionSpace>> vector_space;
    std::vector<std::shared_ptr<const dolfin::Function>> vector_expression;

    std::shared_ptr<const dolfin::FunctionSpace> sub_space;
    std::shared_ptr<const dolfin::Function> const_sub_expression;
    std::shared_ptr<dolfin::Function> sub_expression;

    for (std::size_t i = 0; i < dimension; i++) {
      sub_space.reset(
        new const dolfin::FunctionSpace( *((*_function_space)[i]->collapse()) )
      );
      sub_expression.reset( new dolfin::Function(sub_space) );
      sub_expression->interpolate(expression[i]);

      const_sub_expression.reset( new const dolfin::Function(*sub_expression) );
      vector_expression.push_back(const_sub_expression);
      vector_space.push_back(sub_space);
    }
    dolfin::FunctionAssigner function_assigner(_function_space, vector_space);
    function_assigner.assign(_solution_function, vector_expression);
  }
  else {
    printf("Dimension mismatch!!\n");
    printf("\tsetting solution to zeros %lu \n", dimension);
    std::shared_ptr<dolfin::Constant> constant_fn;
    constant_fn.reset( new dolfin::Constant(dimension, 0.0) );
    _solution_function->interpolate(*constant_fn);
  }

  _linear_form->uu = *_solution_function;
}
//--------------------------------------
dolfin::Function Vector_Poisson::get_solution () {
  return *(_solution_function);
}
//--------------------------------------
void Vector_Poisson::_construct_coefficients () {
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
void Vector_Poisson::print_coefficients () {

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
void Vector_Poisson::set_coefficients (
  std::map<std::string, std::vector<double>> values
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
void Vector_Poisson::remove_Dirichlet_dof (
  std::vector<std::size_t> coordinates
) {

  std::size_t dimension = Vector_Poisson::_get_solution_dimension();

  _dirichlet_SubDomain.clear();
  _dirichlet_SubDomain.resize(dimension);

  _dirichletBC.clear();
  _dirichletBC.resize(dimension);

  std::shared_ptr<dolfin::Constant> zero_constant;
  zero_constant.reset(new dolfin::Constant(0.0));

  for (std::size_t i = 0; i < dimension; i++) {
    _dirichlet_SubDomain[i].reset(
      new Dirichlet_Subdomain::Dirichlet_Subdomain({coordinates[i]}, _mesh_min, _mesh_max, _mesh_epsilon)
    );
    _dirichletBC[i].reset(
      new dolfin::DirichletBC((*_function_space)[i], zero_constant, _dirichlet_SubDomain[i])
    );
  }
}
//--------------------------------------
void Vector_Poisson::set_DirichletBC (
  std::vector<std::size_t> component,
  std::vector<std::vector<double>> boundary
) {

  Vector_Poisson::remove_Dirichlet_dof(component);

  std::vector<Linear_Function::Linear_Function> interpolant_vector;
  for (std::size_t i = 0; i < component.size(); i++) {
    Linear_Function::Linear_Function linear_interpolant(
     component[i],
     _mesh_min[component[i]],
     _mesh_max[component[i]],
     boundary[i][0],
     boundary[i][1]
    );
    interpolant_vector.push_back(linear_interpolant);
  }

  Vector_Poisson::set_solution(interpolant_vector);
}
//--------------------------------------
dolfin::Function Vector_Poisson::dolfin_solve () {
  dolfin::Function solution_update(_function_space);
  dolfin::Equation equation(_bilinear_form, _linear_form);

  std::vector<const dolfin::DirichletBC *> dirichletBC_vector;
  for (std::size_t i = 0; i < _dirichletBC.size(); i++) {
    dirichletBC_vector.push_back( &(*(_dirichletBC.at(i))) );
  }

  dolfin::solve(equation, solution_update, dirichletBC_vector);
  *(_solution_function->vector()) += *(solution_update.vector());

  dolfin::File sol_file("./benchmarks/vector_poisson/poisson/solve_update.pvd");
  sol_file << solution_update[0];
  sol_file << solution_update[1];
  sol_file << solution_update[2];
  sol_file << solution_update[3];

  dolfin::File sol_file1("./benchmarks/vector_poisson/poisson/in_solve.pvd");
  sol_file1 << (*_solution_function)[0];
  sol_file1 << (*_solution_function)[1];
  sol_file1 << (*_solution_function)[2];
  sol_file1 << (*_solution_function)[3];

  _linear_form->uu = *_solution_function;

  return *_solution_function;
}
//--------------------------------------
void Vector_Poisson::update_solution(
  dolfin::Function solution,
  const dolfin::Function& update
) {
  *(solution.vector()) += *(update.vector());
}
//--------------------------------------
std::vector<std::shared_ptr<dolfin::SubDomain>> Vector_Poisson::get_Dirichlet_SubDomain () {
  return _dirichlet_SubDomain;
}
//--------------------------------------
