#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
#include "error.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "L2Error.h"
#include "SemiH1error.h"

using namespace std;

//--------------------------------------
Error::Error (
  std::shared_ptr<const dolfin::Function> exact_solution
) {
  Error::update_exact_solution(exact_solution);

  _l2_form.reset(new L2Error::Functional(_function_space->mesh()));
  _semi_h1_form.reset(new SemiH1error::Functional(_function_space->mesh()));

}
//--------------------------------------
Error::~Error() {}
//--------------------------------------

//--------------------------------------
void Error::update_exact_solution (
  std::shared_ptr<const dolfin::Function> exact_solution
) {
  _function_space.reset(
    new const dolfin::FunctionSpace( *(exact_solution->function_space()) )
  );

  _exact_solution.reset(new const dolfin::Function(_function_space));
  _exact_solution = exact_solution;

  _num_subfunctions = _function_space->element()->num_sub_elements();
}
//--------------------------------------
dolfin::Function Error::compute_error (
  std::shared_ptr<dolfin::Function> computed_solution
) {
  dolfin::Function error(_function_space);
  error.interpolate(*computed_solution);
  error = error - *_exact_solution;

  return error;
}
//--------------------------------------
double Error::compute_l2_error (
  std::shared_ptr<dolfin::Function> computed_solution
) {

  double l2_error = 0.0;
  dolfin::Function error_function(_function_space);
  error_function = Error::compute_error(computed_solution);

  std::shared_ptr<dolfin::Function> error_temp;
  for (std::size_t index = 0; index < _num_subfunctions; index++) {
    error_temp = std::make_shared<dolfin::Function>(error_function[index]);
    _l2_form->error = error_temp;
    l2_error += assemble(*_l2_form);
  }

  return std::sqrt(l2_error);
}
//--------------------------------------
double Error::compute_semi_h1_error (
  std::shared_ptr<dolfin::Function> computed_solution
) {

  double semi_h1_error = 0.0;
  dolfin::Function error_function(_function_space);
  error_function = Error::compute_error(computed_solution);

  std::shared_ptr<dolfin::Function> error_temp;
  for (std::size_t index = 0; index < _num_subfunctions; index++) {
    error_temp = std::make_shared<dolfin::Function>(error_function[index]);
    _semi_h1_form->error = error_temp;
    semi_h1_error += assemble(*_semi_h1_form);
 }

  return std::sqrt(semi_h1_error);
}
//--------------------------------------
double Error::compute_h1_error (
  std::shared_ptr<dolfin::Function> computed_solution
) {
  double l2_error = Error::compute_l2_error(computed_solution);
  double semi_h1_error = Error::compute_semi_h1_error(computed_solution);

  return std::sqrt(l2_error * l2_error + semi_h1_error * semi_h1_error);
}
//--------------------------------------

  // for (std::size_t charge = 1; charge < solution_size; charge++) {
  //   double q = (*(valencies.vector()))[charge];
  //   solution_charge.reset(new dolfin::Function(fixed_charge_space));
  //   solution_charge->interpolate(solution[charge]);

  //   total_charge = total_charge + (*solution_charge);
  // }
