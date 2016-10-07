#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
#include "EAFE.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"

using namespace std;

//--------------------------------------
Linear_PNP::Linear_PNP (
  const std::shared_ptr<const dolfin::Mesh> mesh,
  const std::shared_ptr<dolfin::FunctionSpace> function_space,
  const std::shared_ptr<dolfin::Form> bilinear_form,
  const std::shared_ptr<dolfin::Form> linear_form,
  const std::map<std::string, std::vector<double>> coefficients,
  const std::map<std::string, std::vector<double>> sources,
  const itsolver_param &itsolver,
  const AMG_param &amg
) : PDE (
  mesh,
  function_space,
  bilinear_form,
  linear_form,
  coefficients,
  sources
) {

  diffusivity_space.reset(
    new vector_linear_pnp_forms::CoefficientSpace_diffusivity(*mesh)
  );

  valency_space.reset(
    new vector_linear_pnp_forms::CoefficientSpace_valency(*mesh)
  );

  fixed_charge_space.reset(
    new vector_linear_pnp_forms::CoefficientSpace_fixed_charge(*mesh)
  );

  permittivity_space.reset(
    new vector_linear_pnp_forms::CoefficientSpace_permittivity(*mesh)
  );

  _itsolver = itsolver;
  _amg = amg;
}
//--------------------------------------
Linear_PNP::~Linear_PNP () {}
//--------------------------------------





//--------------------------------------
void Linear_PNP::setup_fasp_linear_algebra () {
  Linear_PNP::setup_linear_algebra();

  std::size_t dimension = Linear_PNP::get_solution_dimension();
  EigenMatrix_to_dCSRmat(_eigen_matrix, &_fasp_matrix);
  _fasp_bsr_matrix = fasp_format_dcsr_dbsr(&_fasp_matrix, dimension);

  EigenVector_to_dvector(_eigen_vector, &_fasp_vector);
  if (_faps_soln_unallocated) {
    fasp_dvec_alloc(_eigen_vector->size(), &_fasp_soln);
    _faps_soln_unallocated = false;
  }

  fasp_dvec_set(_fasp_vector.row, &_fasp_soln, 0.0);
}
//--------------------------------------
dolfin::Function Linear_PNP::fasp_solve () {
  Linear_PNP::setup_fasp_linear_algebra();

  dolfin::Function solution(Linear_PNP::get_solution());

  printf("Solving linear system using FASP solver...\n"); fflush(stdout);
  INT status = fasp_solver_dbsr_krylov_amg (
    &_fasp_bsr_matrix,
    &_fasp_vector,
    &_fasp_soln,
    &_itsolver,
    &_amg
  );

  if (status < 0) {
    printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
    fflush(stdout);
  }
  else {
    printf("Successfully solved the linear system\n");
    fflush(stdout);

    dolfin::EigenVector solution_vector(_eigen_vector->size());
    double* array = solution_vector.data();
    for (std::size_t i = 0; i < _fasp_soln.row; ++i) {
      array[i] = _fasp_soln.val[i];
    }

    dolfin::Function update (
      Linear_PNP::_convert_EigenVector_to_Function(solution_vector)
    );
    *(solution.vector()) += *(update.vector());
  }

  Linear_PNP::set_solution(solution);

  return solution;
}
//--------------------------------------
dolfin::EigenVector Linear_PNP::fasp_test_solver (
  const dolfin::EigenVector& target_vector
) {
  Linear_PNP::setup_fasp_linear_algebra();

  printf("Compute RHS...\n"); fflush(stdout);
  std::shared_ptr<dolfin::EigenVector> rhs_vector;
  rhs_vector.reset( new dolfin::EigenVector(target_vector.size()) );


  _eigen_matrix->mult(target_vector, *rhs_vector);
  EigenVector_to_dvector(rhs_vector, &_fasp_vector);

  dolfin::Function solution(Linear_PNP::get_solution());

  printf("Solving linear system using FASP solver...\n"); fflush(stdout);
  INT status = fasp_solver_dbsr_krylov_amg (
    &_fasp_bsr_matrix,
    &_fasp_vector,
    &_fasp_soln,
    &_itsolver,
    &_amg
  );

  if (status < 0) {
    printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
    fflush(stdout);

    return target_vector;
  }

  printf("Successfully solved the linear system\n");
  fflush(stdout);

  dolfin::EigenVector solution_vector(_eigen_vector->size());
  double* array = solution_vector.data();
  for (std::size_t i = 0; i < _fasp_soln.row; ++i) {
    array[i] = _fasp_soln.val[i];
  }

  return solution_vector;
}
//--------------------------------------
void Linear_PNP::free_fasp () {
  fasp_dcsr_free(&_fasp_matrix);
  fasp_dbsr_free(&_fasp_bsr_matrix);
  fasp_dvec_free(&_fasp_vector);
  fasp_dvec_free(&_fasp_soln);
}
//--------------------------------------


//--------------------------------------
void Linear_PNP::apply_eafe () {

  dolfin::Function solution_function(Linear_PNP::get_solution());

  if (_eafe_uninitialized) {
    _eafe_function_space = solution_function[0].function_space()->collapse();
    _eafe_bilinear_form.reset(
      new EAFE::Form_a(_eafe_function_space, _eafe_function_space)
    );
    _eafe_matrix.reset(new dolfin::EigenMatrix());


    std::shared_ptr<dolfin::Function> _diffusivity;
    _diffusivity.reset(new dolfin::Function(diffusivity_space));
    _diffusivity->interpolate(
      *(_bilinear_form->coefficient("diffusivity"))
    );
    _split_diffusivity = Linear_PNP::split_mixed_function(_diffusivity);


    dolfin::Function val_fn(valency_space);
    val_fn.interpolate(
      (*_bilinear_form->coefficient("valency"))
    );

    std::size_t vals = Linear_PNP::get_solution_dimension() + 1;
    _valency_double.reserve(vals);
    for (uint val_idx = 0; val_idx < vals; val_idx++) {
      _valency_double[val_idx] = (*(val_fn.vector()))[val_idx];
    }
    _valency_double[0] = 0.0;
  }

  std::shared_ptr<dolfin::Function> solution_ptr;
  solution_ptr.reset( new dolfin::Function(solution_function.function_space()) );
  *solution_ptr = solution_function;

  const dolfin::Constant zero(0.0);
  std::vector<std::shared_ptr<dolfin::Function>> solution_vector;
  solution_vector = Linear_PNP::split_mixed_function(solution_ptr);
  std::shared_ptr<dolfin::Function> beta, eta, phi;


  std::size_t eqns = Linear_PNP::get_solution_dimension();
  for (uint eqn_idx = 1; eqn_idx < eqns; eqn_idx++) {
    beta.reset(new dolfin::Function(_eafe_function_space));
    eta.reset(new dolfin::Function(_eafe_function_space));
    phi.reset(new dolfin::Function(_eafe_function_space));

    beta->interpolate( *(solution_vector[eqn_idx]) );
    eta->interpolate( *(solution_vector[eqn_idx]) );
    phi->interpolate( *(solution_vector[0]) );
    if (_valency_double[eqn_idx] != 0.0) {
      *(phi->vector()) *= _valency_double[eqn_idx];
      *beta = *beta + *phi;
      *(phi->vector()) /= _valency_double[eqn_idx];
    }

    _eafe_bilinear_form->alpha = *(_split_diffusivity[eqn_idx]);
    _eafe_bilinear_form->beta = *beta;
    _eafe_bilinear_form->eta = *eta;
    _eafe_bilinear_form->gamma = zero;

    dolfin::assemble(*_eafe_matrix, *_eafe_bilinear_form);

    // replace_matrix (
    //   Linear_PNP::get_solution_dimension(),
    //   eqn_idx,
    //   &_eigen_matrix,
    //   &_eafe_matrix
    // );
  }
}
//--------------------------------------

//--------------------------------------
std::vector<std::shared_ptr<dolfin::Function>> Linear_PNP::split_mixed_function (
  std::shared_ptr<const dolfin::Function> mixed_function
) {
  //construct assignment arguments
  std::shared_ptr<const dolfin::FunctionSpace> mixed_space(
    mixed_function->function_space()
  );

  // construct receiving arguments
  std::vector<std::shared_ptr<dolfin::Function>> function_vector;
  std::vector<std::shared_ptr<const dolfin::FunctionSpace>> subspace_vector;

  // initialize vectors
  std::shared_ptr<dolfin::Function> subfunction;
  std::shared_ptr<const dolfin::FunctionSpace> subspace;
  std::size_t num_components = mixed_space->element()->num_sub_elements();

  for (std::size_t c = 0; c < num_components; c++) {
    subspace.reset(
      new const dolfin::FunctionSpace( *( (*mixed_space)[c]->collapse() ) )
    );
    subfunction.reset(new dolfin::Function(subspace));

    subspace_vector.push_back(subspace);
    function_vector.push_back(subfunction);
  }

  // assign
  dolfin::FunctionAssigner function_assigner(subspace_vector, mixed_space);
  function_assigner.assign(function_vector, mixed_function);

  return function_vector;
}
