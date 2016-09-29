#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
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
void Linear_PNP::free_fasp () {
  fasp_dcsr_free(&_fasp_matrix);
  fasp_dbsr_free(&_fasp_bsr_matrix);
  fasp_dvec_free(&_fasp_vector);
  fasp_dvec_free(&_fasp_soln);
}
//--------------------------------------
