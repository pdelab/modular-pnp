#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>

#include "newton_status.h"

using namespace std;

//--------------------------------------
Newton_Status::Newton_Status (
  const std::size_t max_iterations_in,
  const double initial_residual_in,
  const double rel_residual_tol_in,
  const double max_residual_tol_in
) {
  max_iterations = max_iterations_in;
  initial_residual = initial_residual_in;
  rel_residual_tol = rel_residual_tol_in;
  max_residual_tol = max_residual_tol_in;

  iteration = 1;
  relative_residual = 1.0;
  max_residual = max_residual_tol + 1;
}
//--------------------------------------
Newton_Status::~Newton_Status () {};
//--------------------------------------

//--------------------------------------
void Newton_Status::update_iteration () {
  iteration++;
}
//--------------------------------------
void Newton_Status::update_residuals (
  const double residual_in,
  const double max_residual_in
) {
  Newton_Status::update_rel_residual(residual_in);
  Newton_Status::update_max_residual(max_residual_in);
}
//--------------------------------------
void Newton_Status::update_max_residual (
  const double max_residual_in
) {
  max_residual = max_residual_in;
}
//--------------------------------------
void Newton_Status::update_rel_residual (
  const double residual_in
) {
  residual = residual_in;
  relative_residual = residual / initial_residual;
}
//--------------------------------------
bool Newton_Status::needs_to_iterate () {
  bool iterations_left = iteration < (max_iterations + 1);
  bool rel_res_too_large = relative_residual > rel_residual_tol;
  bool max_res_too_large = max_residual > max_residual_tol;

  return iterations_left && (
    rel_res_too_large || max_res_too_large 
  );
}
//--------------------------------------
bool Newton_Status::converged () {
  bool too_many_iters = iteration > max_iterations;
  bool accept_rel_res = relative_residual < rel_residual_tol;
  bool accept_max_res = max_residual < max_residual_tol;

  return !too_many_iters && (accept_rel_res && accept_max_res);
}
//--------------------------------------
void Newton_Status::print_status () {
  bool too_many_iters = iteration > max_iterations;
  bool accept_rel_res = relative_residual < rel_residual_tol;
  bool accept_max_res = max_residual < max_residual_tol;

  printf("Newton solver status:\n");
  if (too_many_iters) {
    printf("\ttoo many iterations\n");
  }
  if (!accept_rel_res) {
    printf("\trelative residual too large : %e >= %e\n",
      relative_residual, rel_residual_tol
    );
  }
  if (!accept_rel_res) {
    printf("\tmax residual too large : %e >= %e\n",
      max_residual, max_residual_tol
    );
  }

  if (!too_many_iters && (accept_rel_res && accept_max_res)) {
    printf("\tconverged\n");
  } else {
    printf("\tnot converged\n");
  }
}
//--------------------------------------
// void update_solution ();
// void update_residual_vector ();
// dolfin::Function damp_update ();
