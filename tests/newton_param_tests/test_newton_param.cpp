/*! \file test_faspfenics.cpp
 *
 *  \brief Main to test FASP/FENICS interface using the Poisson problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include <vector>
#include "newton.h"
#include "newton_functs.h"

bool DEBUG = false;

int main(int argc, char** argv)
{

  if (argc >1)
  {
    if (std::string(argv[1])=="DEBUG") DEBUG = true;
  }

  /**
   * First test to see if the boundary conditions works
   */
  // initialize mesh
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of Newton parameters                               #### \n";
    std::cout << "################################################################# \n";
  }

  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/PNP/newton_param.dat";
  newton_param_input (newton_param_file,&newtparam);
  if (DEBUG) print_newton_param(&newtparam);

  double adapt_tol 		= 0.0;
  double nonlin_tol 		= 1E-8;
  double nonlin_maxit		= 15;
  double nonlin_damp_factor	= 0.5;
  double nonlin_damp_it		= 5;


  if ( (newtparam.adapt_tol==adapt_tol) && (newtparam.tol==nonlin_tol) && (newtparam.max_it== nonlin_maxit) && (newtparam.damp_factor==nonlin_damp_factor) && (newtparam.damp_it==nonlin_damp_it) )
  {
    printf("Success... passed reading newton parameters\n");
  }
  else {
    printf("***\tERROR IN NEWTON PARAM TEST\n");
    printf("***\n***\n***\n");
    printf("***\tNEWTON PARAM TEST:\n");
    printf("***\tThe parematers are read wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN NEWTON PARAM TEST\n");
    fflush(stdout);
    return -1;
  }

  if (DEBUG){
    std::cout << "################################################################# \n";
    std::cout << "#### End of test of boundary_condtions.cpp                   #### \n";
    std::cout << "################################################################# \n";
  }
  return 0;
}
