#ifndef __PARAMS_FUNCTS_H
#define __PARAMS_FUNCTS_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "params.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
}

/*------------- In file: params_functs.cpp --------------*/

SHORT newton_param_input_init (newton_param *inparam);

SHORT newton_param_check (newton_param *inparam);

void newton_param_input (const char *filenm, newton_param *inparam);

void print_newton_param (newton_param *inparam);

SHORT domain_param_input_init (domain_param *inparam);

SHORT domain_param_check (domain_param *inparam);

void domain_param_input (const char *filenm, domain_param *inparam);

void print_domain_param (domain_param *inparam);

SHORT coeff_param_input_init (coeff_param *inparam);

SHORT coeff_param_check (coeff_param *inparam);

void coeff_param_input (const char *filenm, coeff_param *inparam);

void print_coeff_param (coeff_param *inparam);

dolfin::Mesh domain_build (const domain_param &domain);

#endif
