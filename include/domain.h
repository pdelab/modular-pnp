#ifndef __DOMAIN_H
#define __DOMAIN_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
}

typedef struct {
    //! reference length for domain
    REAL ref_length;
  //! dimension length along x-direction
    REAL length_x;
    //! dimension length along y-direction
    REAL length_y;
    //! dimension length along z-direction
    REAL length_z;
    //! dimension length along time direction
    REAL length_time;

    //! number of vertices along x-direction
    INT grid_x;
    //! number of vertices along y-direction
    INT grid_y;
    //! number of vertices along z-direction
    INT grid_z;
    //! number of vertices along time direction
    INT grid_time;

    //! string specifying location of mesh output file
    char mesh_output[128];
    //! string specifying location of mesh file
    char mesh_file[128];
    //! string specifying location of subdomain file
    char subdomain_file[128];
    //! string specifying location of surface file
    char surface_file[128];

} domain_param;

SHORT domain_param_input_init (domain_param *inparam);

SHORT domain_param_check (domain_param *inparam);

void domain_param_input (const char *filenm, domain_param *inparam);

dolfin::Mesh domain_build (const domain_param &domain);

#endif
