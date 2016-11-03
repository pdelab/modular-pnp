#include <stdio.h>
#include "domain.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
}

/**
 * \fn SHORT domain_param_input_init (domain_param *inparam)
 *
 * \brief Initialize domain input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 */
SHORT domain_param_input_init (domain_param *inparam) {
    SHORT status = FASP_SUCCESS;

    inparam->ref_length = 1.0;
    inparam->length_x = 1.0;
    inparam->length_y = 1.0;
    inparam->length_z = 1.0;
    inparam->length_time = 0.0;

    inparam->grid_x = 10;
    inparam->grid_y = 10;
    inparam->grid_z = 10;
    inparam->grid_time = 0;

    strcpy(inparam->mesh_output,"./output/mesh.pvd");
    strcpy(inparam->mesh_file,"none");
    strcpy(inparam->subdomain_file,"none");
    strcpy(inparam->surface_file,"none");

    return status;
}

/**
 * \fn SHORT domain_param_check (domain_param *inparam)
 *
 * \brief Simple check on domain input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 */
SHORT domain_param_check (domain_param *inparam) {
  SHORT status = FASP_SUCCESS;

  if (
  inparam->ref_length < 0.0
  || inparam->length_x < 0.0
  || inparam->length_y < 0.0
  || inparam->length_z < 0.0
  || inparam->length_time < 0.0
  || inparam->grid_x < 2
  || inparam->grid_y < 2
  || inparam->grid_z < 2
  || inparam->grid_time < 0
  ) {
    status = ERROR_INPUT_PAR;
  }

  return status;
}


/**
 * \fn void domain_param_input (const char *filenm,
 *                              domain_param *inparam)
 *
 * \brief Read in parameters for domain solver
 *
 * \param filenm    File name for input parameters
 * \param inparam   Input parameters
 */
void domain_param_input (
  const char *filenm,
  domain_param *inparam
) {
    char     buffer[500]; // Note: max number of char for each line!
    int      val;
    SHORT    status = FASP_SUCCESS;

    // set default input parameters
    domain_param_input_init(inparam);

    // if input file is not specified, use the default values
    if (filenm==NULL) return;

    FILE *fp = fopen(filenm,"r");
    if (fp==NULL) {
        printf("### ERROR: Could not open file %s...\n", filenm);
        fasp_chkerr(ERROR_OPEN_FILE, "domain_param_input");

    }

    while ( status == FASP_SUCCESS ) {
        int     ibuff;
        double  dbuff;
        char    sbuff[500];
        char   *fgetsPtr;

        val = fscanf(fp,"%s",buffer);
        if (val==EOF) break;
        if (val!=1){ status = ERROR_INPUT_PAR; break; }
        if (buffer[0]=='[' || buffer[0]=='%' || buffer[0]=='|') {
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
            continue;
        }

        // match keyword and scan for value
        if (strcmp(buffer,"ref_length")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->ref_length = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"length_x")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->length_x = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"length_y")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->length_y = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"length_z")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->length_z = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"length_time")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->length_time = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"grid_x")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->grid_x = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"grid_y")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->grid_y = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"grid_z")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->grid_z = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"grid_time")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->grid_time = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"mesh_output")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            strncpy(inparam->mesh_output,sbuff,128);
            fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"mesh_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            strncpy(inparam->mesh_file,sbuff,128);
            fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"subdomain_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            strncpy(inparam->subdomain_file,sbuff,128);
            fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"surface_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            strncpy(inparam->surface_file,sbuff,128);
            fgets(buffer,500,fp); // skip rest of line
        }

        else {
            printf("### WARNING: Unknown input keyword %s!\n", buffer);
            fgets(buffer,500,fp); // skip rest of line
        }
    }

    fclose(fp);

    // sanity checks
    status = domain_param_check(inparam);

    // if meet unexpected input, stop the program
    fasp_chkerr(status,"domain_param_input");
}

void print_domain_param (const domain_param &inparam) {
  printf("\tSuccessfully read-in domain parameters\n");
  if ( strcmp(inparam.mesh_file,"none")==0 ) {
    printf("\tThe reference length is %e meters\n",inparam.ref_length);
    printf("\tDomain: %f x %f x %f\n",  inparam.length_x,inparam.length_y,inparam.length_z);
    printf("\tGrid:   %d x %d x %d\n\n",inparam.grid_x,inparam.grid_y,inparam.grid_z);
    fflush(stdout);
  } else {
    printf("\tThe reference length is %e meters\n",inparam.ref_length);
    printf("\tMesh file:      %s\n",  inparam.mesh_file);
    printf("\tSubdomain file: %s\n",  inparam.subdomain_file);
    printf("\tSurface file:   %s\n\n",inparam.surface_file);
    fflush(stdout);
  }
}


dolfin::Mesh domain_build(const domain_param &domain) {
  if (std::isnan(domain.length_x) || std::isnan(domain.length_y) || std::isnan(domain.length_z)) {
    printf("### WARNING : invalid domain lengths!\n");
    return dolfin::Mesh();
  }

  if (std::isnan(domain.length_x) || std::isnan(domain.length_y) || std::isnan(domain.length_z)) {
    printf("### WARNING : invalid domain grid!\n");
    return dolfin::Mesh();
  }

  if (strcmp(domain.mesh_file, "none") != 0) {
    printf("### ERROR : Reading in meshes is currently unsupported: %s...\n\n", domain.mesh_file);
    return dolfin::Mesh();
  }

  dolfin::Point p0( -domain.length_x/2, -domain.length_y/2, -domain.length_z/2);
  dolfin::Point p1( domain.length_x/2, domain.length_y/2, domain.length_z/2);
  dolfin::BoxMesh box_mesh(p0, p1, domain.grid_x, domain.grid_y, domain.grid_z);

  return box_mesh;
}
