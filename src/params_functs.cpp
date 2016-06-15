#include <stdio.h>
#include "params.h"
#include "params_functs.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
}

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT newton_param_input_init (newton_param *inparam)
 *
 * \brief Initialize Newton input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 */
SHORT newton_param_input_init (newton_param *inparam)
{
    SHORT status = FASP_SUCCESS;

    inparam->max_it = 10;
    inparam->adapt_tol = 1.0e+10;
    inparam->tol = 1.0e-4;
    inparam->damp_factor = 1.0;
    inparam->damp_it = 5;
    inparam->max_cells = -1;

    return status;
}

/**
 * \fn SHORT newton_param_check (input_param *inparam)
 *
 * \brief Simple check on Newton input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 */
SHORT newton_param_check (newton_param *inparam)
{
    SHORT status = FASP_SUCCESS;

    if ( inparam->max_it<1
        || inparam->adapt_tol<0.0
        || inparam->tol<0.0
        || inparam->damp_factor<0.0
        || inparam->damp_it<0
        ) status = ERROR_INPUT_PAR;

    return status;
}


/**
 * \fn void newton_param_input (const char *filenm,
 *                    	 		newton_param *inparam)
 *
 * \brief Read in parameters for Newton solver
 *
 * \param filenm 	File name for input parameters
 * \param inparam	Input parameters
 */
void newton_param_input (const char *filenm,
                      	 newton_param *inparam)
{
	char     buffer[500]; // Note: max number of char for each line!
    int      val;
    SHORT    status = FASP_SUCCESS;

    // set default input parameters
    newton_param_input_init(inparam);

    // if input file is not specified, use the default values
    if (filenm==NULL) return;

    FILE *fp = fopen(filenm,"r");
    if (fp==NULL) {
        printf("### ERROR: Could not open file %s...\n", filenm);
        fasp_chkerr(ERROR_OPEN_FILE, "newton_param_input");

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
        if (strcmp(buffer,"adapt_tol")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->adapt_tol = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"nonlin_tol")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->tol = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"nonlin_maxit")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->max_it = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"nonlin_damp_factor")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->damp_factor = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"nonlin_damp_it")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->damp_it = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"max_cells")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->max_cells = int(dbuff+0.5);
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else {
            printf("### WARNING: Unknown input keyword %s!\n", buffer);
            fgets(buffer,500,fp); // skip rest of line
        }


    }

    fclose(fp);

    // sanity checks
    status = newton_param_check(inparam);

#if DEBUG_MODE > 1
    printf("### DEBUG: Reading input status = %d\n", status);
#endif

    // if meet unexpected input, stop the program
    fasp_chkerr(status,"newton_param_input");
}

/**
 * \fn void print_newton_param (newton_param *inparam)
 * \brief Print Newton solver params
 * \param inparam  Input parameters
 */
void print_newton_param (newton_param *inparam)
{
  printf("\tSuccessfully read-in Newton solver parameters\n");
  printf("\tNewton Maximum iterations:          \t%d\n",inparam->max_it);
  printf("\tNewton tolerance:                   \t%e\n",inparam->tol);
  printf("\tNewton damping factor:              \t%f\n",inparam->damp_factor);
  printf("\tNewton maximum damping iteration:   \t%d\n",inparam->damp_it);
  printf("\tNewton number of cells:             \t%d\n",inparam->max_cells);
  printf("\n");
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
SHORT domain_param_input_init (domain_param *inparam)
{
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
SHORT domain_param_check (domain_param *inparam)
{
    SHORT status = FASP_SUCCESS;

    if ( inparam->ref_length < 0.0
        || inparam->length_x < 0.0
        || inparam->length_y < 0.0
        || inparam->length_z < 0.0
        || inparam->length_time < 0.0
        || inparam->grid_x < 2
        || inparam->grid_y < 2
        || inparam->grid_z < 2
        || inparam->grid_time < 0
        ) status = ERROR_INPUT_PAR;

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

#if DEBUG_MODE > 1
    printf("### DEBUG: Reading input status = %d\n", status);
#endif

    // if meet unexpected input, stop the program
    fasp_chkerr(status,"domain_param_input");
}

/**
 * \fn void print_domain_param (domain_param *inparam)
 * \brief Print domain params
 * \param inparam  Input parameters
 */
void print_domain_param (domain_param *inparam)
{
  printf("\tSuccessfully read-in domain parameters\n");
  if ( strcmp(inparam->mesh_file,"none")==0 ) {
    printf("\tThe reference length is %e meters\n",inparam->ref_length);
    printf("\tDomain: %f x %f x %f\n",  inparam->length_x,inparam->length_y,inparam->length_z);
    printf("\tGrid:   %d x %d x %d\n\n",inparam->grid_x,inparam->grid_y,inparam->grid_z);
    fflush(stdout);
  } else {
    printf("\tThe reference length is %e meters\n",inparam->ref_length);
    printf("\tMesh file:      %s\n",  inparam->mesh_file);
    printf("\tSubdomain file: %s\n",  inparam->subdomain_file);
    printf("\tSurface file:   %s\n\n",inparam->surface_file);
    fflush(stdout);
  }
}



/**
 * \fn SHORT coeff_param_input_init (coeff_param *inparam)
 *
 * \brief Initialize coeff input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if successed; otherwise, error information.
 */
SHORT coeff_param_input_init (coeff_param *inparam)
{
    SHORT status = FASP_SUCCESS;

    inparam->temperature = 2.98e+2;
    inparam->bc_coordinate = 0;

    inparam->relative_permittivity = 1.0;

    inparam->cation_diffusivity = 1.0;
    inparam->cation_mobility = 1.0;
    inparam->cation_valency = 1.0;

    inparam->anion_diffusivity = 1.0;
    inparam->anion_mobility = 1.0;
    inparam->anion_valency = -1.0;



    return status;
}

/**
 * \fn SHORT coeff_param_check (coeff_param *inparam)
 *
 * \brief Simple check on coeff input parameters
 *
 * \param inparam   Input parameters
 *
 * \return          FASP_SUCCESS if success; otherwise, error information.
 */
SHORT coeff_param_check (coeff_param *inparam)
{
    SHORT status = FASP_SUCCESS;

    if ( inparam->ref_voltage < 0.0
        || inparam->temperature < 0.0
        || inparam->bc_coordinate < 0
        || inparam->relative_permittivity < 0.0
        || inparam->cation_upper_val < 0.0
        || inparam->cation_lower_val < 0.0
        || inparam->cation_diffusivity < 0.0
        || inparam->cation_mobility < 0.0
        || inparam->anion_upper_val < 0.0
        || inparam->anion_lower_val < 0.0
        || inparam->anion_diffusivity < 0.0
        || inparam->anion_mobility < 0.0
        ) status = ERROR_INPUT_PAR;

    return status;
}


/**
 * \fn void coeff_param_input (const char *filenm,
 *                             coeff_param *inparam)
 *
 * \brief Read in parameters for coeff solver
 *
 * \param filenm    File name for input parameters
 * \param inparam   Input parameters
 */
void coeff_param_input (const char *filenm,
                        coeff_param *inparam)
{
    char     buffer[500]; // Note: max number of char for each line!
    int      val;
    SHORT    status = FASP_SUCCESS;

    // set default input parameters
    coeff_param_input_init(inparam);

    // if input file is not specified, use the default values
    if (filenm==NULL) return;

    FILE *fp = fopen(filenm,"r");
    if (fp==NULL) {
        printf("### ERROR: Could not open file %s...\n", filenm);
        fasp_chkerr(ERROR_OPEN_FILE, "coeff_param_input");

    }

    while ( status == FASP_SUCCESS ) {
        unsigned int ibuff;
        double  dbuff;
        char   *fgetsPtr;

        val = fscanf(fp,"%s",buffer);
        if (val==EOF) break;
        if (val!=1){ status = ERROR_INPUT_PAR; break; }
        if (buffer[0]=='[' || buffer[0]=='%' || buffer[0]=='|') {
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
            continue;
        }

        // match keyword and scan for value
        if (strcmp(buffer,"ref_voltage")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->ref_voltage = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"ref_density")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->ref_density = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"bc_coordinate")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->bc_coordinate = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"temperature")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->temperature = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"potential_lower_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->potential_lower_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"potential_upper_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->potential_upper_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"relative_permittivity")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->relative_permittivity = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cation_lower_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->cation_lower_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cation_upper_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->cation_upper_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cation_diffusivity")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->cation_diffusivity = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cation_mobility")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->cation_mobility = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cation_valency")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->cation_valency = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"anion_lower_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->anion_lower_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"anion_upper_val")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->anion_upper_val = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"anion_diffusivity")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->anion_diffusivity = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"anion_mobility")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->anion_mobility = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"anion_valency")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                status = ERROR_INPUT_PAR; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { status = ERROR_INPUT_PAR; break; }
            inparam->anion_valency = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else {
            printf("### WARNING: Unknown input keyword %s!\n", buffer);
            fgets(buffer,500,fp); // skip rest of line
        }


    }

    fclose(fp);

    // sanity checks
    status = coeff_param_check(inparam);

#if DEBUG_MODE > 1
    printf("### DEBUG: Reading input status = %d\n", status);
#endif

    // if meet unexpected input, stop the program
    fasp_chkerr(status,"coeff_param_input");
}

/**
 * \fn void print_coeff_param (coeff_param *inparam)
 * \brief Print coeff params
 * \param inparam  Input parameters
 */
void print_coeff_param (coeff_param *inparam)
{
  printf("\tSuccessfully read-in PDE coefficients\n");
  printf("\treference voltage:        \t%15.4e V\n",inparam->ref_voltage);
  if (inparam->ref_density > 0.0)
    printf("\treference density:        \t%15.4e 1 / m^3\n",inparam->ref_density);
  else
    printf("\treference density:        \t%15.4e mM\n",-inparam->ref_density);
  printf("\ttemperature:              \t%15.4e K\n",inparam->temperature);
  printf("\trelative permittivity:    \t%15.4e\n",inparam->relative_permittivity);
  printf("\tcation diffusivity:       \t%15.4e m / s^2\n",inparam->cation_diffusivity);
  printf("\tcation mobility:          \t%15.4e (e_c/k_B*T) * m / s^2 \n",inparam->cation_mobility);
  printf("\tcation valency:           \t%15.4e e_c\n",inparam->cation_valency);
  printf("\tanion diffusivity:        \t%15.4e m / s^2\n",inparam->anion_diffusivity);
  printf("\tanion mobility:           \t%15.4e (e_c/k_B*T) * m / s^2 \n",inparam->anion_mobility);
  printf("\tanion valency:            \t%15.4e e_c\n",inparam->anion_valency);
  printf("\tvoltage in x[%d]:         \t%15.4e to %15.4e\n",
    inparam->bc_coordinate, inparam->potential_lower_val, inparam->potential_upper_val);
  printf("\tcation density in x[%d]:  \t%15.4e to %15.4e\n",
    inparam->bc_coordinate, inparam->cation_lower_val, inparam->cation_upper_val);
  printf("\tanion density in x[%d]:   \t%15.4e to %15.4e\n",
    inparam->bc_coordinate, inparam->anion_lower_val, inparam->anion_upper_val);
  printf("\n");
}

dolfin::Mesh domain_build(const domain_param &domain) {
  if (isnan(domain.length_x) || isnan(domain.length_y) || isnan(domain.length_z)) {
    printf("### WARNING : invalid domain lengths!\n");
    return dolfin::Mesh();
  }

  if (isnan(domain.length_x) || isnan(domain.length_y) || isnan(domain.length_z)) {
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