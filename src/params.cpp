/*! \file params.cpp
 *
 * \brief Contains files for reading in *.dat parameter files
 *        and conversion to structs defined in ./../include/newton.h
 */

#include <stdio.h>
#include "../include/newton.h"
#include "../include/newton_functs.h"
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
        ) status = ERROR_INPUT_PAR;
    
    return status;
}


/**
 * \fn void newton_param_input (const char *filenm,
                      	 		newton_param *inparam)
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

    if ( inparam->length_x < 0.0
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
                                domain_param *inparam)
 *
 * \brief Read in parameters for domain solver
 *
 * \param filenm    File name for input parameters
 * \param inparam   Input parameters
 */
void domain_param_input (const char *filenm,
                         domain_param *inparam)
{
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
        if (strcmp(buffer,"length_x")==0) {
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

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
