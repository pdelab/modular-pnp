// http://fasp.sourceforge.net/formats_8c_source.html


/*!
 * \fn dBSRmat fasp_format_dcsr_dbsr ( dCSRmat *A, const INT nb )
 *
 * \brief Transfer a dCSRmat type matrix into a dBSRmat.
 *
 * \param A   Pointer to the dCSRmat type matrix
 * \param nb  size of each block
 *
 * \return    dBSRmat matrix
 *
 * \author  Zheng Li
 * \date    03/27/2014
 *
 * \note modified by Xiaozhe Hu to avoid potential memory leakage problem
 *
 */
dBSRmat fasp_format_dcsr_dbsr (dCSRmat *A,
                               const INT nb)
{
	// Safe-guard check
    if ((A->row)%nb!=0) {
        printf("### ERROR: A.row=%d is not a multiplication of nb=%d!\n", A->row, nb);
        exit(0);
    }

    if ((A->col)%nb!=0) {
        printf("### ERROR: A.col=%d is not a multiplication of nb=%d!\n", A->col, nb);
        exit(0);
    }

    INT i, j, k, ii, jj, kk, l, mod, nnz;
    INT row   = A->row/nb;
    INT col   = A->col/nb;
    INT nb2   = nb*nb;
    INT *IA   = A->IA;
    INT *JA   = A->JA;
    REAL *val = A->val;

    dBSRmat B;
    B.ROW = row;
    B.COL = col;
    B.nb  = nb;
    B.storage_manner = 0;

    INT *col_flag = (INT *)fasp_mem_calloc(col, sizeof(INT));

    // allocate ia for B
    INT *ia = (INT *) fasp_mem_calloc(row+1, sizeof(INT));

    fasp_iarray_set(col, col_flag, -1);

    // Get ia for BSR format
    nnz = 0;
	  for (i=0; i<row; ++i) {
        ii = nb*i;
        for(j=0; j<nb; ++j) {
            jj = ii+j;
            for(k=IA[jj]; k<IA[jj+1]; ++k) {
                kk = JA[k]/nb;
                if (col_flag[kk]!=0) {
                    col_flag[kk] = 0;
                    //ja[nnz] = kk;
                    nnz ++;
                }
			}
		}
        ia[i+1] = nnz;
        fasp_iarray_set(col, col_flag, -1);
	}

    // set NNZ
    B.NNZ = nnz;

    // allocate ja and bval
    INT *ja = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    REAL *bval = (REAL*)fasp_mem_calloc(nnz*nb2, sizeof(REAL));

    // Get ja for BSR format
    nnz = 0;
    for (i=0; i<row; ++i) {
        ii = nb*i;
        for(j=0; j<nb; ++j) {
            jj = ii+j;
            for(k=IA[jj]; k<IA[jj+1]; ++k) {
                kk = JA[k]/nb;
                if (col_flag[kk]!=0) {
                    col_flag[kk] = 0;
                    ja[nnz] = kk;
                    nnz ++;
                }
			}
		}
        ia[i+1] = nnz;
        fasp_iarray_set(col, col_flag, -1);
	}

    // Get non-zeros of BSR
	for (i=0; i<row; ++i) {
		ii = nb*i;
        for(j=0; j<nb; ++j) {
			jj = ii+j;
			for(k=IA[jj]; k<IA[jj+1]; ++k) {
				for (l=ia[i]; l<ia[i+1]; ++l) {
					if (JA[k]/nb ==ja[l]) {
                        mod = JA[k]%nb;
                        bval[l*nb2+j*nb+mod] = val[k];
                        break;
                    }
				}
			}
		}
	}

    B.IA = ia;
    B.JA = ja;
    B.val = bval;

    fasp_mem_free(col_flag);

    return B;
}
