#ifndef _MATRIX_UTILS_H_
#define _MATRIX_UTILS_H_

#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

typedef struct MatrixInfo {
    unsigned int rows;
    unsigned int cols;
    float * buffer;
} Matrix;


typedef struct MatrixInfo3D {
    unsigned int depth;
    unsigned int rows;
    unsigned int cols;
    float * buffer;
} Matrix3D;


typedef struct MatrixInfo1D {
    unsigned int length;
    float * buffer;
} Matrix1D;


typedef struct SparseMatrixInfo2DCOO {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonZeros;
    unsigned int * rowIdxs;
    unsigned int * colIdxs;
    float * values;
} SparseMatrix2DCOO;


typedef struct SparseMatrixInfo2DCSR {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonZeros;
    unsigned int * rowPtrs;
    unsigned int * colIdxs;
    float * values;
} SparseMatrix2DCSR;

// Declarations.
bool release_matrix(Matrix * mat);
Matrix identity_matrix_2D(unsigned int N);

/**
 * Given the Matrix2D format, converts it into the Sparse matrix
 * CSR format.
 */
SparseMatrix2DCSR _convert_2D_matrix_to_sparse_CSR(Matrix *mat, unsigned int nonZeroElements) {
    SparseMatrix2DCSR csr_matrix;
    csr_matrix.colIdxs = (unsigned int *) malloc(sizeof(unsigned int) * nonZeroElements);
    csr_matrix.rowPtrs = (unsigned int *) malloc(sizeof(unsigned int) * (mat->rows+1));
    csr_matrix.values = (float *) malloc(sizeof(float) * nonZeroElements);
    csr_matrix.numNonZeros = nonZeroElements;
    csr_matrix.numRows = mat->rows;
    csr_matrix.numCols = mat->cols;

    // First row begins at the 0th location.
    csr_matrix.rowPtrs[0] = 0;

    unsigned int nnz_ctr = 0.0;
    for(int i = 0; i < csr_matrix.numRows && nnz_ctr < nonZeroElements; i++) {
        unsigned int nonZeroCols = 0.0;
        for(int j = 0; j < csr_matrix.numCols && nnz_ctr < nonZeroElements; j++) {
            float val = mat->buffer[(i*csr_matrix.numRows)+j];
            if(val != 0.0) {
                csr_matrix.values[nnz_ctr] = val;
                csr_matrix.colIdxs[nnz_ctr] = j;
                nonZeroCols++;
                nnz_ctr++;
            } 
        }
        csr_matrix.rowPtrs[i+1] = csr_matrix.rowPtrs[i] + nonZeroCols;
    }
    return csr_matrix;
}

/**
 * Given the Matrix2D format, converts it into the Sparse matrix
 * COO format.
 */
SparseMatrix2DCOO _convert_2D_matrix_to_sparse_COO(Matrix *mat, unsigned int nonZeroElements) {
    SparseMatrix2DCOO coo_matrix;
    coo_matrix.colIdxs = (unsigned int *) malloc(sizeof(unsigned int) * nonZeroElements);
    coo_matrix.rowIdxs = (unsigned int *) malloc(sizeof(unsigned int) * nonZeroElements);
    coo_matrix.values = (float *) malloc(sizeof(float) * nonZeroElements);
    coo_matrix.numNonZeros = nonZeroElements;
    coo_matrix.numRows = mat->rows;
    coo_matrix.numCols = mat->cols;

    unsigned int nnz_ctr = 0.0;
    for(int i = 0; i < coo_matrix.numRows && nnz_ctr < nonZeroElements; i++) {
        for(int j = 0; j < coo_matrix.numCols && nnz_ctr < nonZeroElements; j++) {
            float val = mat->buffer[(i*coo_matrix.numRows)+j];
            if(val != 0.0) {
                coo_matrix.values[nnz_ctr] = val;
                coo_matrix.rowIdxs[nnz_ctr] = i;
                coo_matrix.colIdxs[nnz_ctr] = j;
                nnz_ctr++;
            } 
        }
    }
    return coo_matrix;

}

/**
 * Creates a 2D sparse matrix in CSR format with values set to 1 or 0. The number of zeros depneds on
 * the desired sparsity factor.
 */
SparseMatrix2DCSR ones_sparse_matrix_2D_CSR(float sparsityFactor, unsigned int rows, unsigned int cols) {

    Matrix mat;
    unsigned int nonZeroElements = 0;

    srand(1000);
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        float generatedProb = ((float) rand() / RAND_MAX);
        if(generatedProb <= sparsityFactor) {
            mat.buffer[i] = 0.0;
        } else {
            mat.buffer[i] = 1.0;
            nonZeroElements++;
        }
    }
    mat.rows = rows;
    mat.cols = cols;

    SparseMatrix2DCSR csr_matrix = _convert_2D_matrix_to_sparse_CSR(&mat, nonZeroElements);
    release_matrix(&mat);
    return csr_matrix;
}


/**
 * Creates a 2D sparse matrix in CSR format with values set to a random value or 0. The number of zeros depneds on
 * the desired sparsity factor.
 */
SparseMatrix2DCSR random_sparse_matrix_2D_CSR(float sparsityFactor, unsigned int rows, unsigned int cols) {

    Matrix mat;
    unsigned int nonZeroElements = 0;

    srand(5005);
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        float generatedProb = ((float) rand() / RAND_MAX);
        if(generatedProb <= sparsityFactor) {
            mat.buffer[i] = 0.0;
        } else {
            mat.buffer[i] = rand();
            nonZeroElements++;
        }
    }
    mat.rows = rows;
    mat.cols = cols;

    SparseMatrix2DCSR csr_matrix = _convert_2D_matrix_to_sparse_CSR(&mat, nonZeroElements);
    release_matrix(&mat);
    return csr_matrix;
}

/**
 * Creates a 2D identity matrix in the Sparse CSR    format.
 */
SparseMatrix2DCSR identity_sparse_matrix_2D_CSR(unsigned int N) {

    Matrix mat = identity_matrix_2D(N);
    SparseMatrix2DCSR csr_matrix = _convert_2D_matrix_to_sparse_CSR(&mat, N);
    release_matrix(&mat);
    return csr_matrix;
}


/**
 * Creates a 2D sparse matrix with values set to 1 or 0. The number of zeros depneds on
 * the desired sparsity factor.
 */
SparseMatrix2DCOO ones_sparse_matrix_2D_COO(float sparsityFactor, unsigned int rows, unsigned int cols) {

    Matrix mat;
    unsigned int nonZeroElements = 0;

    srand(1000);
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        float generatedProb = ((float) rand() / RAND_MAX);
        if(generatedProb <= sparsityFactor) {
            mat.buffer[i] = 0.0;
        } else {
            mat.buffer[i] = 1.0;
            nonZeroElements++;
        }
    }
    mat.rows = rows;
    mat.cols = cols;

    SparseMatrix2DCOO coo_matrix = _convert_2D_matrix_to_sparse_COO(&mat, nonZeroElements);
    release_matrix(&mat);
    return coo_matrix;
}


/**
 * Creates a 2D sparse matrix with values set to a random value or 0. The number of zeros depneds on
 * the desired sparsity factor.
 */
SparseMatrix2DCOO random_sparse_matrix_2D_COO(float sparsityFactor, unsigned int rows, unsigned int cols) {

    Matrix mat;
    unsigned int nonZeroElements = 0;

    srand(5005);
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        float generatedProb = ((float) rand() / RAND_MAX);
        if(generatedProb <= sparsityFactor) {
            mat.buffer[i] = 0.0;
        } else {
            mat.buffer[i] = rand();
            nonZeroElements++;
        }
    }
    mat.rows = rows;
    mat.cols = cols;

    SparseMatrix2DCOO coo_matrix = _convert_2D_matrix_to_sparse_COO(&mat, nonZeroElements);
    release_matrix(&mat);
    return coo_matrix;
}

/**
 * Creates a 2D identity matrix in the Sparse COO format.
 */
SparseMatrix2DCOO identity_sparse_matrix_2D_COO(unsigned int N) {

    Matrix mat = identity_matrix_2D(N);
    SparseMatrix2DCOO coo_matrix = _convert_2D_matrix_to_sparse_COO(&mat, N);
    release_matrix(&mat);
    return coo_matrix;
}

/**
 * Creates a 1D matrix of length elements with
 * all values set to 1.0;
 */
Matrix1D ones_matrix_1D(unsigned int length) {

    Matrix1D mat;
    mat.buffer = (float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        mat.buffer[i] = 1.0;
    }
    mat.length = length;
    return mat;
}

/**
 * Creates a 1D matrix of length elements with
 * all values set to 0.0;
 */
Matrix1D zeros_matrix_1D(unsigned int length) {

    Matrix1D mat;
    mat.buffer = (float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        mat.buffer[i] = 0.0;
    }
    mat.length = length;
    return mat;
}

/**
 * Creates a 1D matrix of length elements with
 * all values set to a random value;
 */
Matrix1D random_matrix_1D(unsigned int length) {

    Matrix1D mat;
    mat.buffer = (float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        mat.buffer[i] = rand();;
    }
    mat.length = length;
    return mat;
}

/**
 * Creates a 1D matrix of length elements with
 * all values set to a random value;
 */
Matrix1D random_clipped_matrix_1D(unsigned int length, unsigned int maxValue) {

    Matrix1D mat;
    mat.buffer = (float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        mat.buffer[i] = rand() % (maxValue + 1);
    }
    mat.length = length;
    return mat;
}

/**
 * Creates a 1D matrix of length elements with
 * all values sorted and set to a random value with the delta
 * between two values not exceeding delta.
 */
Matrix1D random_sorted_matrix_1D(unsigned int length, unsigned int delta) {

    Matrix1D mat;
    mat.buffer = (float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        if(i==0)
            mat.buffer[i] = rand() % (delta + 1);    
        else
            mat.buffer[i] = mat.buffer[i -1] + (rand() % (delta + 1));
    }
    mat.length = length;
    return mat;
}


/**
 * Returns if the 1D matrix is sorted. 
 */
bool is_matrix_sorted_1D(Matrix1D *A) {
    for(unsigned int i =0; i<A->length - 1; i++) {
        if(A->buffer[i+1] < A->buffer[i])
            return false;
    }
    return true;
}
/**
 *  Given the two Matrix, compares their elements and returns true
 * if all the elements are equal, false otherwise.
 */
bool are_matrix_equal_1D(Matrix1D *A, Matrix1D *B ) {

    if(A->length != B->length)
        return false;
    for(unsigned int i=0; i<A->length; i++) {
        if(A->buffer[i] != B->buffer[i])
            return false;
    }
    return true;
}

/**
 * Creates a 2D matrix of rows*cols in the Row Major format with
 * all values set to 1.0;
 */
Matrix ones_matrix_2D(unsigned int rows, unsigned int cols) {

    Matrix mat;
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        mat.buffer[i] = 1.0;
    }
    mat.rows = rows;
    mat.cols = cols;
    return mat;

}

/**
 * Creates a 2D matrix of rows*cols in the Row Major format with
 * all values set to 0.0;
 */
Matrix zeros_matrix_2D(unsigned int rows, unsigned int cols) {

    Matrix mat;
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        mat.buffer[i] = 0.0;
    }
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}


/**
 * Creates a 2D matrix of rows*cols in the Row Major format with
 * all values set to a random value between 0 and maxValue;
 */
Matrix random_clipped_matrix_2D(unsigned int rows, unsigned int cols, unsigned int maxValue) {

    Matrix mat;
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        mat.buffer[i] = rand() % (maxValue + 1);
    }
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}

/**
 * Creates a 2D matrix of rows*cols in the Row Major format with
 * all values set to a random value;
 */
Matrix random_matrix_2D(unsigned int rows, unsigned int cols) {

    Matrix mat;
    mat.buffer = (float *) malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++) {
        mat.buffer[i] = rand();
    }
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}


/**
 * Creates a 2D identity matrix of N*N in the Row Major format with
 * diagnoal values set to 1.0.
 */
Matrix identity_matrix_2D(unsigned int N) {

    Matrix mat;
    mat.buffer = (float *) malloc(N * N * sizeof(float));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            mat.buffer[(i*N)+j] = (i==j) ? 1.0 : 0.0;
    }
    mat.rows = N;
    mat.cols = N;
    return mat;

}

/**
 *  Given the two Matrix, compares their elements and returns true
 * if all the elements are equal, false otherwise.
 */
bool are_matrix_equal(Matrix *A, Matrix *B ) {

    if(A->rows != B->rows)
        return false;
    if(A->cols != B->cols)
        return false;
    for(unsigned int i=0; i<A->rows; i++) {
        for(unsigned int j=0; j<A->cols; j++)
            if(A->buffer[(i*A->cols) +j] != B->buffer[(i*B->cols) +j])
                return false;
    }
    return true;
}

/**
 *  Given the two 3D Matrix, compares their elements and returns true
 * if all the elements are equal, false otherwise.
 */
bool are_matrix_equal_3D(Matrix3D *A, Matrix3D *B ) {

    if(A->depth != B->depth)
        return false;
    if(A->rows != B->rows)
        return false;
    if(A->cols != B->cols)
        return false;
    for(unsigned int i=0; i<A->depth; i++) {
        for(unsigned int j=0; j<A->rows; j++){
            for(unsigned int k=0; k<A->cols; k++){
                unsigned int index = (i*A->rows*A->cols) + (j*A->cols) + k;
                if(A->buffer[index] != B->buffer[index])
                    return false;
            }
        }
    }
    return true;
}


/**
 * Creates a 3D matrix of depth*rows*cols in the depth-wise Row Major format with
 * all values set to 1.0;
 */
Matrix3D ones_matrix_3D(unsigned int depth, unsigned int rows, unsigned int cols) {

    Matrix3D mat;
    mat.buffer = (float *) malloc(depth * rows * cols * sizeof(float));
    for(int i = 0; i < depth * rows * cols; i++) {
        mat.buffer[i] = 1.0;
    }
    mat.depth = depth;
    mat.rows = rows;
    mat.cols = cols;
    return mat;

}

/**
 * Creates a 3D matrix of depth*rows*cols in the depthwise Row Major format with
 * all values set to 0.0;
 */
Matrix3D zeros_matrix_3D(unsigned int depth, unsigned int rows, unsigned int cols) {

    Matrix3D mat;
    mat.buffer = (float *) malloc(depth * rows * cols * sizeof(float));
    for(int i = 0; i < depth * rows * cols; i++) {
        mat.buffer[i] = 0.0;
    }
    mat.depth = depth;
    mat.rows = rows;
    mat.cols = cols;

    return mat;
}


/**
 * Creates a 3D matrix of depth*rows*cols in the depth-wise Row Major format with
 * all values set to a random value;
 */
Matrix3D random_matrix_3D(unsigned int depth, unsigned int rows, unsigned int cols) {

    Matrix3D mat;
    mat.buffer = (float *) malloc(depth * rows * cols * sizeof(float));
    for(int i = 0; i < depth * rows * cols; i++) {
        mat.buffer[i] = rand();
    }
    mat.depth = depth;
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}



/**
 * Releases the dynamically allocated memory for the metrix.
 */
bool release_matrix(Matrix * mat) {
    free(mat->buffer);
    return true;
}

/**
 * Releases the dynamically allocated memory for the metrix.
 */
bool release_matrix_3D(Matrix3D * mat) {
    free(mat->buffer);
    return true;
}

/**
 * Releases the dynamically allocated memory for the metrix.
 */
bool release_matrix_1D(Matrix1D * mat) {
    free(mat->buffer);
    return true;
}

/**
 * Releases the Sparse 2D Matrix represented in the COO format.
 */
bool release_sparse_matrix_2D_COO(SparseMatrix2DCOO * matrix) {
    free(matrix->rowIdxs);
    free(matrix->colIdxs);
    free(matrix->values);
    return true;
}

/**
 * Releases the Sparse 2D Matrix represented in the COO format.
 */
bool release_sparse_matrix_2D_CSR(SparseMatrix2DCSR * matrix) {
    free(matrix->rowPtrs);
    free(matrix->colIdxs);
    free(matrix->values);
    return true;
}

#endif