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
 * Releases the dynamically allocated memory for the metrix.
 */
bool release_matrix(Matrix * mat) {
    free(mat->buffer);
    return true;
}

#endif