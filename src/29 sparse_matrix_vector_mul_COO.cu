#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

#define BLOCK_SIZE 1024

__global__ void mat_vec_mul_kernel(SparseMatrix2DCOO mat, float *vector, float *result) {

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < mat.numNonZeros) {
        unsigned int row = mat.rowIdxs[i];
        unsigned int col = mat.colIdxs[i];
        float value = mat.values[i];
        atomicAdd(&result[row], (value*vector[col]));
    }
}

void mat_vec_mul_wrapper(SparseMatrix2DCOO mat, float *vector, float *result) {
    // Allocate memory on the GPU device.
    float *vector_d, *result_d;
    SparseMatrix2DCOO coo_matrix_d;

    coo_matrix_d.numRows = mat.numRows;
    coo_matrix_d.numCols = mat.numCols;
    coo_matrix_d.numNonZeros = mat.numNonZeros;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaError_t error;
    error = cudaMalloc((void **) &vector_d, mat.numCols * sizeof(float));
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for vector.");
        return;
    }
    error = cudaMalloc((void **) &result_d, mat.numRows * sizeof(float));
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for result.");
        return;
    }
    error = cudaMalloc((void **) &coo_matrix_d.values, mat.numNonZeros * sizeof(float));
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for Sparse matrix values.");
        return;
    }
    error = cudaMalloc((void **) &coo_matrix_d.rowIdxs, mat.numNonZeros * sizeof(unsigned int));
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for Sparse matrix RowIdxs.");
        return;
    }
    error = cudaMalloc((void **) &coo_matrix_d.colIdxs, mat.numNonZeros * sizeof(unsigned int));
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for Sparse matrix ColIdxs.");
        return;
    }
    cudaDeviceSynchronize();
    printf("Allocated required memory on the CUDA device.\n\n");
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);

    

    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);

    // Copy data from host to device.
    cudaMemcpy(vector_d, vector, mat.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_d, result, mat.numRows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(coo_matrix_d.values, mat.values, mat.numNonZeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(coo_matrix_d.rowIdxs, mat.rowIdxs, mat.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(coo_matrix_d.colIdxs, mat.colIdxs, mat.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);


    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);

    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (mat.numNonZeros + threadsPerBlock - 1) / threadsPerBlock;
    mat_vec_mul_kernel<<<numBlocks, threadsPerBlock>>>(coo_matrix_d, vector_d, result_d);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(result, result_d, mat.numRows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree((void*) vector_d);
    cudaFree((void*) result_d);
    cudaFree((void*) coo_matrix_d.values);
    cudaFree((void*) coo_matrix_d.rowIdxs);
    cudaFree((void*) coo_matrix_d.colIdxs);
    cudaDeviceSynchronize();

}

int main(int argc, char** argv) {
    
    //unsigned int N = 40000;
    unsigned int N = 10240;

    //Create a sparse matrix and a 1D matrix as a vector:
    SparseMatrix2DCOO A = identity_sparse_matrix_2D_COO(N);
    Matrix1D b = random_matrix_1D(A.numCols);

    // Allocate memory on the host for holding resulting matrix.
    float * result = (float*) malloc(A.numRows * sizeof(float));

    // Run the matrix vector multiplication.
    mat_vec_mul_wrapper(A, b.buffer, result);
    cudaDeviceSynchronize();


    Matrix1D result_mat;
    result_mat.length = A.numRows;
    result_mat.buffer = result;

    printf("\nDoes output match expected results? %d\n", are_matrix_equal_1D(&b, &result_mat));
    
    // Manually analyze the first few rows of the result array.
    int limit = (N < 256)? N : 256;
    printf("\nResults of Matrix Vector Multiplication: \n");
    for(int i =0; i<limit; i++) {
      printf("%d: %20.6f    %20.6f\n", i, b.buffer[i], result_mat.buffer[i]);
    }

    // Free up allocated space on the host Matrix A, B and buffer holding the results.
    release_sparse_matrix_2D_COO(&A);
    release_matrix_1D(&b);
    free(result);

    return 0;
}