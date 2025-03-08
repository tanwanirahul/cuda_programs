#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"
#include <math.h>

#define EPS 0.00001
#define THREADS_PER_BLOCK 1024
#define NUM_ROWS 10240
#define NUM_COLS 10240


void softmax_CPU(float *mat, float *result, unsigned int M, unsigned int N) {

    for(unsigned int i = 0; i<M; i++) {
        float rowMax = -INFINITY;
        float rowSum = 0.0;

        // Find RowMax.
        for(unsigned int j = 0; j<N; j++) {
            float current = mat[(i * N) + j];
            if(current > rowMax)
                rowMax = current;
        }

        // Find Exponents and RowSum
        for(unsigned int j = 0; j<N; j++) {
            float current = mat[(i * N) + j];
            float elementExp = exp(current - rowMax);
            result[(i*N) + j] = elementExp;
            rowSum+= elementExp;
        }

        // Normalize by RowSum;
        rowSum = (rowSum == 0.0)? EPS : rowSum;
        for(unsigned int j = 0; j<N; j++)
            result[(i*N) + j] = result[(i*N) + j] / rowSum;
    }
}

/**
 * An imoproved implementation of Softmax by utilizing the shared memory effectively.
 * Multiple threads in a block fetch the data from Global Memory to Shared Memory.
 * One thread performs the sequential operation of find out the max and the row sum.
 * Multiple threads cooperate to normalize the data by row sum and putting the data
 * back to global memory.
 * 
 * Given the M*N matrix mat, computes the softmax and stores the output
 * at the memory location pointed to by result.
 */
__global__ void softmax_kernel(float *mat, float *result, unsigned int M, unsigned int N) {

    // Each block is responsible for loading N elements.
    unsigned int rowStart = (blockIdx.x * N);

    // Shared memory to contain the row Data.
    __shared__ float rowData[NUM_COLS];
    __shared__ float rowSum_s;

    // Load the data into shared memory from the input matrix - mat.
    for(unsigned int i=0; i<( N  + blockDim.x - 1)/blockDim.x; i++) {

        unsigned int elementToLoad = (i * blockDim.x) + threadIdx.x;
        if(elementToLoad < N) {
            rowData[elementToLoad] = mat[rowStart + elementToLoad];
        }
    }
    // Make sure all threads have completed loading data into the shared memory.
    __syncthreads();
    
    // Only the first thread in the block is responsible for
    // finding the rowMax and rowSum;
    if(threadIdx.x == 0) {

        // Find the max for the row using the data loaded in shared memory.
        float rowMax = -INFINITY;
        float rowSum = 0.0;
        for(unsigned int i=0; i<N; i++)
        {
            float current = rowData[i];
            if( current > rowMax) {
                rowMax = current;
            }
        }

        // compute numerically stable exponents and the row sum.
        for(unsigned int i=0; i<N; i++)
        {
            float current = rowData[i];
            float elementExp = exp(current - rowMax);
            rowSum += elementExp;
            rowData[i] = elementExp;
        }

        rowSum_s = (rowSum == 0.0)? EPS : rowSum;
    }
    // Make sure the rowMax and rowSum operations are done before proceeding.
    __syncthreads();

    // normalize the exponent variable by the row sum and write
    // the normalized results back to global memory.
    for(unsigned int i=0; i<(N + blockDim.x -1)/blockDim.x; i++) {
        unsigned int elementToWrite = (i * blockDim.x) + threadIdx.x;
        if(elementToWrite < N) {
            result[rowStart + elementToWrite] = rowData[elementToWrite] / rowSum_s;
        }
        
    }
}

void softmax_wrapper(float *mat, float * result, unsigned int M, unsigned int N) {
    // Allocate memory on the GPU device.
    float *mat_d, *result_d;
    size_t mat_size = sizeof(float) * M * N;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaError_t error;
    error = cudaMalloc((void **) &mat_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocate memory on CUDA device for input matrix.\n");
        return;
    }
    error = cudaMalloc((void **) &result_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocate memory on CUDA device for result matrix.\n");
        return;
    }
    cudaDeviceSynchronize();
    printf("Allocated required memory on the CUDA device.\n\n");
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);


    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);

    error = cudaMemcpy(mat_d, mat, mat_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("\nfailed to copy input matrix on CUDA device.");
        return;
    }
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);


    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);

    unsigned int threadsPerBlock = (THREADS_PER_BLOCK < N)? THREADS_PER_BLOCK: N;
    unsigned int numBlocks = M;
    
    softmax_kernel <<<numBlocks, threadsPerBlock>>>(mat_d, result_d, M, N);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);

    cudaMemcpy(result, result_d, mat_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree((void*) mat_d);
    cudaFree((void*) result_d);
    cudaDeviceSynchronize();

}

int main(int argc, char** argv) {

    //unsigned int N = 40000;
    unsigned int N = NUM_COLS;
    unsigned int M = NUM_ROWS;

    size_t mat_size = sizeof(float) * M * N;

    // A - N*N matrix with Random values clipped by 10
    Matrix A = random_clipped_matrix_2D(M , N, 10);

    printf("Initialized an input matrix\n");

    // Allocate memory on the host for holding resulting matrix.
    float * result_CPU = (float*) malloc(mat_size);
    float * result_GPU = (float*) malloc(mat_size);

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    softmax_CPU(A.buffer, result_CPU, M, N);
    stopAndPrintElapsed(&timer, "CPU Execution Time: ", CYAN);


    // Convert the image to grey;
    timer = initTimer(1);
    startTimer(&timer);
    softmax_wrapper(A.buffer, result_GPU, M, N);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU Execution Time: ", GREEN);

    Matrix cpuResult, gpuResult;
    cpuResult.rows = M;
    cpuResult.cols = N;
    cpuResult.buffer = result_CPU;

    gpuResult.rows = M;
    gpuResult.cols = N;
    gpuResult.buffer = result_GPU;

    bool areEqual = are_matrix_close(&cpuResult, &gpuResult, 0.00001f);
    printf("\nDo results from CPU and GPU implementation match? %d\n", areEqual);

    // Manually analyze the first few elements of the matrix to compare results
    int limit = (N < 1024)? N : 1024;
    printf("\nResults of Softmax for a row (Input   CPU     GPU): \n");
    for(int i =0; i<limit; i++) {
      printf("%d: %20.6f  %20.6f  %20.6f\n", i, A.buffer[i], result_CPU[i], result_GPU[i]);
    }

    // Free up allocated space on the host Matrix A, B and buffer holding the results.
    release_matrix(&A);
    free(result_CPU);
    free(result_GPU);

    return 0;
}