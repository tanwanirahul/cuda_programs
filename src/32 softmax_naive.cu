
#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"
#include <math.h>


#define EPS 0.00001


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
 * A naive implementation of the Softmax function. Each 
 * thread is reponsible for computing the output row 
 * which requires loading of the entire row of data
 * for normalization purposes. The degree of parallelism is limited by
 * the number of rows. Furthermore, since each thread need to read the entire
 * row of the data, it places too much pressure on the memory.
 * 
 * Given the M*N matrix mat, computes the softmax and stores the output
 * at the memory location pointed to by result.
 */
__global__ void softmax_kernel(float *mat, float *result, unsigned int M, unsigned int N) {

    unsigned int threadNum = (blockIdx.x * blockDim.x + threadIdx.x);

    if(threadNum < M) {

        // Each thread is responsible for entire row of N elements.
        unsigned int rowStart = threadNum * N;

        // Find the max for the row.
        float rowMax = -INFINITY;
        float rowSum = 0.0;

        for(unsigned int i=0; i<N; i++)
        {
            float current = mat[rowStart + i];
            if( current > rowMax) {
                rowMax = current;
            }            
        }

        // compute numerically stable exponents and the row sum.
        for(unsigned int i=0; i<N; i++)
        {
            float current = mat[rowStart + i];
            float elementExp = exp(current - rowMax);
            rowSum += elementExp;
            result[rowStart + i] = elementExp;            
        }

        // normalize the exponent variable by the row sum.
        rowSum = (rowSum == 0.0)? EPS : rowSum;
        for(unsigned int i=0; i<N; i++)
            result[rowStart + i] = result[rowStart + i] / rowSum;
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
        printf("\nfailed to allocated memory on CUDA device for input matrix.");
        return;
    }
    error = cudaMalloc((void **) &result_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for result matrix.");
        return;
    }
    cudaDeviceSynchronize();
    printf("Allocated required memory on the CUDA device.\n\n");
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);
    

    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);
    
    cudaMemcpy(mat_d, mat, mat_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);


    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);

    unsigned int threadsPerBlock = 1024;
    unsigned int numBlocks = ( M + threadsPerBlock - 1) / threadsPerBlock; 
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
    unsigned int N = 10240;

    size_t mat_size = sizeof(float) * N * N;

    // A - N*N matrix with Random values clipped by 10
    Matrix A = random_clipped_matrix_2D(N , N, 10);

    printf("Initialized an input matrix\n");

    // Allocate memory on the host for holding resulting matrix.
    float * result_CPU = (float*) malloc(mat_size);
    float * result_GPU = (float*) malloc(mat_size);

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    softmax_CPU(A.buffer, result_CPU, N, N);
    stopAndPrintElapsed(&timer, "CPU Execution Time: ", CYAN);
    

    // Convert the image to grey;
    timer = initTimer(1);
    startTimer(&timer);
    softmax_wrapper(A.buffer, result_GPU, N, N);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU Execution Time: ", GREEN);

    Matrix cpuResult, gpuResult;
    cpuResult.rows = N;
    cpuResult.cols = N;
    cpuResult.buffer = result_CPU;

    gpuResult.rows = N;
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