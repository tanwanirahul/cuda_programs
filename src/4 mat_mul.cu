%%writefile mat_mul.cu

#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

__global__ void mat_mul_kernel(float *A, float *B, float *C, unsigned int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N and col < N) {
        int i = row * N + col;
        for(unsigned int j=0; j < N; j++) {
            C[i]+= A[row*N + j] * B[j * N + col];
        }
    }
}

void mat_mul_wrapper(float * A, float * B, float * C, unsigned int N) {
    // Allocate memory on the GPU device.
    float *A_d, *B_d, *C_d;
    size_t mat_size = sizeof(float) * N * N;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaError_t error;
    error = cudaMalloc((void **) &A_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for matrix A.");
        return;
    }
    error = cudaMalloc((void **) &B_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for matrix B.");
        return;
    }
    error = cudaMalloc((void **) &C_d, mat_size);
    if (error != cudaSuccess) {
        printf("\nfailed to allocated memory on CUDA device for matrix C.");
        return;
    }
    cudaDeviceSynchronize();
    printf("Allocated required memory on the CUDA device.\n\n");
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);

    

    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);
    
    cudaMemcpy(A_d, A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, mat_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);


    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y); 
    mat_mul_kernel <<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);

    cudaMemcpy(C, C_d, mat_size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree((void*) A_d);
    cudaFree((void*) B_d);
    cudaFree((void*) C_d);
    cudaDeviceSynchronize();

}

int main(int argc, char** argv) {
    
    //unsigned int N = 40000;
    unsigned int N = 10240;

    size_t mat_size = sizeof(float) * N * N;

    // Create 2 matrixs:
    // A - N*N matrix with Random values
    // B - N*N Identity matrix.
    Matrix A = random_matrix_2D(N , N);
    Matrix B = identity_matrix_2D(N);

    printf("Initialized matrix A and B\n");
    // Allocate memory on the host for holding resulting matrix.
    float * C = (float*) malloc(mat_size);

    // Convert the image to grey;
    mat_mul_wrapper(A.buffer, B.buffer, C, N);
    cudaDeviceSynchronize();


    // Manually analyze the first few elements of the matrix to compare results
    int limit = (N < 256)? N : 256;
    printf("\nResults of Matrix Multiplication: \n");
    for(int i =0; i<limit; i++) {
      printf("%d: %20.6f    %20.6f\n", i, A.buffer[i], C[i]);
    }

    // Free up allocated space on the host Matrix A, B and buffer holding the results.
    release_matrix(&A);
    release_matrix(&B);
    free(C);

    return 0;
}