/**
 * Genralized Matrix Multiplication (GEMM) using the CuTe Tensors
 and Layout abstractions.

 C = A * B

 A: M * K matrix.
 B: N * K matrix.
 C: M * N matrix.

 
 The implementation leverages wmma instruction executed by Tensor cores 
 for matrix multiplication that is available from Volta architecture and later. 
 The code is using SM75 wmma instructions and is expected to be compiled on
 Turing architecture such as T4.

 CUDA provides warp synchronous APIs that allow threads within a warp to collectively 
 participate to load and store the data from global memory to shared memory and vice versa, 
 and perform matrix multiplication using the wmma instructions.

 SM75 supported wmma instructions for half floating point matrix multiplication include:

  - wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype d, a, b, c;

  .alayout = {.row, .col};
  .blayout = {.row, .col};
  .shape  =  {.m16n16k16, .m8n32k16, .m32n8k16};
  .dtype   = {.f16, .f32};
  .atype   = {.s8, .u8};
  .btype   = {.s8, .u8};
  .ctype   = {.f16, .f32};
 */

#include<cstdio>
#include<cstdlib>
#include<stdbool.h>

#include<cute/tensor.hpp>
#include<mma.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>

#include "matrix_utils.h"
#include "check_cuda_errors.h"
#include "timer.h"

#define DEBUG_PRINT 0
#define PRINT_MATRIX 0
#define COMPARE_WITH_CPU 1

/**
    matrix multiplication implementation on CPU to verify the results.
    A: M * K matrix.
    B: N * K matrix.
    C: M * N matrix.

 */
template<typename TA, typename TB, typename TC>
void matmul(const TA *A, const TB *B, TC *C, unsigned int M, unsigned int N, unsigned int K) {

    // Initialize the values of C to 0.
    for(unsigned int i=0; i<M; i++)
        for(unsigned int j=0; j<N; j++)
            C[(i*N)+j] = 0.0;

    // Run the matrix multiply.
    for(unsigned int i=0; i<M; i++) {
        for(unsigned int j=0; j<K; j++) {
            for(unsigned int l=0; l<N; l++) {
                C[(i*N)+ l] += A[(i*K)+j] * B[(l*K)+j];
            }
        }
    }
}

/**
    Convert the float matrix to half precision array/matrix. The conversion
    is required since there is no support for half precision in C++. 
 */
__global__ void fp32_to_fp16 (half *out, float *in, unsigned int numElements) {
   int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
   if (idx < numElements) {
      out[idx] = in[idx];
   }
}


// Warp Tile Size. Each warp will compute the 16 x 16 tile of C.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

__global__ void
//__maxnreg__(64)
gemm( __half const *A, __half const *B, float * C, 
      unsigned int ldA, unsigned int ldB, unsigned int ldC,
      unsigned int M, unsigned int N, unsigned int K) {
  

    unsigned int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    unsigned int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Create the fragements for A, B and C matrix.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    // Initialize the fragement C to 0.
    wmma::fill_fragment(frag_c, 0.0f);

    // Loop over the K-dimension and accumulate the multiplication results.
    for(unsigned int k=0; k<K; k+=WMMA_K) {

        unsigned int aRow = (warpM * WMMA_M);
        unsigned int aCol = k;
        unsigned int bRow = (warpN * WMMA_N);
        unsigned int bCol = k;

        if(aRow < M && aCol < K && bRow < N && bCol < K) {
          // Load the data from A and B matrix to the fragments.
          wmma::load_matrix_sync(frag_a, A + (aRow * ldA) + aCol, ldA);
          wmma::load_matrix_sync(frag_b, B + (bRow * ldB) + bCol, ldB);

          // Perform the matrix multiplication using wmma instructions.
          wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        
    }

    unsigned int cRow = warpM * WMMA_M;
    unsigned int cCol = warpN * WMMA_N;

    if(cRow < M && cCol < N) {
      // Store the result in C matrix.
      wmma::store_matrix_sync(C + (cRow * ldC) + cCol, frag_c, ldC, wmma::mem_row_major);
    }
    
}


void gemm_wrapper(float const *host_A, float const *host_B, float *host_C, unsigned int M, unsigned int N, unsigned int K) {

    // Define the types for A, B and C matrices.
    float *A_d, * B_d, * C_d;
    __half *A_d_half, * B_d_half;
    
    // leading dimensions. Both A and B are K-major.
    unsigned int ldA = K;
    unsigned int ldB = K;
    unsigned int ldC = N;

    // Define Strides for A, B and C matrix.
    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaError_t error;
    error = cudaMalloc((void **) &A_d, sizeof(float) * M * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &A_d_half, sizeof(__half) * M * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &B_d, sizeof(float) * N * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &B_d_half, sizeof(__half) * N * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &C_d, sizeof(float) * M * N);
    cudaHandleSyncError(error);
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);

    // Copy the Input matrix to device memory.
    timer = initTimer(1);
    startTimer(&timer);

    error = cudaMemcpy( A_d, host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaHandleSyncError(error);
    error = cudaMemcpy( B_d, host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    cudaHandleSyncError(error);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU Global Memory: ", CYAN);

    // Convert from float to half precision.
    timer = initTimer(1);
    startTimer(&timer);
    fp32_to_fp16<<<(M*K + 1023) / 1024, 1024>>>(A_d_half, A_d, M*K);
    cudaHandleAsyncError();
    fp32_to_fp16<<< (N*K + 1023) / 1024, 1024>>>(B_d_half, B_d, N*K);
    cudaHandleAsyncError();
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to convert from float to half: ", CYAN);

    // Define the number of threads per block and number of blocks.    
    // 128 * 4 means 16 warps; they will compute the 64 x 64 tile of C.
    dim3 threadsPerBlock(128, 4);
    dim3 numBlocks( ( M + (WMMA_M * (threadsPerBlock.x / 32) - 1)) / (WMMA_M * (threadsPerBlock.x / 32)), 
                  (N + (WMMA_N * threadsPerBlock.y - 1)) / (WMMA_N * threadsPerBlock.y));
    std::cout << "\nGrim Dims: (X,Y)=(" << numBlocks.x << "," << numBlocks.y << ")\n";

    // launch the kernel.
    timer = initTimer(1);
    startTimer(&timer);

    gemm<<<numBlocks, threadsPerBlock>>>(A_d_half, B_d_half, C_d,
                                        ldA, ldB, ldC,
                                        M, N, K);

    
    cudaHandleAsyncError();
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // copy the output from device memory to host memory.
    timer = initTimer(1);
    startTimer(&timer);

    error = cudaMemcpy( host_C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaHandleSyncError(error);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Free the memory allocated on device.
    cudaFree(A_d);
    cudaFree(A_d_half);
    cudaFree(B_d);
    cudaFree(B_d_half);
    cudaFree(C_d);

    cudaDeviceSynchronize();
}

template<typename TA>
void print2DMatrix(const TA *A, unsigned int M, unsigned int N, char * annotation) {
    if(PRINT_MATRIX) {
      std::cout << "\n" << annotation << std::endl;
      for(unsigned int i = 0; i < M; i++) {
        for(unsigned int j = 0; j < N; j++) {
            std::cout << A[(i*N)+j] << "    ";
        }
        std::cout << "\n";
      }
    }
}

int main(int argc, char** argv) {

    // matrix dimensions.
    unsigned int M = 4096;
    unsigned int N = 2048;
    unsigned int K = 512;

    unsigned int maxValue = 5;

    // Hold the results from GPU implementation.
    float *host_C_GPU = (float *) malloc(sizeof(float) * M * N);

    // Initialize A and B matrix with the random values.
    // We define the A and B matrix as M,K and N,K.
    Matrix matA = random_clipped_matrix_2D(M, K, maxValue);
    Matrix matB = random_clipped_matrix_2D(N, K, maxValue);


    print2DMatrix(matA.buffer, M, K, "Matrix A:");

    print2DMatrix(matB.buffer, N, K, "Matrix B:");

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    std::cout << "\nComputing Matrix Multiplication on GPU: \n";
    // Matrix multiplication - Tiled implementation on GPU.
    gemm_wrapper(matA.buffer, matB.buffer, host_C_GPU, M, N, K);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU End to End Execution Time: ", GREEN);

    if(COMPARE_WITH_CPU)
    {
        // Hold results from CPU implementation.
        float *host_C_CPU = (float *) malloc(sizeof(float) * M * N);

        std::cout << "\nComputing Matrix Multiplication on CPU: \n";
        timer = initTimer(1);
        startTimer(&timer);

        // matrix multiplication on CPU.
        matmul(matA.buffer, matB.buffer, host_C_CPU, M, N, K);
        stopAndPrintElapsed(&timer, "CPU Execution Time: ", CYAN);

        std::cout << "\nComparing Results: \n";

        Matrix matCHost;
        matCHost.rows = M;
        matCHost.cols = N;
        matCHost.buffer = host_C_CPU;

        Matrix matCGPU;
        matCGPU.rows = M;
        matCGPU.cols = N;
        matCGPU.buffer = host_C_GPU;

        print2DMatrix(host_C_CPU, M, N, "Matrix C CPU:");

        print2DMatrix(host_C_GPU, M, N, "Matrix C GPU:");

        //float eps = 0.00001;
        bool areEqual = are_matrix_equal(&matCGPU, &matCHost);
        std::cout << "\nAre Matrix Equal? " << areEqual << std::endl;

        if(areEqual == 0) {
          for(unsigned int i=0; i<M; i++) {
            for(unsigned int j=0; j<N; j++) {
              if(host_C_CPU[(i*N)+j] != host_C_GPU[(i*N)+j]) {
                printf("\nFound Mismatch at (%d, %d). CPU: %4.2f, GPU: %4.2f\n", i, j, host_C_CPU[(i*N)+j], host_C_GPU[(i*N)+j]);
                break;
              }
            }
          }
        }

        free(host_C_CPU);
    }

    // release all the memory allocated.
    release_matrix(&matA);
    release_matrix(&matB);
    free(host_C_GPU);

    return 0;
}
