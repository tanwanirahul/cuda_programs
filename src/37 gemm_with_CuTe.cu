/**
 * Genralized Matrix Multiplication (GEMM) using the CuTe Tensors
 and Layout abstractions.

 C = A * B

 A: M * K matrix.
 B: N * K matrix.
 C: M * N matrix.

 Note: A is m-major (column major) and B is n-major (column major).
 */

#include<cstdio>
#include<cstdlib>
#include<stdbool.h>

#include<cute/tensor.hpp>

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


template<typename MatShape, typename BLOCK_Tiler,
        typename TA, typename TB, typename TC,
        typename AStride, typename BStride, typename CStride,
        typename ASmemLayout, typename BSmemLayout,
        typename AThreadLayout, typename BThreadLayout, typename CThreadLayout>
__global__ void 
//__maxnreg__(64)
gemm(MatShape MNK_shape, BLOCK_Tiler block_tiler,
                    TA const *A, TB const *B, TC * C,
                    AStride dA, BStride dB, CStride dC,
                    ASmemLayout sA_layout, BSmemLayout sB_layout,
                    AThreadLayout tA, BThreadLayout tB, CThreadLayout tC) {


    using namespace cute;

    // Global memory tensors - represent the whole matrix.
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(MNK_shape), dA); //(M, K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(MNK_shape), dB); //(N, K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(MNK_shape), dC); //(M, N)

    // Block Coordinate - Select the specific tiled block, keeping all the elements on k dimension.
    auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);

    // Create tensors representing the block tile.
    Tensor blockTile_A = local_tile(mA, block_tiler, block_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, NUM_K_Tiles)
    Tensor blockTile_B = local_tile(mB, block_tiler, block_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, NUM_K_Tiles)
    Tensor blockTile_C = local_tile(mC, block_tiler, block_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

    // Create the shared memory buffer for the block.
    __shared__ TA smemA[cosize(sA_layout)]; // bM * bK
    __shared__ TB smemB[cosize(sB_layout)]; // bN * bK

    // Create Tensor pointing to SMemory created for A and B tiles.
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // Create a thread layout to assign each thread the subtile to copy
    // from global memory to shared memory. We need to define tensors for both
    // global memory and shared memory.
    Tensor tA_blockTile_A = local_partition(blockTile_A, tA, threadIdx.x); // (THREAD_M, THREAD_K, NUM_K_TILES)
    Tensor tA_sA = local_partition(sA, tA, threadIdx.x); // (THREAD_M, THREAD_K)

    Tensor tB_blockTile_B = local_partition(blockTile_B, tB, threadIdx.x); // (THREAD_N, THREAD_K, NUM_K_TILES)
    Tensor tB_sB = local_partition(sB, tB, threadIdx.x); // (THREAD_N, THREAD_K)

    // Thread partitioning layout for computing the accumulators.
    Tensor tC_sA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THREAD_M, BLK_K)
    Tensor tC_sB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THREAD_N, BLK_K)
    Tensor tC_blockTile_C = local_partition(blockTile_C, tC, threadIdx.x, Step<_1, _1>{}); // (THREAD_M, THREAD_N)

    // Create a space for local accumulator.
    Tensor tC_rC = make_tensor_like(tC_blockTile_C); // (THREAD_M, THREAD_N)

    clear(tC_rC);

    #if DEBUG_PRINT
    if(thread0()) {
        print("  numBlocks: ("); print(gridDim.x); print(",  "); print(gridDim.y); print(")\n");
        print("  block tiler : "); print(block_tiler); print("\n");
        print("\n\n");
        print("  mA : "); print(  mA); print("\n");
        print("  gA : "); print(  blockTile_A); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print(tA_blockTile_A); print("\n");
        print("tAsA : "); print(tA_sA); print("\n");
        print("\n\n");
        print("  mB : "); print(  mB); print("\n");
        print("  gB : "); print(  blockTile_B); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tB_blockTile_B); print("\n");
        print("tBsB : "); print(tB_sB); print("\n");
        print("\n\n");
        print("  mC : "); print(  mC); print("\n");
        print("  gC : "); print(  blockTile_C); print("\n");
        print("tCsA : "); print(tC_sA); print("\n");
        print("tCsB : "); print(tC_sB); print("\n");
        print("tCgC : "); print(tC_blockTile_C); print("\n");
        print("tCrC : "); print(tC_rC); print("\n");
    }
    #endif

    // Each thread will move the NUM_K_TILES sequentially into shared memory from global memory
    // compute the appropriate output matrix elements.
    auto num_K_tiles = size<2>(tA_blockTile_A);

    for(unsigned int K_tile=0; K_tile < num_K_tiles; K_tile++) {
        // load the tile of data from global memory to shared memory.
        copy(tA_blockTile_A(_,_,K_tile), tA_sA);
        copy(tB_blockTile_B(_,_,K_tile), tB_sB);

        // Sync before using the data from shared memory.
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tC_sA, tC_sB, tC_rC);
        __syncthreads();
    }

    copy(tC_rC, tC_blockTile_C);
}


template<typename TA, typename TB, typename TC>
void gemm_wrapper(TA *host_A, TB *host_B, TC *host_C, unsigned int M, unsigned int N, unsigned int K) {

    using namespace cute;

    // Define the MNK_shape
    auto MNK_shape = make_shape(M, N, K);

    // leading dimensions. Both A and B are K-major.
    unsigned int ldA = K;
    unsigned int ldB = K;
    unsigned int ldC = N;

    // Define Strides for A, B and C matrix.
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});

    // Define the block tiler.
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<4>{};

    auto block_tiler = make_shape(bM, bN, bK);

    // Define layout for shared memory.
    auto sA_layout = make_layout(make_shape(bM, bK), LayoutRight{});
    auto sB_layout = make_layout(make_shape(bN, bK), LayoutRight{});

    // Define the thread partitioning for sA, sB and C matrix.
    auto tA = make_layout(make_shape(Int<64>{}, Int<4>{}), LayoutRight{});
    auto tB = make_layout(make_shape(Int<64>{}, Int<4>{}), LayoutRight{});
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    // Allocate memory on the device for A, B and C matrix.
    TA * A_d;
    TB * B_d;
    TC * C_d;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaError_t error;
    error = cudaMalloc((void **) &A_d, sizeof(TA) * M * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &B_d, sizeof(TB) * N * K);
    cudaHandleSyncError(error);
    error = cudaMalloc((void **) &C_d, sizeof(TC) * M * N);
    cudaHandleSyncError(error);
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);

    // Copy the Input matrix to device memory.
    timer = initTimer(1);
    startTimer(&timer);

    error = cudaMemcpy( A_d, host_A, sizeof(TA) * M * K, cudaMemcpyHostToDevice);
    cudaHandleSyncError(error);
    error = cudaMemcpy( B_d, host_B, sizeof(TB) * N * K, cudaMemcpyHostToDevice);
    cudaHandleSyncError(error);

    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU Global Memory: ", CYAN);

    // launch the kernel.
    timer = initTimer(1);
    startTimer(&timer);

    // Set the cache configuration for the kernel.
    // TODO: The behaviour of the kernel is such that it will benefit from having a larger L1 cache to hold 
    // register spills. However, setting the cache configuration to prefer L1 cache on T4 is resulting in 
    // a kernel launch failure. Need to investigate this further.
    //error = cudaFuncSetCacheConfig((const char *) "gemm", cudaFuncCachePreferEqual);
    //cudaHandleSyncError(error);

    dim3 threadsPerBlock(size(tC));
    dim3 numBlocks(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm<<<numBlocks, threadsPerBlock>>>(MNK_shape, block_tiler,
                                        A_d, B_d, C_d,
                                        dA, dB, dC,
                                        sA_layout, sB_layout,
                                        tA, tB, tC);

    cudaHandleAsyncError();
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // copy the output from device memory to host memory.
    timer = initTimer(1);
    startTimer(&timer);

    error = cudaMemcpy( host_C, C_d, sizeof(TC) * M * N, cudaMemcpyDeviceToHost);
    cudaHandleSyncError(error);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Free the memory allocated on device.
    cudaFree(A_d);
    cudaFree(B_d);
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

    // define the types for A, B and C matrices.
    using TA = float;
    using TB = float;
    using TC = float;

    // matrix dimensions.
    unsigned int M = 4096;
    unsigned int N = 2048;
    unsigned int K = 512;

    unsigned int maxValue = 5;

    // Hold the results from GPU implementation.
    TC *host_C_GPU = (TC *) malloc(sizeof(TC) * M * N);

    // Initialize A and B matrix with the random values.
    // We define the A and B matrix as M,K and N,K.
    Matrix matA = random_clipped_matrix_2D(M, K, maxValue);
    Matrix matB = random_clipped_matrix_2D(N, K, maxValue);
    //Matrix matB = identity_matrix_2D(N);

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
        TC *host_C_CPU = (TC *) malloc(sizeof(TC) * M * N);

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