//%%writefile gemm_with_CuTe_limited_registers.cu

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
#define COMPARE_WITH_CPU 0

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
        typename ASmemLayout, typename BSmemLayout, typename CSmemLayout,
        typename AThreadLayout, typename BThreadLayout, typename CThreadLayout, typename RegisterTiler>
__global__ void
__maxnreg__(64)
gemm(MatShape MNK_shape, BLOCK_Tiler block_tiler,
                    TA const *A, TB const *B, TC * C,
                    AStride dA, BStride dB, CStride dC,
                    ASmemLayout sA_layout, BSmemLayout sB_layout, CSmemLayout sC_layout,
                    AThreadLayout tA, BThreadLayout tB, CThreadLayout tC, RegisterTiler registerTiler) {


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
    __shared__ TC smemC[cosize(sC_layout)]; // bM * bN

    // Create Tensor pointing to SMemory created for A, B and C tiles.
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);

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
    Tensor tC_sC = local_partition(sC, tC, threadIdx.x, Step<_1, _1>{}); // (THREAD_M, THREAD_N)
    Tensor tC_blockTile_C = local_partition(blockTile_C, tC, threadIdx.x, Step<_1, _1>{}); // (THREAD_M, THREAD_N)


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
        print("tCsC : "); print(tC_sC); print("\n");
        //print("tCrC : "); print(tC_rC); print("\n");
    }
    #endif

    // Each thread will move the NUM_K_TILES sequentially into shared memory from global memory
    // compute the appropriate output matrix elements.
    auto num_K_tiles = size<2>(tA_blockTile_A);
    clear(tC_sC);

    for(unsigned int K_tile=0; K_tile < num_K_tiles; K_tile++) {
        // load the tile of data from global memory to shared memory.
        copy(tA_blockTile_A(_,_,K_tile), tA_sA);
        copy(tB_blockTile_B(_,_,K_tile), tB_sB);

        // Sync before using the data from shared memory.
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();


        auto tC_sC_registerTile = local_tile(tC_sC, registerTiler, _); // (RegTile_M, RegTile_N, (Num_M_Tiles, Num_N_Tiles)
        unsigned int num_M_TILES = size<2,0>(tC_sC_registerTile);
        unsigned int num_N_TILES = size<2,1>(tC_sC_registerTile);

        // Create a space for local accumulator.
        Tensor tC_rC = make_tensor_like(tC_sC_registerTile(_,_,0)); // (RegTile_M, RegTile_N)

        for(unsigned int M_Tile=0; M_Tile<num_M_TILES; M_Tile++) {
        for(unsigned int N_Tile=0; N_Tile<num_N_TILES; N_Tile++) {

            // Get the appropriate tile for A and B.
            auto tC_sA_registerTile = local_tile(tC_sA, make_shape(select<0>(registerTiler), shape<1>(tC_sA)), M_Tile);
            auto tC_sB_registerTile = local_tile(tC_sB, make_shape(select<0>(registerTiler), shape<1>(tC_sB)), N_Tile);

            #if DEBUG_PRINT
            if(thread0() && M_Tile==0 && N_Tile==0 && K_tile==0) {
                print("\n\n");
                print("  threadIdx.x"); print(threadIdx.x); print("\n");
                print("  blockIdx.x"); print(blockIdx.x); print("\n");
                print("  blockIdx.y"); print(blockIdx.y); print("\n");
                print("  M_Tile : "); print(  M_Tile); print("\n");
                print("  N_Tile : "); print(  N_Tile); print("\n");
                print("  tC_sA_rt : "); print(  tC_sA_registerTile); print("\n");
                print("  tC_sB_rt : "); print(  tC_sB_registerTile); print("\n");
                print("  tC_sC_rt : "); print(  tC_sC_registerTile); print("\n");
                print("  tC_rC : "); print(  tC_rC); print("\n");
            }
            #endif

            // Copy partial sums from C shared memory to register memory.
            copy(tC_sC_registerTile(_,_,make_coord(M_Tile, N_Tile)), tC_rC);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            #if DEBUG_PRINT
            if(thread0() && M_Tile==0 && N_Tile==0) {
                print("\n M_Tile: "); print(M_Tile); print(" N_Tile: "); print(N_Tile); print(" K_Tile: "); print(K_tile); print("\n");

                print("\nAccumulator elements:\n");
                for(unsigned int i=0; i<size<0>(tC_rC); i++) {
                    for(unsigned int j=0; j<size<1>(tC_rC); j++) {
                        print(tC_rC(i, j)); print(" ");
                    }
                    print("\n");
                }
            }
            #endif

            // Run the GEMM and accumulate the results in register memory tC_rC.
            gemm(tC_sA_registerTile, tC_sB_registerTile, tC_rC);

            #if DEBUG_PRINT
            if(thread0() && M_Tile==0 && N_Tile==0) {
                print("\n M_Tile: "); print(M_Tile); print(" N_Tile: "); print(N_Tile); print(" K_Tile: "); print(K_tile); print("\n");

                print("\nA elements:\n");
                for(unsigned int i=0; i<size<0>(tC_sA_registerTile); i++) {
                    for(unsigned int j=0; j<size<1>(tC_sA_registerTile); j++) {
                        print(tC_sA_registerTile(i, j)); print(" ");
                    }
                    print("\n");
                }

                print("\nB elements:\n");
                for(unsigned int i=0; i<size<0>(tC_sB_registerTile); i++) {
                    for(unsigned int j=0; j<size<1>(tC_sB_registerTile); j++) {
                        print(tC_sB_registerTile(i, j)); print(" ");
                    }
                    print("\n");
                }

                print("\nResulting elements:\n");
                for(unsigned int i=0; i<size<0>(tC_rC); i++) {
                    for(unsigned int j=0; j<size<1>(tC_rC); j++) {
                        print(tC_rC(i, j)); print(" ");
                    }
                    print("\n");
                }

            }
            #endif

            // Copy the update accumulated results back to C's shared memory.
            copy(tC_rC, tC_sC_registerTile(_,_,make_coord(M_Tile, N_Tile)));
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            #if DEBUG_PRINT
            if(thread0() && M_Tile==0 && N_Tile==0) {
                print("\n M_Tile: "); print(M_Tile); print(" N_Tile: "); print(N_Tile); print(" K_Tile: "); print(K_tile); print("\n");

                print("\nC shared memory elements:\n");
                for(unsigned int i=0; i<size<0>(tC_sC); i++) {
                    for(unsigned int j=0; j<size<1>(tC_sC); j++) {
                        print(tC_sC(i, j)); print(" ");
                    }
                    print("\n");
                }
            }
            #endif

        }
      }
        __syncthreads();
    }
    copy(tC_sC, tC_blockTile_C);
}


template<typename TA, typename TB, typename TC>
void gemm_wrapper(TA *host_A, TB *host_B, TC *host_C, unsigned int M, unsigned int N, unsigned int K) {

    using namespace cute;

    // Define the MNK_shape
    auto MNK_shape = make_shape(M, N, K);

    // leading dimensions. Both A and B are K-major (Row Major).
    unsigned int ldA = K;
    unsigned int ldB = K;
    unsigned int ldC = N;

    // Define Strides for A, B and C matrix.
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});

    // Define the block tiler.
    auto bM = Int<64>{};
    auto bN = Int<64>{};
    auto bK = Int<32>{};

    auto block_tiler = make_shape(bM, bN, bK);

    // Define layout for shared memory.
    auto sA_layout = make_layout(make_shape(bM, bK), LayoutRight{});
    auto sB_layout = make_layout(make_shape(bN, bK), LayoutRight{});
    auto sC_layout = make_layout(make_shape(bM, bN), LayoutRight{});

    // Define the thread partitioning for sA, sB and C matrix.
    auto tA = make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
    auto tB = make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));


    auto registerTiler = make_shape(Int<4>{}, Int<4>{});

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

    dim3 threadsPerBlock(size(tC));
    dim3 numBlocks(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm<<<numBlocks, threadsPerBlock>>>(MNK_shape, block_tiler,
                                        A_d, B_d, C_d,
                                        dA, dB, dC,
                                        sA_layout, sB_layout, sC_layout,
                                        tA, tB, tC, registerTiler);

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
