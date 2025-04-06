//%%writefile GEMM_using_mma_with_CuTe.cu

/**
 * Genralized Matrix Multiplication (GEMM) using the CuTe Tensors
 and Layout abstractions.

 C = A * B

 A: M * K matrix.
 B: N * K matrix.
 C: M * N matrix.

 
 The implementation leverages mma instruction executed by Tensor cores 
 for matrix multiplication that is available from Volta architecture and later. 
 The code is using SM75 wmma instructions and is expected to be compiled on
 Turing architecture such as T4.

 Cutlass's CuTe exposes the mma instructions through an abstraction called Atom. We will
 use the Atom that exposes one of mma instructions supported by sm75 device.
  
  - mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32


  IMP: To pipeline global memory read operations and compute operations, we prefetch a Tile
  into registers. At each iteration, we copy the the prev tile from registers to shared memory,
  fetch the next tile from global memory to registers, and compute the tile already moved to shared
  memory using the mma instruction. This way, the mma computation can execute while the data is being
  read from global memory for the next tile.
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
        typename ASmemLayout, typename BSmemLayout,
        typename CopyTilerA, typename CopyTilerB, typename MMATilerC>
__global__ void 
//__maxnreg__(64)
gemm(MatShape MNK_shape, BLOCK_Tiler block_tiler,
                    TA const *A, TB const *B, TC * C,
                    AStride dA, BStride dB, CStride dC,
                    ASmemLayout sA_layout, BSmemLayout sB_layout,
                    CopyTilerA copyA, CopyTilerB copyB, MMATilerC mma) {


    using namespace cute;

    // Global memory tensors - represent the whole matrix.
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(MNK_shape), dA); //(M, K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(MNK_shape), dB); //(N, K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(MNK_shape), dC); //(M, N)

    // Block Coordinate - Select the specific tiled block, keeping all the elements on k dimension.
    auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);

    // Create tensors representing the block tile.
    Tensor gA = local_tile(mA, block_tiler, block_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, NUM_K_Tiles)
    Tensor gB = local_tile(mB, block_tiler, block_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, NUM_K_Tiles)
    Tensor gC = local_tile(mC, block_tiler, block_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

    // Create the shared memory buffer for the block.
    __shared__ TA smemA[cosize(sA_layout)];
    __shared__ TB smemB[cosize(sB_layout)];

    // Create Tensor pointing to SMemory created for A and B tiles.
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // Partition the A and B matrices based on the CopyTiler.
    ThrCopy thr_copyA = copyA.get_slice(threadIdx.x);
    Tensor tAgA = thr_copyA.partition_S(gA); // (copy_inst, copy_inst_M, copy_inst_K, num_K_tiles)
    Tensor tAsA = thr_copyA.partition_D(sA); // (copy_inst, copy_inst_M, copy_inst_K)
    Tensor tArA = make_fragment_like(tAsA); // (copy_inst, copy_inst_M, copy_inst_K)

    ThrCopy thr_copyB = copyB.get_slice(threadIdx.x);
    Tensor tBgB = thr_copyB.partition_S(gB); // (copy_inst, copy_inst_N, copy_inst_K, num_K_tiles)
    Tensor tBsB = thr_copyB.partition_D(sB); // (copy_inst, copy_inst_N, copy_inst_K)
    Tensor tBrB = make_fragment_like(tBsB); // (copy_inst, copy_inst_N, copy_inst_K)

    // Copy the first tile from global memory to registers.
    copy(copyA, tAgA(_,_,_,0), tArA);
    copy(copyB, tBgB(_,_,_,0), tBrB);

    // Partition the C matrix based on the MMATiler.
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (mma_inst, mma_inst_M, mma_inst_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (mma_inst, mma_inst_N, mma_inst_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (mma_inst, mma_inst_M, mma_inst_N)
    // Create a space for local accumulator.
    Tensor tCrC = make_fragment_like(tCgC); // (mma_inst, mma_inst_M, mma_inst_N)

    clear(tCrC);

    #if DEBUG_PRINT
    if(thread0()) {
        print("  numBlocks: ("); print(gridDim.x); print(",  "); print(gridDim.y); print(")\n");
        print("  block tiler : "); print(block_tiler); print("\n");
        print("\n\n");
        print("  mA : "); print(  mA); print("\n");
        print("  gA : "); print(  gA); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print( tAgA); print("\n");
        print("tAsA : "); print( tAsA); print("\n");
        print("tArA : "); print( tArA); print("\n");

        print("\n\n");
        print("  mB : "); print(  mB); print("\n");
        print("  gB : "); print(  gB); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        print("tBrB : "); print(tBrB); print("\n");

        print("\n\n");
        print("  mC : "); print(  mC); print("\n");
        print("  gC : "); print(  gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrC : "); print(tCrC); print("\n");
    }
    #endif

    // Loop over the K dimension to load the block tile one at a time.
    auto num_K_tiles = size<3>(tAgA);

    for(unsigned int K_tile=0; K_tile < num_K_tiles; K_tile++) {

        // load the prev loaded tile to shared memory.
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        // load the next tile from global memory to registers.
        unsigned int next_tile = (K_tile + 1) < num_K_tiles? (K_tile + 1): K_tile;
        copy(copyA, tAgA(_,_,_,next_tile), tArA);
        copy(copyB, tBgB(_,_,_,next_tile), tBrB);

        // perform gemm using the mma instruction.
        gemm(mma, tCsA, tCsB, tCrC);
    }
    __syncthreads();
    // copy the result from registers to global memory.
    copy(tCrC, tCgC);
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

    // We are using the mma instruction with the shape 16.8.8. 

    // Define the block tiler.
    auto bM = Int<128>{};
    auto bN = Int<64>{};
    auto bK = Int<8>{};

    auto block_tiler = make_shape(bM, bN, bK);

    // Define layout for shared memory. 
    auto sA_layout = make_layout(make_shape(bM, bK), 
                                 make_stride(Int<1>{}, bM+Int<1>{})); // Padded M-Major.

    auto sB_layout = make_layout(make_shape(bN, bK),
                                 make_stride(Int<1>{}, bN+Int<1>{})); // Padded N-Major.


    // Define the Copy Atoms to copy from global memory.
    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                      Layout<Shape<_4,_8>, Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                      Layout<Shape<_4,_8>, Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    // Define the MMA Atom to perform the matrix multiplication.
    TiledMMA mma = make_tiled_mma(MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>{},
                                  Layout<Shape<_1, _1>>{}, 
                                  Tile<_128,_64,_8>{});

    
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

    dim3 threadsPerBlock(size(mma));
    dim3 numBlocks(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm<<<numBlocks, threadsPerBlock>>>(MNK_shape, block_tiler,
                                        A_d, B_d, C_d,
                                        dA, dB, dC,
                                        sA_layout, sB_layout,
                                        copyA, copyB, mma);

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