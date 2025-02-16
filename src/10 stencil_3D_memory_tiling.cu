#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

#define OUTPUT_TILE_SIZE 8
#define INPUT_TILE_SIZE (OUTPUT_TILE_SIZE + 2)
#define BLOCK_SIZE INPUT_TILE_SIZE

#define C0 1.0
#define C1 0.5

/**
 * Performs stencil convolution operation on input of N*N*N 3 dimensional data
 * and stores results in output. The implementation uses memory tiling where threads 
 * in the same block cooperate to load the (part of) input data into a shared memory before using
 * it.
 */
__global__ void stencil_kernel(float * input, float * output, unsigned int N) {

    int i = blockIdx.z * OUTPUT_TILE_SIZE + threadIdx.z - 1;
    int j = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - 1;
    int k = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - 1;
    
    __shared__ float input_tile[INPUT_TILE_SIZE][INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    // load input tile.
    if (i>=0 && i<(N) && j>=0 && j<(N) && k>=0 && k<(N) ) {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[(i*N+j)*N+k];
    }
    else {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    // compute stencil.
    if(i>=1 && i<(N-1) && j>=1 && j<(N-1) && k>=1 && k<(N-1) ) {
        if(threadIdx.x >= 1 && threadIdx.x < N-1 && threadIdx.y >= 1 && threadIdx.y < N-1 && threadIdx.z >= 1 && threadIdx.z < N-1) {
            output[(i*N+j)*N+k] = C0 * (input_tile[threadIdx.z][threadIdx.y][threadIdx.x]) +
                C1 * ((input_tile[threadIdx.z][threadIdx.y][threadIdx.x - 1]) +
                    (input_tile[threadIdx.z][threadIdx.y][threadIdx.x + 1]) +
                    (input_tile[threadIdx.z][threadIdx.y - 1][threadIdx.x]) +
                    (input_tile[threadIdx.z][threadIdx.y + 1][threadIdx.x - 1]) +
                    (input_tile[threadIdx.z - 1][threadIdx.y][threadIdx.x - 1]) +
                    (input_tile[threadIdx.z + 1][threadIdx.y][threadIdx.x - 1]));
        }
    }
    __syncthreads();
}

void stencil_wrapper(float * input, float * output, unsigned int N) {

    // Allocate memory on the GPU device.
    float *input_d, *output_d;
    size_t mat_size = sizeof(float) * N * N * N;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaMalloc((void **) &input_d, mat_size);
    cudaMalloc((void **) &output_d, mat_size);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);

    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(input_d, input, mat_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);

    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE ); 
    stencil_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, N);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(output, output_d, mat_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(int argc, char** argv) {
    unsigned int N = 1024;

    size_t mat_size = sizeof(float) * N * N * N;

    // Allocate memory on the host for output.
    float * output = (float*) malloc(mat_size);

    // Create a Matrix of width * height containing fkoat values.
    Matrix3D input = ones_matrix_3D(N, N, N);

    // Run the convolution.
    stencil_wrapper(input.buffer, output, N);
    cudaDeviceSynchronize();

    Matrix3D output_mat;
    output_mat.depth = N;
    output_mat.rows = N;
    output_mat.cols = N;
    output_mat.buffer = output;

    bool areEqual = 1;
    for(int i= 1; i<N-1; i++) {
        for(int j= 1; j<N-1; j++) {
            for(int k= 1; k<N-1; k++) {
                unsigned int index = (i*N+j)*N+k;
                if(output_mat.buffer[index]!=4.0)
                    areEqual = 0;
            }
        }   
    }
    printf("\nDo outputs of Input and Output match? %d\n", areEqual);
    

    //printf("Manually inspect first few elements:\n");
    //int limit = (N * N * N) > 1024? 1024: (N * N * N);
    //for(int i=10000; i<10000+limit; i++) {
    //    printf("%d: %18.6f       %18.6f\n", i, input.buffer[i], output[i]);
    //}

    // Free up the space allocated for holding the input matrix and the output.
    release_matrix_3D(&input);
    release_matrix_3D(&output_mat);
    //free(output);

    return 0;
}