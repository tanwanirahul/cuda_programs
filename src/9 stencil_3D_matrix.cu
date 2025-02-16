#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

#define BLOCK_SIZE 8
#define C0 1.0
#define C1 0.5

/**
 * Performs stencil convolution operation on input of N*N*N 3 dimensional data
 * and stores results in output.
 */
__global__ void stencil_kernel(float * input, float * output, unsigned int N) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i>=1 && i<(N-1) && j>=1 && j<(N-1) && k>=1 && k<(N-1) ) {
        float result = 0;
        result = C0 * (input[(i*N*N) + (j*N) + k]) +
                C1 * ((input[(i*N*N) + (j*N) + k-1]) +
                    (input[(i*N*N) + (j*N) + k+1]) +
                    (input[(i*N*N) + ((j-1)*N) + k]) +
                    (input[(i*N*N) + ((j+1)*N) + k]) +
                    (input[((i-1)*N*N) + (j*N) + k]) +
                    (input[((i+1)*N*N) + (j*N) + k]));

      output[(i*N+j)*N+k] = result;
    }
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
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, (N + threadsPerBlock.z - 1) / threadsPerBlock.z ); 
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