#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

#define MASK_SIZE 5

__constant__ float MASK[MASK_SIZE][MASK_SIZE];

__global__ void convolution_kernel(float * input, float * output, unsigned int width, unsigned int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height and col < width) {
        float average = 0;
        for(int j = 0; j < MASK_SIZE; j++) {
            for(int k = 0; k < MASK_SIZE; k++) {
                int input_row = row - (MASK_SIZE/2) + j;
                int input_col = col - (MASK_SIZE/2) + k;
                if(input_row >= 0 && input_row < height && input_col >=0 && input_col < width) {
                    average+= MASK[j][k] * input[input_row * width + input_col];
                }
            }
        }
      output[row*width+col] = average; 
    }
}

void convolution_wrapper(float conv_mask[MASK_SIZE][MASK_SIZE], float * input, float * output, unsigned int width, unsigned int height) {

    // Allocate memory on the GPU device.
    float *input_d, *output_d;
    size_t mat_size = sizeof(float) * width * height;

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

     // Copy mask to GPU device's constant memory.
    cudaMemcpyToSymbol(MASK, conv_mask, MASK_SIZE*MASK_SIZE*sizeof(float));
    cudaDeviceSynchronize();

    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y ); 
    convolution_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, width, height);
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
    unsigned int width = 10240;
    unsigned int height = 10240;
    //unsigned int blur_size = 5;

    // Initialize the kernel to compute the identity value.
    float conv_mask[MASK_SIZE][MASK_SIZE];
    printf("Mask for convolution: \n");
    for(int i=0; i<MASK_SIZE; i++) {
        for(int j=0; j<MASK_SIZE; j++) {
            conv_mask[i][j]= (i==MASK_SIZE/2 && i==j)?1.0:0.0;
            printf("%2.1f ", conv_mask[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    size_t mat_size = sizeof(float) * width * height;

    // Allocate memory on the host for output.
    float * output = (float*) malloc(mat_size);

    // Create a Matrix of width * height containing fkoat values.
    Matrix input = random_matrix_2D(width, height);

    // Run the convolution.
    convolution_wrapper(conv_mask, input.buffer, output, width, height);
    cudaDeviceSynchronize();

    Matrix output_mat;
    output_mat.rows = width;
    output_mat.cols = height;
    output_mat.buffer = output;

    bool areEqual = are_matrix_equal(&input, &output_mat);
    printf("\nDo outputs of Input and Output match? %d\n", areEqual);

    //printf("Manually inspect first few elements:\n");
    //int limit = (height * width) > 1024? 1024: (height * width);
    //for(int i=0; i<limit; i++) {
    //    printf("%d: %18.6f       %18.6f\n", i, input.buffer[i], output[i]);
    //}

    // Free up the space allocated for holding the input matrix and the output.
    release_matrix(&input);
    release_matrix(&output_mat);
    //free(output);

    return 0;
}