#include "stdlib.h"
#include "matrix_utils.h"
#include "timer.h"

#define MASK_SIZE 5
#define MASK_RADIUS 2
#define OUTPUT_TILE_SIZE 16
#define INPUT_TILE_SIZE  (OUTPUT_TILE_SIZE + (2*MASK_RADIUS))

__constant__ float MASK[MASK_SIZE][MASK_SIZE];

__global__ void convolution_kernel(float * input, float * output, unsigned int width, unsigned int height) {

    // We only  need to compute the output for the output tile elements. Block
    // is launched with (OUTPUT_TILE_SIZE + 2 * MASK_RADIUS).
    int out_tile_threadIdx_y = threadIdx.y - MASK_RADIUS;
    int out_tile_threadIdx_x = threadIdx.x - MASK_RADIUS;
    int out_tile_block_start_y = blockIdx.y * OUTPUT_TILE_SIZE; 
    int out_tile_block_start_x = blockIdx.x * OUTPUT_TILE_SIZE; 

    // In every block we only compute the values for OUTPUT_TILE_SIZE dimensions. So 
    // we need to align based on that instead of blockDims.
    int in_tile_row = out_tile_block_start_y + out_tile_threadIdx_y;
    int in_tile_col = out_tile_block_start_x + out_tile_threadIdx_x;
    

    // Create an input tile for the block in shared memory.
    __shared__ float input_tile[INPUT_TILE_SIZE][INPUT_TILE_SIZE];
 
    // load the input tile.
    if(in_tile_row >=0 && in_tile_row < height && in_tile_col >=0 && in_tile_col < width){
        input_tile[threadIdx.y][threadIdx.x] = input[in_tile_row * width + in_tile_col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    

    // Apply the mask to the loaded input tile.
    if(out_tile_threadIdx_y >= 0 && out_tile_threadIdx_y < OUTPUT_TILE_SIZE && out_tile_threadIdx_x >=0 && out_tile_threadIdx_x < OUTPUT_TILE_SIZE) {
        float average = 0;
        for(int j = 0; j < MASK_SIZE; j++) {
            for(int k = 0; k < MASK_SIZE; k++) {
                int input_row = threadIdx.y - MASK_RADIUS + j;
                int input_col = threadIdx.x - MASK_RADIUS + k;
                if(input_row >= 0 && input_row < INPUT_TILE_SIZE && input_col >=0 && input_col < INPUT_TILE_SIZE) {
                    average+= MASK[j][k] * input_tile[input_row][input_col];
                }
                else {
                    printf("Shouldn't reach here. Something is wrong. in_tile_row: %d, in_tile_col: %d, out_tile_row: %d, out_tile_col: %d", in_tile_row, in_tile_col, out_tile_threadIdx_y, out_tile_threadIdx_x);
                }
            }
        }
        __syncthreads();

        // Update the output tile with the result of applying the convolutio mask.
        output[in_tile_row * width + in_tile_col] = average; 

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
    // dim3 threadsPerBlock(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
    // dim3 numBlocks((width + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (height + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE); 
    dim3 threadsPerBlock(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
    dim3 numBlocks((width + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (height + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE ); 
    printf("Threads per block - x: %d, y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Num Blocks - x: %d, y: %d\n", numBlocks.x, numBlocks.y);
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
    //   printf("%d: %18.6f       %18.6f\n", i, input.buffer[i], output[i]);
    //}

    // Free up the space allocated for holding the input matrix and the output.
    release_matrix(&input);
    release_matrix(&output_mat);
    //free(output);

    return 0;
}