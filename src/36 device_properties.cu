#include <stdio.h>

#define GB_BYTES (1024*1024*1024)

int main(int argc, char** argv) {
    
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    printf("\nNumber of GPU devices found: %d\n", numDevices);
    for(unsigned int i=0; i<numDevices; i++) {
        printf("\n===========================================\n");
        printf("Properties for Device: %d\n", i);
        printf("===========================================\n");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("    Device Name: %s\n", prop.name);
        printf("    Total Global Memory (GiB): %20.18f\n", prop.totalGlobalMem / (GB_BYTES));
        printf("    Peak Memory Bandwidth (GiB/s): %20.18f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8)/(1024*1024));
        printf("    Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("    Memory Bus Width (Gib): %20.18f\n", prop.memoryBusWidth / (GB_BYTES));
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Number of SMs: %d\n", prop.multiProcessorCount);
        printf("    Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("    Max Threads Dimensions: X:%d, Y:%d, Z:%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("    Max Grid Dimensions: X:%d, Y:%d, Z:%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("    Global L1 Cache (KiB): %20.18f\n", prop.globalL1CacheSupported / 1024);
        printf("    Local L1 Cache (KiB): %20.18f\n", prop.localL1CacheSupported / 1024);
        printf("    Shared Memory per Block (KiB): %20.18f\n", prop.sharedMemPerBlock / 1024);
        printf("    Shared Memory per SM (KiB): %20.18f\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("    Num of Registers Per SM: %d\n", prop.regsPerMultiprocessor);
        printf("    Max Blocks Per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("    Max Threads Per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    }

    return 0;
}