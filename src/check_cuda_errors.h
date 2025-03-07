// Error checking macro
#define cudaHandleSyncError(error) { handleSyncError((error), __FILE__, __LINE__); }
inline void handleSyncError(cudaError_t err, const char *file, int line, bool abort=true) {
    if (err != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define cudaHandleAsyncError() { handleAsyncError(__FILE__, __LINE__); }
inline void handleAsyncError(const char *file, int line, bool abort=true) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Async Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}