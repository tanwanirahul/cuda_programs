# CUDA programs
A repo containing CUDA programs for anyone looking to get started with CUDA programming. 

The problems are adapted from professor Izzat El Hajj's course on [GPU computing](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) which draws upon the content from a classic reference book `Programming Massively Parallel Processors by David B. Kirk, Wen-mei W. Hwu', Izzat El Hajj.`

All the programs are tested in an environment having either of Nvidia L4/T4 GPU running with cuda version (12.1).

In addition to the problems covered in the course, this respository contains:

1. Appropriate boundary checks while accessing global memory.
2. Utility functions to create input data for your kernel code.
3. CUDA programs for the exercises mentioned in the lectures. 


## Profiling

The profiling tools mentioned in the lecture 7 of the course were nvprof and nvvp. Unfortunately, these profiling tools have been deperecated and aren'r available for latest version of the cuda environment. Nvidia's Nsight Systems and Nsight Compute are the recommended and supported tools for profiling and visualization purposes. 

You can learn about Nsight Compute and System tools and their usage with [this youtube video](https://www.youtube.com/watch?v=nhTjq0P9uc8&list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj&index=9).

## Debugging

As you build CUDA programs, you will eventually run into various kind of problems that might not be very apparent even to experienced developers. There are a few aspects that makes debugging CUDA programs challenging:
1. **Host and Device execution paths** - Some part of your program will run on HOST while the remaining (kernels)
will run on  the device. The error might result from any of these execution paths.
2. **Asynchronous execution at the host** - The execution on the host end is asynchronous. If there is an issue in performing a task on the device end, it would be known at the host only at later time in host thread execution. 
3. **Parallel execution on the device** - If the code is successfully launched from the host, the error might result from the parallel execution of the kernel code at the device by 1000's of threads. Issues like illegal access to a memory location due to lack of boundary checks, Race condition resulting from unsychronized access to a shared memory are typical examples of the problems that you will likely run into when building CUDA programs. 


There are best practices you can follow to detect and handle such errors. Furthermore, Nvidia has built various tools to help debug common issues mentioned above. The resource that helped me learn more about the debugging best practices and the tools available [is this video](https://www.youtube.com/watch?v=nAsMhH1tnYw).

## Extras
In addition to the problems covered in the PMPP course, the repo contains some problems that weren't part of the couese. All cuda program from #32 onwards are outside of the course material. 