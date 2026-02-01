#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
    else return;
}

void saxpyCpu(int N, float alpha, float* x, float* y, float* result) {
    for (int i = 0; i < N; i++) {
        result[i] = alpha * x[i] + y[i];
    }
}


void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray, float* result_cpu) {

    double start = CycleTimer::currentSeconds();
    saxpyCpu(N, alpha, xarray, yarray, result_cpu);
    double end = CycleTimer::currentSeconds();

    double cpuTime = end - start;
    printf("CPU time: %.3f ms\t\t[%.3f GB/s]\n",
        1000.f * cpuTime, toBW(sizeof(float)*3*N, cpuTime));

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 1024; //Default 512
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;
    int memSize = sizeof(float)*N;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    cudaMalloc(&device_x, sizeof(float)*N);
    cudaMalloc(&device_y, sizeof(float)*N); 
    cudaMalloc(&device_result, sizeof(float)*N);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_x, xarray, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, memSize, cudaMemcpyHostToDevice);


    //
    // TODO: insert time here to begin timing only the kernel
    //
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart);

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaThreadSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    float kernelMs = 0.f;
    cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop);

    printf("GPU kernel time: %.3f ms\n", kernelMs);

    // cleanup
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);


    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(resultarray, device_result, memSize, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    //cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

