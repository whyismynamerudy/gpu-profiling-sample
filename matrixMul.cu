#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

__global__ void matrixMulKernel(float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

void matrixMul(float *a, float *b, float *c, int N) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim: " << deviceProp.maxThreadsDim[0] << " x " 
                 << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid size: " << deviceProp.maxGridSize[0] << " x " 
                 << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s): " << 
            2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << std::endl;
    }
}

int main(int argc, char **argv) {
    // Print CUDA device properties
    printDeviceProperties();
    
    // Set matrix size (default 1024 if not specified)
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    std::cout << "\nMatrix size: " << N << " x " << N << std::endl;

    // Allocate host memory
    size_t size = N * N * sizeof(float);
    float *a = new float[N * N];
    float *b = new float[N * N];
    float *c = new float[N * N];

    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    for(int i = 0; i < N * N; i++) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // Record compile time and initialization
    auto compile_start = std::chrono::high_resolution_clock::now();
    
    // Warmup run
    std::cout << "\nPerforming warmup run..." << std::endl;
    matrixMul(a, b, c, N);
    cudaDeviceSynchronize();
    
    auto compile_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compile_time = compile_end - compile_start;
    
    // Performance measurement runs
    const int num_runs = 5;
    double total_time = 0.0;
    
    std::cout << "\nPerforming " << num_runs << " timed runs..." << std::endl;
    
    for(int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        matrixMul(a, b, c, N);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        total_time += diff.count();
        std::cout << "Run " << run + 1 << ": " << diff.count() << " seconds" << std::endl;
    }

    // Calculate and print statistics
    double avg_time = total_time / num_runs;
    double gflops = (2.0 * N * N * N) / (avg_time * 1e9);  // 2N³ operations for matrix multiplication
    
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "Initialization + Compile time: " << compile_time.count() << " seconds" << std::endl;
    std::cout << "Average runtime: " << avg_time << " seconds" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Calculate and print checksum for verification
    float checksum = 0.0f;
    for(int i = 0; i < N * N; i++) {
        checksum += c[i];
    }
    std::cout << "Result checksum: " << checksum << std::endl;

    // Cleanup
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}