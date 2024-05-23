#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void histogramKernel(unsigned char* input, int* histogram, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelValue = input[y * width + x];
        atomicAdd(&histogram[pixelValue], 1);
        /**(histogram + pixelValue) += 1;
        __syncthreads();*/
    }
}

void computeHistogram(const cv::Mat& inputImage, int* histogram) {
    int imageSize = inputImage.rows * inputImage.cols;
    unsigned char* d_input;
    int* d_histogram;

    cudaMalloc(&d_input, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemcpy(d_input, inputImage.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((inputImage.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputImage.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    histogramKernel <<<blocksPerGrid, threadsPerBlock >>> (d_input, d_histogram, inputImage.cols, inputImage.rows);

    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_histogram);
}

int main() {
    cv::Mat inputImage = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // Save the grayscale image to verify
    cv::imwrite("grayscale.jpg", inputImage);

    int histogram[256] = { 0 };

    computeHistogram(inputImage, histogram);

    // Save histogram to a file
    FILE* f = fopen("histogram.txt", "w");
    for (int i = 0; i < 256; i++) {
        fprintf(f, "%d\n", histogram[i]);
    }
    fclose(f);

    return 0;
}
