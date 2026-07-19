%%cuda
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vector_add(const float* A, const float* B, float* C, int N) { //kernel function
  int i= blockIdx.x * blockDim.x + threadIdx.x; //data index
  if (i<N) { //avoid out of bound
      C[i]=A[i]+B[i];
  }
}

int main() {
    const int N=10;
    float A[N],B[N],C[N];

    for(int i=0;i<N;i++) {
        A[i]=i;
        B[i]=i*2;
    }
    
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,N*sizeof(float)); //allocate memory on device
    cudaMalloc(&d_b,N*sizeof(float));
    cudaMalloc(&d_c,N*sizeof(float));
    cudaMemcpy(d_a,A,N*sizeof(float),cudaMemcpyHostToDevice); //copy data from host to device
    cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice);




    int block_size=256; //number of thread in a block
    int grid_size = (N + block_size - 1) / block_size; //number of blocks

    vector_add<<<grid_size,block_size>>>(d_a,d_b,d_c,N); //launch kernel
    cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost); //copy data from device to host

    for(int i=0;i<N;i++) {
      std::cout<<C[i]<<" "; //0 3 6 9 12 15 18 21 24 27
    }
    cout<<std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
