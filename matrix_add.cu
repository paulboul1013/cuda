%%cuda
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void Matrix_add_A(const float*A,const float *B,float *C,int N) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    int j= blockIdx.y * blockDim.y + threadIdx.y;

    if ((i>=N) && (j>=N)){
        return;
    }

    C[i*N+j]=A[i*N+j]+B[i*N+j];
}

__global__ void Matrix_add_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<N){
        for(int j=0;j<N;j++){
            C[i*N+j]=A[i*N+j]+B[i*N+j];
        }
    }
    return;
}

__global__ void Matrix_add_C(const float* A, const float* B, float* C, int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j<N){
        for(int i=0;i<N;i++) {
            C[i*N+j]=A[i*N+j]+B[i*N+j];
        }
    }

    return;
}

int main() {

  const int N=10;
  float *A,*B,*C;

  A=(float*)malloc(N*N*sizeof(float));
  B=(float*)malloc(N*N*sizeof(float));
  C=(float*)malloc(N*N*sizeof(float));
     
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++) {
        A[i*N+j] = 1.0f;
        B[i*N+j] = 2.0f;
        C[i*N+j] = 0.0f;
    }
  }

  float *d_a,*d_b,*d_c;
  cudaMalloc((void**)&d_a,N*N*sizeof(float));
  cudaMalloc((void**)&d_b,N*N*sizeof(float));
  cudaMalloc((void**)&d_c,N*N*sizeof(float));

  cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,B,N*N*sizeof(float),cudaMemcpyHostToDevice);

  dim3 dim_block(32,16);
  dim3 dim_grid(ceil(N/32.0f),ceil(N/16.0f));
  //Matrix_add_A<<<dim_grid,dim_block>>>(d_a,d_b,d_c,N);
  //Matrix_add_B<<<dim_grid,dim_block>>>(d_a,d_b,d_c,N);
  Matrix_add_C<<<dim_grid,dim_block>>>(d_a,d_b,d_c,N);

  cudaDeviceSynchronize();

  cudaMemcpy(C,d_c,N*N*sizeof(float),cudaMemcpyDeviceToHost);
  printf("C:\n");
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++) {
        printf("%f ",C[i*N+j]);
    }
    printf("\n");
  }

  printf("A:\n");
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++) {
        printf("%f ",A[i*N+j]);
    }
    printf("\n");
  }

  printf("B:\n");
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++) {
        printf("%f ",B[i*N+j]);
    }
    printf("\n");
  }
  
  
  return 0;
}