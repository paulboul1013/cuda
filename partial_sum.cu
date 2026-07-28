%%cuda
#include <iostream>
#include <cuda_runtime.h>

__global__ void partial_sum_kernel(int *input,int *output,int n) {
    //shared memory
    extern __shared__ int shared_memory[];

    int tid=threadIdx.x;
    int index=blockIdx.x*blockDim.x*2+tid;

    if (index < n){
        //load input into shared memory and optimize loading to do coalescing
        shared_memory[tid]=input[index]+input[index+blockDim.x];
        __syncthreads();

        //inclusive scan in shard memory
        for(int stride=1;stride < blockDim.x;stride*=2) {
            int temp=0;
            if (tid>=stride) {
                temp=shared_memory[tid-stride];
            }
            __syncthreads();
            shared_memory[tid]+=temp;
            __syncthreads();
            
        }

        //write result to global memory 
        output[index]=shared_memory[tid];
        
    }
}


int main(){
  const int N =16;
  const int block_size=8;
  
  int h_input[N]={
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
  };
  int h_output[N];

  int *d_input,*d_output;
  size_t size=N*sizeof(int);
  
  cudaMalloc(&d_input,size);
  cudaMalloc(&d_output,size);

  cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice);

  partial_sum_kernel<<<N/block_size,block_size,block_size*sizeof(int)>>>(d_input,d_output,N);
  
  
  cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost);

  printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);

  return 0;
}