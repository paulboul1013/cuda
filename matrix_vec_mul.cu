%%cuda
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrix_vec_mult(const float* mat_a,const float* vec_b,float *result_c,int dim){
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx<dim){
        float acc=0.0f;

        const float *row_ptr=mat_a+idx*dim;

        for(int k=0;k<dim;k++){
            acc+=row_ptr[k]*vec_b[k];
        }

        result_c[idx]=acc;
    }
                                                                                              
}

int main() {
  const int dim=10;
  
  float *mat_a=new float[dim*dim];
  float *vec_b=new float[dim];
  float *result_c=new float[dim];

  for(int i=0;i<dim;i++){
      vec_b[i]=2.0f;
      result_c[i]=0.0f;
      for(int j=0;j<dim;j++){
          mat_a[i*dim+j]=1.0f;
      }
  }

  float *d_mat,*d_vec,*d_res;
  cudaMalloc(&d_mat,dim*dim*sizeof(float));
  cudaMalloc(&d_vec,dim*sizeof(float));
  cudaMalloc(&d_res,dim*sizeof(float));

  cudaMemcpy(d_mat,mat_a,dim*dim*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec,vec_b,dim*sizeof(float),cudaMemcpyHostToDevice);

  const int threads_per_block=128;
  int blocks=(dim+threads_per_block-1)/threads_per_block;

  matrix_vec_mult<<<blocks,threads_per_block>>>(d_mat,d_vec,d_res,dim);

  //wait for gpu to finish
  cudaDeviceSynchronize();


  cudaMemcpy(result_c,d_res,dim*sizeof(float),cudaMemcpyDeviceToHost);

  cout<<"Matrix A:\n";
  for(int i=0;i<dim;i++){
      for(int j=0;j<dim;j++){
          cout<<mat_a[i*dim+j]<<" ";
      }
      cout<<endl;
  }

  cout<<"vector B:";
  for(int i=0;i<dim;i++){
      cout<<vec_b[i]<<" ";
  }
  cout<<endl;

  cout<<"result C:";
  for(int i=0;i<dim;i++) {
      cout<<result_c[i]<<" ";
  }
  cout<<endl;

  cudaFree(d_mat);
  cudaFree(d_vec);
  cudaFree(d_res);

  delete[] mat_a;
  delete[] vec_b;
  delete[] result_c;

  
  return 0;
}