# cuda

## introduece
for personal learning cuda kernel programming and gpu architecture knowledge

## environment
using online colab

## colab setting cuda environment

click 執行階段->變更執行階段類型->選擇 GPU->T4 GPU->儲存

---

tools install and enviroment check
```c
!nvidia-smi //gpu info
!nvcc --version //cuda version
!pip install nvcc4jupyter //for using %%cuda in colab
%load_ext nvcc4jupyter 
```

## usage cuda in colab
template code
```c
%%cuda
#include <iostream>
int main() {
    std::cout<<"hello world"<<std::endl;
    return 0;
}
```