#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#define N 100
using namespace std;
__global__ void test_shfl(int A[], int B[])
{
    int a[(int)1e9];
    int tid = threadIdx.x;
    int best = B[tid];
   
    best = __shfl_up(best, 3);
    A[tid] = best;
}
__global__ void test_uint2(uint2 a[], unsigned n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= n) {
        printf("idle\n");
        return;
    }
    printf("d_uA[%d].x: %d \n",i, a[i].x);
}
uint32_t MurmurHash2(const void * key, int len, uint32_t seed) 
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    // Initialize the hash to a 'random' value
    uint32_t h = seed ^ len;
    // Mix 4 bytes at a time into the hash
    const unsigned char * data = (const unsigned char *) key;
    while (len >= 4) 
    {
        uint32_t k = *(uint32_t*) data;
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }
    // Handle the last few bytes of the input array
    switch (len) 
    {
        case 3:
            h ^= data[2] << 16;
        case 2:
            h ^= data[1] << 8;
        case 1:
          h ^= data[0];
          h *= m;
    };
    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}


int main()
{
    int *A,*Ad, *B, *Bd;
    uint2 *uA, *d_uA;
    int x = -1;
    unsigned y = x;
    cout<<"y="<<y<<endl;
    unsigned a = 5;
    unsigned b[2] = {1,3};
    unsigned pos1 = MurmurHash2(&y, 4, 17);
    unsigned pos2 = MurmurHash2(b, 8, 17);
    cout<<pos1<<" "<<pos2;
    int n = 32;
    int size = n * sizeof(int);
 
    // CPU端分配内存
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    uA = (uint2*)malloc(sizeof(uint2) * N);
 
    for (int i = 0; i < n; i++)
   {   
      B[i] = rand()%101;
      std::cout << B[i] << " ";
   }
   for (int i = 0; i < N; i++) {
       uA[i].x = i;
       uA[i].y = i + 1;
   }
   cout<<endl;
   
    std::cout <<"----------------------------" << std::endl;
   
    // GPU端分配内存
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc(&d_uA, sizeof(uint2) * N);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_uA, uA, sizeof(uint2) * N,cudaMemcpyHostToDevice);
    // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
    dim3 dimBlock(128);
    dim3 dimGrid(1000);
 
    // 执行kernel
    const auto t1 = std::chrono::system_clock::now();
 
    //test_shfl << <1, 32 >> > (Ad,Bd);
    //test_uint2<<<1, 1000>>> (d_uA, N);
    cudaMemcpy(A, Ad, size, cudaMemcpyDeviceToHost);
 
    // 校验误差
    float max_error = 0.0;
    for (int i = 0; i <     32; i++)
    {
       
            std::cout << A[i] << " ";
    }
    cout<<endl;
 
    std::cout << "max error is " << max_error << std::endl;
 
    // 释放CPU端、GPU端的内存
    free(A);
    free(B);   
    cudaFree(Ad);
    cudaFree(Bd);
 
    return 0;
}