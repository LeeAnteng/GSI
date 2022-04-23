#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include "Util.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <stdio.h>  
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define N 10
using namespace std;
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__device__ __managed__ unsigned qnum[N];
__device__ unsigned* temp_res[N];
__device__ unsigned b[N];
__constant__ unsigned *aa;
void initGPU(int dev)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
	//NOTE: 48KB shared memory per block, 1024 threads per block, 30 SMs and 128 cores per SM
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
               devProps.name, devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate,
			   devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
    }
	cout<<"GPU selected"<<endl;
	//GPU initialization needs several seconds, so we do it first and only once
	//https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/
	int* warmup = NULL;
	/*unsigned long bigg = 0x7fffffff;*/
	/*cudaMalloc(&warmup, bigg);*/
	/*cout<<"warmup malloc"<<endl;*/
    //NOTICE: if we use nvprof to time the API calls, we will find the time of cudaMalloc() is very long.
    //The reason is that we do not add cudaDeviceSynchronize() here, so it is asynchronously and will include other instructions' time.
    //However, we do not need to add this synchronized function if we do not want to time the API calls
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
    //heap corruption for 3 and 4
	/*size_t size = 0x7fffffff;*/    //size_t is unsigned long in x64
    unsigned long size = 0x7fffffff;   //approximately 2G
    /*size *= 3;   */
    size *= 4;
	/*size *= 2;*/
	//NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
	/*cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);*/
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;

	// Runtime API
	// cudaFuncCachePreferShared: shared memory is 48 KB
	// cudaFuncCachePreferEqual: shared memory is 32 KB
	// cudaFuncCachePreferL1: shared memory is 16 KB
	// cudaFuncCachePreferNone: no preference
	/*cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared)*/
	//The initial configuration is 48 KB of shared memory and 16 KB of L1 cache
	//The maximum L2 cache size is 3 MB.
	//also 48 KB read-only cache: if accessed via texture/surface memory, also called texture cache;
	//or use _ldg() or const __restrict__
	//4KB constant memory, ? KB texture memory. cache size?
	//CPU的L1 cache是根据时间和空间局部性做出的优化，但是GPU的L1仅仅被设计成针对空间局部性而不包括时间局部性。频繁的获取L1不会导致某些数据驻留在cache中，只要下次用不到，直接删。
	//L1 cache line 128B, L2 cache line 32B, notice that load is cached while store not
	//mmeory read/write is in unit of a cache line
	//the word size of GPU is 32 bits
    //Titan XP uses little-endian byte order
}

__device__ unsigned low_bound(unsigned target, unsigned* array, unsigned len) {
    unsigned left = 0, right = len - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (array[mid] >= target) right = mid;
        else left = mid + 1;
    }
    return left;
}
void exclusive_sum(unsigned* d_array, unsigned size)
{
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL; //must be set to distinguish two phase
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    cudaFree(d_temp_storage);
}

__global__ void test_kernel () {
    
   int idx = threadIdx.x;
   if (idx >= N) return;
   if (idx == 0) {
       unsigned* res = (unsigned*)malloc(sizeof(unsigned) * N);
       for (int i = 0; i < N; i++)
       res[i] = 3*i;
       //temp_res[idx] = res;
       free(res);
    //    for (int i = 0; i < N; i++)
    //    printf("%d ",res[i]);
       //printf("device:%p\n", temp_res[idx]);
   }


    // __shared__ unsigned s[N];
    // unsigned nei_start, nei_end;
    // __shared__ bool flag[N];
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if(idx == 0) {
    //     nei_start = low_bound(target, d_array, len);
    //     nei_end = low_bound(target + 1, d_array, len);
    //     printf("start:%d, end:%d\n", nei_start, nei_end);
    //     flag[nei_start] = false;
    //     printf("flag:%d\n",flag[nei_start]);
    // }

}
__global__ void test2_kernel(unsigned* d_array1, unsigned** d_array2, unsigned a) {
    int id = threadIdx.x;
    if (id >= N) return;
    d_array1[id] = d_array2[a][id];
}
__global__ void test3_kernel() {
    unsigned tid = threadIdx.x;
    unsigned block_id = blockIdx.x;
    if(tid == 0) printf("block_id = %d",block_id);
    __shared__ unsigned* res ;
    if(tid == 0){
            printf("res address = %p\n", res);
        res = (unsigned*) malloc(sizeof(unsigned) * 1024);
    printf("res address = %p\n", res);
    }
    __syncthreads();
    if (tid < 100)
    printf("res[%d] = %d\n", tid, res[tid]);
    // b[tid] |= 31;
    // atomicAdd(&b,1);
    // __syncthreads();
    // if (tid == 0) printf("b = %d\n",b);
}
/**
 * @brief 
 *  这个实验性的 用一个block计算 长度为 length数组的前缀和，Input为长度为n 的输入数组，只保证一个block计算前缀和
 *  output则是长度为n的Inpute 前缀和 pre_sum 前缀和 sum[i]= a[0]+a[1]+a[2]+..a[i]
 * @tparam index_t 
 * @tparam value_t 
 * @tparam warp_size  =32 
 * @param Input 
 * @param Output 
 * @param length 
 */
# define warp_size 32
__global__ void pre_sum_block(unsigned * flag,unsigned length)
{
    //const int thid = blockDim.x*blockIdx.x + threadIdx.x; // 总的线程
    const int tid=threadIdx.x;
    const int wrapId = tid / 32;
    const int wraps =blockDim.x / 32; // wraps<=32
    const int laneId = tid & (32-1);// 取二进制最后五位，是 threadIdx对32取模的结果。

    if(tid>=length) return;
    // 越界
    unsigned val = flag[tid]; // 每个线程的 负责一个数据，本地寄存器上
    __shared__ unsigned pre_sum_block [32]; // 每个wrap的最后一个前缀和放在上面
    // const int iters = 
    // 计算 wrap内的前缀和
    #pragma unroll 5
    for(int delta=1;delta<32;delta=delta*2) // 因为warp_size=32，否则应该是 delta< log2f(warp_size)
    {
        
         unsigned temp=__shfl_up_sync(0xFFFFFFFF,val,delta,32);
         if (laneId >=delta)
             val += temp;
        
    }
    // wrap是隐式同步的，限制每个wrap单独计算了前缀和
    if( laneId == 32-1)
    {
        // 一个wrap最后一个数
        pre_sum_block[wrapId]=val;
    }
    // 对shared memory的数求前缀和 ,wraps肯定是少于32的
    __syncthreads();// block内同步

    if(tid<32) // 取第一个wrap对pre_sum_block计算
    {
        unsigned warp_share_val = tid<wraps ?  pre_sum_block[tid] :0;
        #pragma unroll 5
        for(int delta=1;delta<32;delta=delta*2) // 因为warp_size=32，否则应该是 delta< log2f(warp_size)
        {
            unsigned temp=__shfl_up_sync(0xFFFFFFFF,warp_share_val,delta,32);
            if (laneId >=delta)
                warp_share_val += temp;
        }

        if(tid<wraps) 
            pre_sum_block[tid]= warp_share_val; // 每个wrap最后一个前缀和组成共享数组 的前缀和

    }
    __syncthreads();// block内同步，因为不同的warp要读share_memory
    if(wrapId>=1)  // 这里是 >=
    {
        //取wrap左边一个数
        val+=pre_sum_block[wrapId-1];
    }
    if (tid == 0) flag[tid] = 0;
    flag[tid + 1]=val;
}
# define BLOCK_SIZE 1024
__global__ void work_efficient_scan_kernel(unsigned *X, int InputSize) {
// XY[2*BLOCK_SIZE] is in shared memory
  __shared__ unsigned XY[BLOCK_SIZE * 2];
    int i =  threadIdx.x;
    if (i < InputSize) {XY[threadIdx.x] = X[i];}
    
  // the code below performs iterative scan on XY    
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
      __syncthreads();
      int index = (threadIdx.x+1)*stride*2 - 1; 
    if(index < 2*BLOCK_SIZE)
        XY[index] += XY[index - stride];//index is alway bigger than stride
    __syncthreads();
  }
  // threadIdx.x+1 = 1,2,3,4....
  // stridek index = 1,3,5,7...
  
  
  for (unsigned int stride = BLOCK_SIZE/2; stride > 0 ; stride /= 2) {
    __syncthreads();
      int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE)
        XY[index + stride] += XY[index];
    
  }
  __syncthreads();
  if (i < InputSize) {
      if (i == 0) X[i] = 0;
      else X[i] = XY[i - 1];
  }
//   X[i] = XY[threadIdx.x];
}

int main()
{
    // unsigned *d_array;
    // cudaMalloc(&d_array, sizeof(unsigned) * N);
    // unsigned* h_array = new unsigned[N];
    // for (int i = 0;i < N; i++) {
    //     h_array[i] = i*3;
    // }
    // cudaMemcpy(d_array, h_array, sizeof(unsigned) * N, cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(aa,&d_array,sizeof(unsigned*));
    test3_kernel<<<200,1024>>>();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    // cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, dev);
    // std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    // std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    // std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    // std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    // std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    //initGPU(0);
    // int GRIDSIZE = 10;
    // int BLOCKSIZE = 128;
    // dim3 grid(1024);
    // dim3 block(1024);
    // cudaMalloc(&d_array, sizeof(unsigned) * N);
    // for (int i = 0; i < N ;i++) {
    //     h_array[i] = i;
    // }
    // cudaMemcpy(d_array, h_array, sizeof(unsigned) * N, cudaMemcpyHostToDevice);

    // printf("ok1");
    // test_kernel <<<GRIDSIZE, BLOCKSIZE>>>(d_array);
    // cudaDeviceSynchronize();
    // printf("ok2");
    // cudaMalloc(&d_array, sizeof(unsigned) * N);
    // for (int i = 0; i < N ;i++) {
    //     h_array[i] = 2;
    // }
    // cudaMemcpy(d_array, h_array, sizeof(unsigned) * N, cudaMemcpyHostToDevice);
    // long t1 = Util::get_cur_time();
    
    // // thrust::device_ptr<unsigned> dev_ptr(d_array);
	// // thrust::exclusive_scan(dev_ptr, dev_ptr+N, dev_ptr);
    // exclusive_sum(d_array, N);
    // long t2 = Util::get_cur_time();
    // printf("scan time %ld\n", t2-t1);
    // cudaMemcpy(h_array, d_array, sizeof(unsigned) * N, cudaMemcpyDeviceToHost);
    // // for (int i = 0; i < N ;i++) {
    // //     cout<<h_array[i] <<" ";
    // // }
    // cudaFree(d_array);
    // delete[] h_array;
}