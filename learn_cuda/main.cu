#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh> 

#include <iostream>
#define N 10
using namespace std;
// using namespace thrust;
void exclusive_sum(unsigned* d_array, unsigned size)
{
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL; //must be set to distinguish two phase
    size_t   temp_storage_bytes = 0;
    printf("OK2\n");
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printf("OK3\n");

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    printf("OK4\n");
    cudaFree(d_temp_storage);
    // for (int i = 0; i < size; i++) {
    //     printf("%d ,", d_array[i]);
    // }
}

int main()
{
    unsigned h_arr[N];
    unsigned h_res[N + 1];
    cout<<"OK";
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;
    }
    unsigned* d_status = NULL;
    cudaMalloc(&d_status, sizeof(unsigned)*(N+1));
    
    cudaMemcpy(d_status, h_arr, sizeof(unsigned)*(N),cudaMemcpyHostToDevice);
    exclusive_sum(d_status, N+1);
    cudaMemcpy(h_res, d_status, sizeof(unsigned)*(N + 1), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%d ", h_res[i]);
    }
    printf("\n");
}
