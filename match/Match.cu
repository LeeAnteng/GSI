#include <cub/cub.cuh> 
#include "Match.h"

using namespace std;

__constant__ unsigned* c_row_offset;
// __constant__ unsigned* c_col_index;
// __constant__ unsigned* c_col_label;
__constant__ unsigned* c_col_offset;

__constant__ unsigned c_data_vertex_count;
__constant__ unsigned c_data_edge_count;
__constant__ unsigned c_link_pos;
__constant__ unsigned c_link_count;


__constant__ unsigned c_key_num;
__constant__ unsigned* c_result_tmp_pos;
__constant__ unsigned* c_result;
__constant__ unsigned* c_candidate;
/*__constant__ unsigned c_candidate_num;*/
__constant__ unsigned c_result_row_num;
__constant__ unsigned c_result_col_num;
/*__constant__ unsigned c_link_num;*/
/*__constant__ unsigned c_link_pos[MAX_DEGREE];*/
/*__constant__ unsigned c_link_edge[MAX_DEGREE];*/
// __constant__ unsigned c_link_pos;
__constant__ unsigned c_link_edge;
__constant__ unsigned c_signature[SIGNUM];



__device__ unsigned
binary_search(unsigned _key, unsigned* _array, unsigned _array_num)
{
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host
/*#if defined(__CUDA_ARCH__)*/
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
/*#else*/
/*#endif*/
}


__global__ void
candidate_kernel(unsigned* d_candidate, unsigned* d_candidate_tmp, unsigned candidate_num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= candidate_num)
	{
		return; 
	}
    //int atomicOr(int* address, int val);
    unsigned ele = d_candidate_tmp[i];
    unsigned num = ele >> 5;
    ele &= 0x1f;
    ele = 1 << ele;
    atomicOr(d_candidate+num, ele);
}

void 
Match::initGPU(int dev)
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
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
    unsigned long size = 0x7fffffff;   //approximately 2G
    /*size *= 3;   */
    size *= 4;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;}
__device__ uint32_t 
MurmurHash2(const void * key, int len, uint32_t seed) 
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


void 
Match::copyHtoD(unsigned*& d_ptr, unsigned* h_ptr, unsigned bytes)
{
    unsigned* p = NULL;
    cudaMalloc(&p, bytes);
    cudaMemcpy(p, h_ptr, bytes, cudaMemcpyHostToDevice);
    d_ptr = p;
    checkCudaErrors(cudaGetLastError());
}

void Match::exclusive_sum(unsigned* d_array, unsigned size)
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
Match::Match(Graph* _query, Graph* _data)
{
	this->query = _query;
	this->data = _data;
	id2pos = pos2id = NULL;
}
Match::~Match()
{
	delete[] this->id2pos;
}
__host__ float
compute_score(int size)
{
	return 0.0f +size;
}

bool
Match::score_node(float* _score, int* _qnum)
{
	bool success = true;
	for(int i = 0; i < this->query->vertex_num; ++i)
	{
		//BETTER: consider degree and substructure in the score
		if(_qnum[i] == 0)  // not found
		{
			/*d_score[i] = -1.0f;*/
			success = false;
            break;
		}
		else
		{
			_score[i] = compute_score(_qnum[i]);
			/**d_success = true;*/
		}
	}
	return success;
}
__global__ void
filter_kernel(unsigned* d_signature_table, unsigned* d_status, unsigned dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    unsigned flag = 1;
    /* if(i == 0)
    {
        unsigned t = 0x01020304;
        char* p = (char*)&t;
        printf("check endian: %d %d %d %d\n", *p, *(p+1), *(p+2), *(p+3));
    } */
    // if (i == 0) {
    //     for (int k = 0; k < SIGNUM; k++) {
    //     printf("签名第%d个字段：\n",k);
    //     for (int idx = 0; idx < dsize; idx++) {
    //         printf("vid=%d,sig=%d\n",idx,d_signature_table[k * dsize + idx]);
    //     }
    // }
    // }

    //以下信息需要关注，不然会出错
    //TODO+DEBUG: the first vertex label, should be checked via a==b
    for(int j = 0; j < SIGNUM; ++j)
    {
        unsigned usig = c_signature[j];
        unsigned vsig = d_signature_table[dsize*j+i];
        // if (i == 0)
        // printf("graph_id: %d, %d 's ,usign %d, vsign: %d\n",i,j,usig,vsig);
        //BETTER: reduce memory access here?
        if(flag)
        {
            if(j == 0) flag = (usig == vsig) ? 1: 0;//第一个lablel用特判
            else flag = ((usig & vsig) == usig)?1:0;
            //WARN: usig&vsig==usig is not right because the priority of == is higher than bitwise operation
            
            //WARN: below is wrong because usig may have many 1s
            /*flag = ((usig & vsig) != 0)?1:0;*/
        }
    }
    d_status[i] = flag;
    // printf("data id:%d, flag:%d\n", i, flag);
}


__global__ void
scatter_kernel(unsigned* d_status, unsigned* d_cand, unsigned dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    int pos = d_status[i];
    if(pos != d_status[i+1])
    {
        d_cand[pos] = i;
        // printf("%d\n",i);
    }
}

__global__ void
first_kernel(unsigned* d_result_tmp_pos) {
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned laneid = i & 0x1f;
    unsigned warpid = i >> 5; //group ID
	/*printf("compare %d and %d\n", i, result_row_num);*/
	if(warpid >= c_result_row_num)
	{
		return; 
	}
    unsigned* record = c_result+warpid*c_result_col_num;
    unsigned vid = record[c_link_pos];
    if (laneid == 0) {
        d_result_tmp_pos[warpid] = c_row_offset[vid+1] - c_row_offset[vid];
    }
}

__global__ void
second_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_num)
{
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];//row_pool
    __shared__ unsigned s_pool3[1024];
    __shared__ unsigned s_pool4[32];
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i & 0x1f;
    i = i >> 5; //group ID(warp index) within the whole kernel
	if(i >= c_result_row_num)
	{
		return; 
	}
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned gidx = threadIdx.x >> 5;  //warp index within this block
    if(idx < c_result_col_num)
    {
        s_pool2[bgroup+idx] = c_result[i*c_result_col_num+idx];
    }
    d_result_tmp += c_result_tmp_pos[i];
    unsigned vid = s_pool2[bgroup + c_link_pos];
    unsigned size = c_row_offset[vid + 1] - c_row_offset[vid];
    unsigned* list = c_col_offset + c_row_offset[vid];
    unsigned pos = 0;
    unsigned loop = size >> 5;
    size = size & 0x1f;
    unsigned pred, presum;
    unsigned cand_num = 0;
    s_pool4[gidx] = 0;
    
    for (int j = 0; j < loop;++j, pos += 32) {
        s_pool1[bgroup + idx] = list[pos + idx];
        unsigned k;
        //减去该行已匹配的点
        for(k = 0; k < c_result_col_num; ++k)
        {
            if(s_pool2[bgroup+k] == s_pool1[bgroup+idx])
            {
                break;
            }
        }
        pred = 0;
        //与C(u)作交集
        if(k == c_result_col_num)
        {
            unsigned num = s_pool1[bgroup+idx] >> 5;
            unsigned res = s_pool1[bgroup+idx] & 0x1f;
            res = 1 << res;
            if((c_candidate[num] & res) == res)
            {
                pred = 1;
            }
        }
        presum = pred;

        for(unsigned stride = 1; stride < 32; stride <<= 1)
        {
            //NOTICE: this must be called by the whole warp, not placed in the judgement
            unsigned tmp = __shfl_up(presum, stride);
            if(idx >= stride)
            {
                presum += tmp;
            }
        }
        //this must be called first, only in inclusive-scan the 31-th element is the sum
        unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
        //transform inclusive prefixSum to exclusive prefixSum
        presum = __shfl_up(presum, 1);
        //NOTICE: for the first element, the original presum value is copied
        if(idx == 0)
        {
            presum = 0;
        }
        if(pred == 1)
        {
            if(s_pool4[gidx]+presum < 32)
            {
                s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
            }
        }
        //flush 128B: one 4-segment writes is better than four 1-segment writes
        if(s_pool4[gidx]+total >= 32)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
            cand_num += 32;
            if(pred == 1)
            {
                unsigned pos = s_pool4[gidx] + presum;
                if(pos>=32)
                {
                    s_pool3[bgroup+pos-32] = s_pool1[bgroup+idx];
                }
            }
            s_pool4[gidx] = s_pool4[gidx] + total - 32;
        }
        else
        {
            //NOTICE:for a warp this is ok due to SIMD feature: sync read and sync write
            s_pool4[gidx] += total;
        }

    }
    //处理剩余的邻居
    presum = pred = 0; //init all threads to 0s first because later there is a judgement
    if(idx < size)
    {
        s_pool1[bgroup+idx] = list[pos+idx];
        unsigned k;
        for(k = 0; k < c_result_col_num; ++k)
        {
            if(s_pool2[bgroup+k] == s_pool1[bgroup+idx])
            {
                break;
            }
        }
        if(k == c_result_col_num)
        {
            unsigned num = s_pool1[bgroup+idx] >> 5;
            unsigned res = s_pool1[bgroup+idx] & 0x1f;
            res = 1 << res;
            if((c_candidate[num] & res) == res)
            {
                pred = 1;
            }
        }
        presum = pred;
    }
    for(unsigned stride = 1; stride < 32; stride <<= 1)
    {
        unsigned tmp = __shfl_up(presum, stride);
        if(idx >= stride)
        {
            presum += tmp;
        }
    }
    unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
    presum = __shfl_up(presum, 1);
    if(idx == 0)
    {
        presum = 0;
    }
    if(pred == 1)
    {
        if(s_pool4[gidx]+presum < 32)
        {
            s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
        }
    }
    unsigned newsize = s_pool4[gidx] + total;
    if(newsize >= 32)
    {
        d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
        cand_num += 32;
        if(pred == 1)
        {
            unsigned pos = s_pool4[gidx] + presum;
            if(pos>=32)
            {
                d_result_tmp[cand_num+pos-32] = s_pool1[bgroup+idx];
            }
        }
        cand_num += (newsize - 32);
    }
    else
    {
        if(idx < newsize)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
        }
        cand_num += newsize;
    }

    if(idx == 0)
    {
        d_result_tmp_num[i] = cand_num;
    }

}

__global__ void
join_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_num)
{
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];
    __shared__ unsigned s_pool3[1024];
    __shared__ unsigned s_pool4[32];
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i & 0x1f;
    i = i >> 5; //group ID
	if(i >= c_result_row_num)
	{
		return; 
	}

	unsigned res_num = d_result_tmp_num[i];
    //NOTICE: though invalid rows exist, but a warp will end directly here and not occupy resource any more(no divergence)
    if(res_num == 0)   //early termination
    {
        return;
    }
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned gidx = threadIdx.x >> 5;
    if(idx == 0)
    {
        s_pool2[bgroup+c_link_pos] = c_result[i*c_result_col_num+c_link_pos];
    }

    unsigned vid = s_pool2[bgroup+c_link_pos];
    unsigned list_num = c_row_offset[vid + 1] - c_row_offset[vid];
    unsigned* list = c_col_offset + c_row_offset[vid];
    d_result_tmp += c_result_tmp_pos[i];
    unsigned pos1 = 0, pos2 = 0;
    unsigned pred, presum;
    unsigned cand_num = 0;
    int choice = 0;
    s_pool4[gidx] = 0;
    while(pos1 < res_num && pos2 < list_num)
    {
        if(choice <= 0)
        {
            s_pool1[bgroup+idx] = INVALID;
            if(pos1 + idx < res_num)
            {
                s_pool1[bgroup+idx] = d_result_tmp[pos1+idx];
            }
        }
        if(choice >= 0)
        {
            s_pool2[bgroup+idx] = INVALID;
            if(pos2 + idx < list_num)
            {
                s_pool2[bgroup+idx] = list[pos2+idx];
            }
        }
        pred = 0;  //some threads may fail in the judgement below
        unsigned valid1 = (pos1+32<res_num)?32:(res_num-pos1);
        unsigned valid2 = (pos2+32<list_num)?32:(list_num-pos2);
        if(pos1 + idx < res_num)
        {
            pred = binary_search(s_pool1[bgroup+idx], s_pool2+bgroup, valid2);
            if(pred != INVALID)
            {
                pred = 1;
            }
            else
            {
                pred = 0;
            }
        }
        presum = pred;
        for(unsigned stride = 1; stride < 32; stride <<= 1)
        {
            unsigned tmp = __shfl_up(presum, stride);
            if(idx >= stride)
            {
                presum += tmp;
            }
        }
        unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
        presum = __shfl_up(presum, 1);
        if(idx == 0)
        {
            presum = 0;
        }
        if(pred == 1)
        {
            if(s_pool4[gidx]+presum < 32)
            {
                s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
            }
        }
        if(s_pool4[gidx]+total >= 32)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
            cand_num += 32;
            if(pred == 1)
            {
                unsigned pos = s_pool4[gidx] + presum;
                if(pos>=32)
                {
                    s_pool3[bgroup+pos-32] = s_pool1[bgroup+idx];
                }
            }
            s_pool4[gidx] = s_pool4[gidx] + total - 32;
        }
        else
        {
            s_pool4[gidx] += total;
        }

        //set the next movement
        choice = s_pool1[bgroup+valid1-1] - s_pool2[bgroup+valid2-1];
        if(choice <= 0)
        {
            pos1 += 32;
        }
        if(choice >= 0)
        {
            pos2 += 32;
        }
    }
    if(idx < s_pool4[gidx])
    {
        d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
    }
    cand_num += s_pool4[gidx];

    if(idx == 0)
    {
        d_result_tmp_num[i] = cand_num;
    }


}

__global__ void
link_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_pos, unsigned* d_result_tmp_num, unsigned* d_result_new)
{
    //BETTER:consider bank conflicts here, should we use column-oriented table for global memory and shared memory?
    //In order to keep in good occupancy(>=50%), the shared mem usage should <= 24KB for 1024-threads block
    __shared__ unsigned cache[1024];
    /*__shared__ unsigned s_pool[1024*5];  //the work poll*/
    //NOTICE: though a block can be synchronized, we should use volatile to ensure data is not cached in private registers
    //If shared mem(or global mem) is used by a warp, then volatile is not needed.
    //http://www.it1352.com/539600.html
    /*volatile __shared__ unsigned swpos[1024];*/
    __shared__ unsigned swpos[32];

	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    i >>= 5;
    //NOTICE: we should not use this if we want to control the whole block
    //(another choice is to abandon the border block)
	/*if(i >= c_result_row_num)*/
	/*{*/
		/*return; */
	/*}*/
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned idx = threadIdx.x & 0x1f;  //thread index within the warp
    unsigned gidx = threadIdx.x >> 5; //warp ID within the block

    unsigned tmp_begin = 0, start = 0, size = 0;
    if(i < c_result_row_num)
    {
        tmp_begin = d_result_tmp_pos[i];
        start = d_result_tmp_num[i];
        //NOTICE: the size is ok to be 0 here
        size = d_result_tmp_num[i+1] - start;
        start *= (c_result_col_num+1);
    }

    //Usage of Shared Memory: cache records only when size > 0
    if(idx == 0)
    {
        if(size > 0)
        {
            //NOTICE: we use a single thread to read a batch a time
            memcpy(cache+gidx*32, c_result+i*c_result_col_num, sizeof(unsigned)*c_result_col_num);
        }
    }
    unsigned curr = 0;
    unsigned* record = cache + gidx * 32;

    //Usage of Load Balance
    __syncthreads();
    //use a block to deal with tasks >=1024
    while(true)
    {
        if(threadIdx.x == 0)
        {
            swpos[0] = INVALID;
        }
        //NOTICE: the sync function is needed, but had better not use it too much 
        //It is costly, which may stop the running warps and replace them with other warps
        __syncthreads();
        if(size >= curr+1024)
        {
            swpos[0] = gidx;
        }
        __syncthreads();
        if(swpos[0] == INVALID)
        {
            break;
        }
        //WARN:output info within kernel function will degrade perfomance heavily
        //printf("FOUND: use a block!\n");
        unsigned* ptr = cache + 32 * swpos[0];
        if(swpos[0] == gidx)
        {
            swpos[1] = tmp_begin;
            swpos[2] = start;
            swpos[3] = curr;
            swpos[4] = size;
        }
        __syncthreads();
        //NOTICE: here we use a block to handle the task as much as possible
        //(this choice may save the work of preparation)
        //Another choice is only do 1024 and set size-=1024, later vote again
        while(swpos[3]+1023 <swpos[4])
        {
            unsigned pos = (c_result_col_num+1)*(swpos[3]+threadIdx.x);
            memcpy(d_result_new+swpos[2]+pos, ptr, sizeof(unsigned)*c_result_col_num);
            d_result_new[swpos[2]+pos+c_result_col_num] = d_result_tmp[swpos[1]+swpos[3]+threadIdx.x];
            if(threadIdx.x == 0)
            {
                swpos[3] += 1024;
            }
            __syncthreads();
        }
        if(swpos[0] == gidx)
        {
            curr = swpos[3];
        }
        __syncthreads();
    }
    __syncthreads();

    //combine the tasks of rows and divide equally
    //NOTICE: though we can combine even when the tasks of some row is very small, it is not good.
    //(the time of combining may be consuming compared to using exactly a warp for each row, when the size is nearly 32)

    while(curr < size)
    {
        //this judgement is fine, only causes divergence in the end
        if(curr+idx < size)
        {
            unsigned pos = (c_result_col_num+1)*(curr+idx);
            memcpy(d_result_new+start+pos, record, sizeof(unsigned)*c_result_col_num);
            d_result_new[start+pos+c_result_col_num] = d_result_tmp[tmp_begin+curr+idx];
        }
        curr += 32;
    }
    //BETTER: the implementation of memcpy() may be optimized for single thread with batch read/write
    //using a struct representing more bytes? or use vload4
}


bool
Match::filter(float* _score, int* _qnum) {
    int qsize = this->query->vertex_num, dsize = this->data->vertex_num;
    this->candidates = new unsigned*[qsize];
    unsigned* d_signature_table = NULL;
    int bytes = dsize * SIGBYTE;
    cudaMalloc(&d_signature_table, bytes);
    cudaMemcpy(d_signature_table, this->data->sig_table, bytes, cudaMemcpyHostToDevice);
   
    unsigned* d_status = NULL;
    cudaMalloc(&d_status, sizeof(unsigned)*(dsize+1));

    int BLOCK_SIZE = 1024;
	int GRID_SIZE = (dsize+BLOCK_SIZE-1)/BLOCK_SIZE;
    for (int i = 0; i <qsize; i++) {
        cudaMemcpyToSymbol(c_signature, this->query->sig_table+SIGNUM*i, SIGBYTE);
        filter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_signature_table, d_status, dsize);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        exclusive_sum(d_status, dsize+1);
        checkCudaErrors(cudaGetLastError());
        cudaMemcpy(&_qnum[i], d_status+dsize, sizeof(unsigned), cudaMemcpyDeviceToHost);
        printf("%d 's cands num %d\n",i, _qnum[i]);
        if(_qnum[i] == 0)
        {
            break;
        }
        unsigned* d_cand = NULL;
        cudaMalloc(&d_cand, sizeof(unsigned)*_qnum[i]);
        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_status, d_cand, dsize);
	    checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        this->candidates[i] = d_cand;

    }
    cudaFree(d_status);
	cudaFree(d_signature_table);

    //get the num of candidates and compute scores
	bool success = score_node(_score, _qnum);
	if(!success)
	{
#ifdef DEBUG
		cout<<"query already fail after filter"<<endl;
#endif
		return false;
	}

	return true;
}
void 
Match::match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map)
{
//NOTICE: device variables can not be assigned and output directly on Host
/*unsigned maxTaskLen = 0, minTaskLen = 1000000;*/
/*cudaMemcpyToSymbol(d_maxTaskLen, &maxTaskLen, sizeof(unsigned));*/
/*cudaMemcpyToSymbol(d_minTaskLen, &minTaskLen, sizeof(unsigned));*/
//CUDA device variable (can only de declared on Host, not in device/global functions), can be used in all kernel functions like constant variables
/*https://blog.csdn.net/rong_toa/article/details/78664902*/
/*cudaGetSymbolAddress((void**)&dp,devData);*/
/*cudaMemcpy(dp,&value,sizeof(float),cudaMemcpyHostToDevice);*/

	long t0 = Util::get_cur_time();
	// copyGraphToGPU();
    //在GPU端分配内存，存储data图结构
    unsigned * d_row_offset, *d_col_offset;
    unsigned vertex_count = this->data->vertex_num, edge_count = this->data->undir_edge_num;
    copyHtoD(d_row_offset, this->data->row_offset, sizeof(unsigned) * (vertex_count + 1));
    copyHtoD(d_col_offset, this->data->col_offset, sizeof(unsigned) * edge_count);

    cudaMemcpyToSymbol(c_row_offset, &d_row_offset, sizeof(unsigned*));
    cudaMemcpyToSymbol(c_col_offset, &d_col_offset, sizeof(unsigned*));    
    cudaMemcpyToSymbol(c_data_vertex_count, &vertex_count, sizeof(unsigned));
    cudaMemcpyToSymbol(c_data_edge_count, &edge_count, sizeof(unsigned));

	long t1 = Util::get_cur_time();
	cerr<<"copy graph used: "<<(t1-t0)<<"ms"<<endl;
#ifdef DEBUG
	cout<<"graph copied to GPU"<<endl;
#endif

	int qsize = this->query->vertex_num;
    assert(qsize <= 12);
	float* score = new float[qsize];
	/*float* d_score = NULL;*/
	/*cudaMalloc(&d_score, sizeof(float)*qsize);*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cout<<"assign score"<<endl;*/

	int* qnum = new int[qsize+1];
	/*int* d_qnum = NULL;*/
	/*cudaMalloc(&d_qnum, sizeof(int)*(qsize+1));*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cout<<"assign d_qnum"<<endl;*/

	/*cout<<"to filter"<<endl;*/
	bool success_filter = filter(score, qnum);
    long t2 = Util::get_cur_time();
	cout<<"filter used: "<<(t2-t1)<<"ms"<<endl;
    for(int i = 0; i < qsize; ++i)
    {
        cout<<qnum[i]<<" ";
    }cout<<endl;

#ifdef DEBUG
	cout<<"filter finished"<<endl;
#endif
	if(!success_filter)
	{
		delete[] score;
		delete[] qnum;
		result_row_num = 0;
		result_col_num = qsize;
		final_result = NULL;
		release();
		return; 
	}

    unsigned bitset_size = sizeof(unsigned) * Util::RoundUpDivision(this->data->vertex_num, sizeof(unsigned)*8);
    cout<<"data vertex num: "<<this->data->vertex_num<<" bitset size: "<<bitset_size<<"B"<<endl;
    //NOTICE: the bitset is very large, we should only keep one set at a time
    unsigned* d_candidate = NULL;  //candidate bitset
    cudaMalloc(&d_candidate, bitset_size);
    checkCudaErrors(cudaGetLastError());

    unsigned* d_summary = NULL;

#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
	cout<<"candidates prepared"<<endl;
#endif
	long t3 = Util::get_cur_time();
	cerr<<"build candidates used: "<<(t3-t2)<<"ms"<<endl;

    this->id2pos = new int[qsize];
	this->pos2id = new int[qsize];
	this->current_pos = 0;
	memset(id2pos, -1, sizeof(int)*qsize);
	memset(pos2id, -1, sizeof(int)*qsize);
	//select the minium score and fill the table
	int idx = this->get_minimum_idx(score, qsize);
	cout<<"start node found: "<<idx<<" "<<this->query->vertices[idx].label<<" candidate size: "<<qnum[idx]<<endl;

    //intermediate table of join results
	result_row_num = qnum[idx];
	result_col_num = 1;
	unsigned* d_result = this->candidates[idx];  
	cout<<"intermediate table built"<<endl;

    
    bool success;
    for (int step = 1; step < qsize; step++) {
        cout<<"this is the "<<step<<" round"<<endl;

        long t4 = Util::get_cur_time();
        // update the scores of query nodes
        update_score(score, qsize, idx);
        long t5 = Util::get_cur_time();
        cerr<<"update score used: "<<(t5-t4)<<"ms"<<endl;

        int idx2 = this->get_minimum_idx(score, qsize);

        long t6 = Util::get_cur_time();
        cerr<<"get minimum idx used: "<<(t6-t5)<<"ms"<<endl;
    /*#ifdef DEBUG*/
        unsigned node_label = this->query->vertices[idx2].label;
        cout<<"next node to join: "<<idx2<<" "<<node_label<<" candidate size: "<<qnum[idx2]<<endl;
    /*#endif*/
    //acquire the edge linkings on CPU, and pass to GPU
		int *link_pos, *link_edge, link_num;
		this->acquire_linking(link_pos, link_edge, link_num, idx2);
        long t7 = Util::get_cur_time();
        cerr<<"acquire linking used: "<<(t7-t6)<<"ms"<<endl;
        

        long tmp1 = Util::get_cur_time();
        //build the bitset
		checkCudaErrors(cudaGetLastError());
        cudaMemset(d_candidate, 0, bitset_size);
		checkCudaErrors(cudaGetLastError());
		int candidate_num = qnum[idx2];
        candidate_kernel<<<Util::RoundUpDivision(candidate_num, 1024), 1024>>>(d_candidate, this->candidates[idx2], candidate_num);
		checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
        long tmp2 = Util::get_cur_time();
        cout<<"candidate kernel used: "<<(tmp2-tmp1)<<"ms"<<endl;
        cudaFree(this->candidates[idx2]);
		checkCudaErrors(cudaGetLastError());
        
        
        printf("step = %d, result_row_num = %d\n",step, result_row_num);
        success = this->join(node_label, link_pos, link_num, d_result, d_candidate, candidate_num, result_row_num, result_col_num);
        
        #ifdef DEBUG
	    checkCudaErrors(cudaGetLastError());
        #endif
        delete[] link_pos;
        if (!success) break;
        idx = idx2;
        
    }
    long t8 = Util::get_cur_time();
    cout<<"total time cost:"<<t8 - t1<<"ms"<<endl;
    if (success) {
        final_result = new unsigned[result_row_num * result_col_num];
        cudaMemcpy(final_result, d_result, sizeof(unsigned) * result_row_num * result_col_num, cudaMemcpyDeviceToHost);
    }
    else {
        final_result = NULL;
        result_row_num = 0;
        result_col_num = qsize;
    }
    id_map = this->id2pos;

    delete[] score;
    delete[] qnum;
    release();
}

bool
Match::join(unsigned label, int* link_pos,int link_num, unsigned*& d_result, unsigned* d_candidate, unsigned d_cand_num, unsigned& result_row_num, unsigned& result_col_num)
{
    unsigned sum;
	unsigned* d_result_tmp = NULL;
	unsigned* d_result_tmp_pos = NULL;
	unsigned* d_result_tmp_num = NULL;
	cudaMalloc(&d_result_tmp_pos, sizeof(unsigned)*(result_row_num+1));
	cudaMalloc(&d_result_tmp_num, sizeof(unsigned)*(result_row_num+1));
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
    cudaMemcpyToSymbol(c_result, &d_result, sizeof(unsigned*));
	cudaMemcpyToSymbol(c_candidate, &d_candidate, sizeof(char*));
	/*cudaMemcpyToSymbol(c_candidate_num, &num, sizeof(unsigned));*/
	cudaMemcpyToSymbol(c_result_row_num, &result_row_num, sizeof(unsigned));
	cudaMemcpyToSymbol(c_result_col_num, &result_col_num, sizeof(unsigned));
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (result_row_num*32+BLOCK_SIZE-1)/BLOCK_SIZE;
   #ifdef DEBUG
        cout<<"now to do join kernel "<<result_row_num<<" "<<result_col_num<<" "<<GRID_SIZE<<" "<<BLOCK_SIZE<<endl;
#endif
	long begin = Util::get_cur_time();
    for (int i = 0; i < link_num; i++) {
        cudaMemcpyToSymbol(c_link_pos, link_pos+i, sizeof(unsigned));
        cudaMemcpyToSymbol(c_result_tmp_pos, &d_result_tmp_pos, sizeof(unsigned*));
        cout<<"the "<<i<<"-th edge"<<endl;
        if (i == 0) {
            first_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp_pos);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
            cout<<"first kernel finished"<<endl;

            /*thrust::device_ptr<unsigned> dev_ptr(d_result_tmp_pos);*/
            /*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
            exclusive_sum(d_result_tmp_pos, result_row_num+1);

            cudaMemcpy(&sum, &d_result_tmp_pos[result_row_num], sizeof(unsigned), cudaMemcpyDeviceToHost);
            cout<<"To malloc on GPU: "<<sizeof(unsigned)*sum<<endl;
            assert(sum < 2000000000);  //keep the bytes < 8GB
            cudaMalloc(&d_result_tmp, sizeof(unsigned)*sum);
            checkCudaErrors(cudaGetLastError());
            second_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_num);
        }
        {
            join_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_num);
        }
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        cout<<"iteration kernel finished"<<endl;
    }
    long end = Util::get_cur_time();
	cerr<<"join_kernel used: "<<(end-begin)<<"ms"<<endl;
#ifdef DEBUG
	cout<<"join kernel finished"<<endl;
#endif

	/*thrust::device_ptr<unsigned> dev_ptr(d_result_tmp_num);*/
	/*//link the temp result into a new table*/
	/*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
    exclusive_sum(d_result_tmp_num, result_row_num+1);
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	/*sum = thrust::reduce(dev_ptr, dev_ptr+result_row_num);*/
	cudaMemcpy(&sum, d_result_tmp_num+result_row_num, sizeof(unsigned), cudaMemcpyDeviceToHost);
	//BETTER: judge if success here
	/*cout<<"new table num: "<<sum<<endl;*/
	/*int tmp = 0;*/
	/*for(int i = 0; i < result_row_num; ++i)*/
	/*{*/
		/*cudaMemcpy(&tmp, d_result_tmp_num+i, sizeof(int), cudaMemcpyDeviceToHost);*/
		/*cout<<"check tmp: "<<tmp<<endl;*/
	/*}*/
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif

	unsigned* d_result_new = NULL; 
	if(sum > 0)
	{
		cudaMalloc(&d_result_new, sizeof(unsigned)*sum*(result_col_num+1));
#ifdef DEBUG
		checkCudaErrors(cudaGetLastError());
#endif
		/*BLOCK_SIZE = 512;*/
		/*GRID_SIZE = (result_row_num+BLOCK_SIZE-1)/BLOCK_SIZE;*/
		//BETTER?: combine into a large array(value is the record id) and link per element
		long begin = Util::get_cur_time();
		link_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_pos, d_result_tmp_num, d_result_new);
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();
		long end = Util::get_cur_time();
#ifdef DEBUG
		cerr<<"link_kernel used: "<<(end-begin)<<"ms"<<endl;
#endif
#ifdef DEBUG
		checkCudaErrors(cudaGetLastError());
#endif
	}
    //if the original result table is exactly the first candidate set, then here also delete it
    cudaFree(d_result);
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	d_result = d_result_new;

	cudaFree(d_result_tmp);  
	cudaFree(d_result_tmp_pos);  
	cudaFree(d_result_tmp_num);  
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	result_col_num++;
	result_row_num = sum;

	if(result_row_num == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}


void
Match::acquire_linking(int*& link_pos, int*& link_edge, int& link_num, int idx)
{
	vector<int> tmp_vertex;
    int i, qsize = this->query->vertex_num;
    bool* isNeiOfidx = new bool[qsize];
    memset(isNeiOfidx, 0, sizeof(bool) * qsize);
    const auto& neibors = this->query->vertices[idx].neighbors;
    int nei_size = neibors.size();
    for (int i = 0; i < nei_size; i++) {
        int vid = neibors[i].id;
        isNeiOfidx[vid] = true;
    }
    for (i = 0; i < this->current_pos; i++) {
        int vid = this->pos2id[i];
        if (isNeiOfidx[vid]) {
            tmp_vertex.push_back(i);
        }
    }
    delete[] isNeiOfidx;
    link_num = tmp_vertex.size();
    link_pos = new int[link_num];
    for (int i = 0; i < link_num; i++) {
        link_pos[i] = tmp_vertex[i];
    }
}


void
Match::update_score(float* _score, int qsize, int _idx)
{
	//BETTER: acquire it from edge label frequence: p = (P2num)/T, divide in and out edge?
	//score the node or the edge?? how about m*n*p, cost of the current step and the join result's size(cost of the next step)
	float p = 0.9f;
	/*float p = 0.1f;*/
	/*float p = 0.5f;*/
    int nei_size = this->query->vertices[_idx].neighbors.size();
    int i, j;
    for (i = 0; i < nei_size; i++) {
        j = this->query->vertices[_idx].neighbors[i].id;
        _score[j] *= p;
    }
}

int
Match::get_minimum_idx(float* score, int qsize)
{
    float* min_ptr = NULL;
    float minscore = FLT_MAX;
    //choose the start node based on score
    if(this->current_pos == 0)
    {
        min_ptr = min_element(score, score+qsize);
        minscore = *min_ptr;
    }

    for(int i = 0; i < this->current_pos; ++i)
    {
        int id = this->pos2id[i];
        int nei_size = this->query->vertices[id].neighbors.size();
        for (int j = 0; j < nei_size; j++) {
            int id2 = this->query->vertices[id].neighbors[j].id;
            if (score[id2] < minscore) {
                minscore = score[id2];
                min_ptr = score + id2;
            }
        }
    }
	int min_idx = min_ptr - score;
    //set this ID to maximum so it will not be chosed again
	memset(min_ptr, 0x7f, sizeof(float));
	/*thrust::device_ptr<float> dev_ptr(d_score);*/
	/*float* min_ptr = thrust::raw_pointer_cast(thrust::min_element(dev_ptr, dev_ptr+qsize));*/
	/*int min_idx = min_ptr - d_score;*/
	/*//set this node's score to maximum so it won't be chosed again*/
	/*cudaMemset(min_ptr, 0x7f, sizeof(float));*/

	//NOTICE: memset is used per-byte, so do not set too large value, otherwise it will be negative
	//http://blog.csdn.net/Vmurder/article/details/46537613
	/*cudaMemset(d_score+min_idx, 1000.0f, sizeof(float));*/
	/*float tmp = 0.0f;*/
	/*cout<<"to check the score: ";*/
	/*for(int i = 0; i < qsize; ++i)*/
	/*{*/
		/*cudaMemcpy(&tmp, d_score+i, sizeof(float), cudaMemcpyDeviceToHost);*/
		/*cout<<tmp<<" ";*/
	/*}cout<<endl;*/
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif

	this->add_mapping(min_idx);
	return min_idx;
}


void
Match::release()
{
	delete[] this->pos2id;
    delete[] this->candidates;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
}
inline void 
Match::add_mapping(int _id)
{
	pos2id[current_pos] = _id;
	id2pos[_id] = current_pos;
	this->current_pos++;
}