#include <cub/cub.cuh> 
#include "Match.h"

using namespace std;

__constant__ unsigned* c_row_offset;
__constant__ unsigned* c_col_index;
__constant__ unsigned* c_col_label;
__constant__ unsigned* c_col_offset;

__constant__ unsigned c_data_vertex_count;
__constant__ unsigned c_data_edge_count;
__constant__ unsigned c_link_pos[100];
__constant__ unsigned c_link_count;

__device__ __managed__ unsigned* temp_res[1024];
__device__ __managed__ unsigned temp_row_count[1024];

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


__global__ void
clean_kernel(unsigned result_row_num) {
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > result_row_num) return;
    temp_row_count[tid] = 0;
    if (temp_res[tid] != NULL) {
        free (temp_res[tid]);
        temp_res[tid] = NULL;
    }
}
__global__ void 
link_kernel(unsigned*d_result, unsigned* d_new_result, unsigned result_col_num, unsigned result_row_num) {
    __shared__ unsigned write_row_count;
    __shared__ unsigned begin_write_row;
    __shared__ unsigned row[64];
    unsigned block_id = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned warpid = tid >> 5;
    if(block_id >= result_row_num) return;
    if (tid == 0) {
        write_row_count = temp_row_count[block_id + 1] - temp_row_count[block_id];
        begin_write_row = temp_row_count[block_id];
    }
    __syncthreads();
    if (write_row_count == 0) return;
    if(tid <= result_col_num) {
        if (tid != result_col_num)
        row[tid] = d_result[block_id * result_col_num + tid];
        __syncthreads();
        for (int i = 0; i < write_row_count; i++) {
            if (tid != result_col_num)
            d_new_result[(begin_write_row + i) * (result_col_num + 1) + tid] = row[tid];
            else 
            d_new_result[(begin_write_row + i) * (result_col_num + 1) + tid] = temp_res[block_id][i];
        }
        __syncthreads();
    }
    // memcpy(row, d_result + block_id * result_col_num, result_col_num * sizeof(unsigned));
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


__global__  void 
join_kernel(unsigned label, unsigned* d_result, unsigned* d_candidate, unsigned result_row_num, unsigned result_col_num) {
    __shared__ unsigned init_neibors[1024];
    __shared__ unsigned flag[1024];
    __shared__ unsigned start_pos;
    __shared__ unsigned col_nei_start;
    __shared__ unsigned *res;
    //__shared__ unsigned isSelected[1024];
    __shared__ unsigned row[64];
    __shared__ unsigned label_start_idx , label_end_idx;
    __shared__ unsigned neiwithlabel_len;
    __shared__ unsigned final_res_len;
    unsigned block_id = blockIdx.x;
    unsigned tid = threadIdx.x;
    if (block_id >= result_row_num) return;
    if (tid < result_col_num)
    // memcpy(row, d_result + block_id * result_col_num, sizeof(unsigned)*result_col_num);
    row[tid] = d_result[block_id * result_col_num + tid];
    if (tid == 0) {
        start_pos = c_link_pos[0];
        unsigned vid = row[start_pos];
        col_nei_start = c_row_offset[vid];
        unsigned nei_count = c_row_offset[vid + 1] - c_row_offset[vid];
        unsigned* nei_label_begin = c_col_label + col_nei_start;
        unsigned* nei_vid_begin = c_col_index + col_nei_start;
        label_start_idx = low_bound(label, nei_label_begin, nei_count);
        label_end_idx = -1;
        if (nei_label_begin[label_start_idx] == label){
            label_end_idx = low_bound(label + 1, nei_label_begin, nei_count);
            neiwithlabel_len = label_end_idx - label_start_idx;
            // final_res_len = neiwithlabel_len;
            // memcpy(init_neibors, nei_vid_begin + label_start_idx, sizeof(unsigned) * neiwithlabel_len);
        }
        
    }
    __syncthreads();
    if(label_end_idx == -1) return;
    if (tid >= neiwithlabel_len) return;
    
    init_neibors[tid] = c_col_index[col_nei_start + label_start_idx + tid];
    flag[tid] = 1;
    //与C(u)作交集
    unsigned cur_vid = init_neibors[tid];
    unsigned a = cur_vid >> 5;
    unsigned b = cur_vid & 0x1f;
    b = 1 << b;
    if ((c_candidate[a] & b ) != b) {flag[tid] = 0;}
    __syncthreads();
    //减去已匹配的点
    for (int j = 0; j < result_col_num ; j++) {
        if(j == start_pos) continue;
        if(flag[tid] != 0 && row[j] == init_neibors[tid]) {flag[tid] = 0;}
    }
    __syncthreads();
    //与N(vi,l0)作交集
    for (int k = 1; k < c_link_count; k++) {
        if (flag[tid] != 0) {
            unsigned vid = row[c_link_pos[k]];
            unsigned col_nei_begin = c_row_offset[vid];
            unsigned nei_count = c_row_offset[vid + 1] -c_row_offset[vid];
            unsigned isFound = binary_search(init_neibors[tid], c_col_offset + col_nei_begin, nei_count);
            if (isFound == INVALID) {
                flag[tid] = 0;
            }
        }
    }


    //与N(vi,l0)作交集
    // for (int k = 1; k < c_link_count; k++){
    //     if (init_neibors[tid] != -1 ) {
    //         unsigned vid = row[c_link_pos[k]];
    //         unsigned col_nei_start = c_row_offset[vid];
    //         unsigned nei_count = c_row_offset[vid + 1] - c_row_offset[vid];
    //         unsigned* nei_label_begin = c_col_label + col_nei_start;
    //         unsigned* nei_vid_begin = c_col_index + col_nei_start;
    //         unsigned label_start = low_bound(label, nei_label_begin, nei_count);
    //         if (nei_label_begin[label_start] == label){
    //             unsigned label_end = low_bound(label + 1, nei_label_begin, nei_count);
    //             unsigned isFound = binary_search(init_neibors[tid], nei_vid_begin + label_start, label_end - label_start);
    //             if(isFound == INVALID) {
    //                 init_neibors[tid] = -1;final_res_len--;
    //             }
    //         }
    //         else {
    //             init_neibors[tid] = -1;final_res_len--;
    //         }
            
    //     } 
    // }
    
    __syncthreads();

    const int wrapId = tid / 32;
    const int wraps =blockDim.x / 32; // wraps<=32
    const int laneId = tid & (32-1);// 取二进制最后五位，是 threadIdx对32取模的结果。

    // if(tid>=length) return;
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
    __syncthreads();
    final_res_len = flag[neiwithlabel_len];
    if (final_res_len == 0) return;
    if (tid == 0) {
        res = (unsigned*) malloc(sizeof(unsigned) * final_res_len);
        temp_row_count[block_id] = final_res_len;
        temp_res[block_id] = res;
    }
    __syncthreads();
    unsigned pos = flag[tid];
    if (pos != flag[tid + 1]){
        res[pos] = init_neibors[tid];
    }


    // if(final_res_len <= 0) return;
    // if (tid == 0) {
    //     unsigned* res = (unsigned*)malloc(sizeof(unsigned)*final_res_len);
    //     unsigned i = 0;
    //     for (int j = 0; j < neiwithlabel_len; j++) {
    //         if (init_neibors[j] != -1) res[i++] = init_neibors[j];
    //     }
    //     temp_row_count[block_id] = final_res_len;
    //     temp_res[block_id] = res;
    // }

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
            //WARN: usig&vsig==usig is not right because the priority of == is higher than bitwise operation
            flag = ((usig & vsig) == usig)?1:0;
            //WARN: below is wrong because usig may have many 1s
            /*flag = ((usig & vsig) != 0)?1:0;*/
        }
    }
    d_status[i] = flag;
    printf("data id:%d, flag:%d\n", i, flag);
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
        printf("%d\n",i);
    }
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
    unsigned * d_row_offset, *d_col_nei_offset, *d_col_label_offset, *d_col_offset;
    unsigned vertex_count = this->data->vertex_num, edge_count = this->data->undir_edge_num;
    copyHtoD(d_row_offset, this->data->row_offset, sizeof(unsigned) * (vertex_count + 1));
    copyHtoD(d_col_nei_offset, this->data->col_nei_offset, sizeof(unsigned) * edge_count);
    copyHtoD(d_col_label_offset, this->data->col_label_offset, sizeof(unsigned) * edge_count);
    copyHtoD(d_col_offset, this->data->col_offset, sizeof(unsigned) * edge_count);

    cudaMemcpyToSymbol(c_row_offset, &d_row_offset, sizeof(unsigned*));
    cudaMemcpyToSymbol(c_col_index, &d_col_nei_offset, sizeof(unsigned*));
    cudaMemcpyToSymbol(c_col_label, &d_col_label_offset, sizeof(unsigned*));
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
	bool success = filter(score, qnum);
    long t2 = Util::get_cur_time();
	cout<<"filter used: "<<(t2-t1)<<"ms"<<endl;
    for(int i = 0; i < qsize; ++i)
    {
        cout<<qnum[i]<<" ";
    }cout<<endl;

#ifdef DEBUG
	cout<<"filter finished"<<endl;
#endif
	if(!success)
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
        
        
        cudaMemcpyToSymbol(c_link_count, &link_num, sizeof(unsigned));
        cudaMemcpyToSymbol(c_link_pos, link_pos, sizeof(unsigned) * link_num);

        bool success = this->join(node_label, link_pos, link_num, d_result, d_candidate, candidate_num, result_row_num, result_col_num);

    }
}

bool
Match::join(unsigned label, int* link_pos,int link_num, unsigned*& d_result, unsigned* d_candidate, unsigned d_cand_num, unsigned& result_row_num, unsigned& result_col_num)
{

   unsigned BLOCKSIZE = 1024;
   unsigned GRIDSIZE = (result_row_num + BLOCKSIZE - 1) / BLOCKSIZE;
   join_kernel<<<GRIDSIZE, BLOCKSIZE>>>(label, d_result, d_candidate, result_row_num, result_col_num);
   cudaDeviceSynchronize();

   exclusive_sum(temp_row_count, result_row_num + 1);
//    unsigned* h_temp_row_count = new unsigned[result_row_num + 1];
//    cudaMemcpyFromSymbol(h_temp_row_count, temp_row_count, sizeof(unsigned) * (result_row_num + 1));
   unsigned new_result_row_num = temp_row_count[result_row_num];
   if (new_result_row_num == 0) return false;
   unsigned temp_res_size = new_result_row_num * (result_col_num + 1);
   unsigned* d_new_result;
   cudaMalloc(&d_new_result, sizeof(unsigned) * temp_res_size);
   link_kernel<<<GRIDSIZE, BLOCKSIZE>>>(d_result, d_new_result, result_col_num, result_row_num);
   cudaDeviceSynchronize();
   clean_kernel<<<GRIDSIZE,BLOCKSIZE>>>(result_row_num);
   cudaDeviceSynchronize();
   result_row_num = new_result_row_num;
   result_col_num++;
   cudaFree(d_result);
   d_result = d_new_result;
   return true;
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