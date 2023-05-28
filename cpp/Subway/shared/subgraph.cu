
#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>


template <class E>
Subgraph<E>::Subgraph(uint num_nodes, uint num_edges, ull max_size)
{
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if(error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();
	/*long long percentage = 60;
	long long reserve_memory =((24*(1UL<<30))/100)*percentage + (5*(1UL<<30));  
	max_partition_size = 0.1 * ((dev.totalGlobalMem-reserve_memory) - 8*4*num_nodes) / sizeof(E);*/

	//ull max_size = 25445793792;
	//ull init_size = 1073741824;

	ull mem_free, mem_avail;

    cudaMemGetInfo((size_t *)&mem_free, (size_t *)&mem_avail);

    cout << "\n Available memory " << mem_avail << " \n";

	cout << "\n Free memory " << mem_free << " \n";

    /* max_partition_size = 0.5 * ( mem_free  - (6*(1UL<<30))) / sizeof(E);*/

	//max_partition_size = floor((double)(mem_free*1.0 / max_size) * init_size);

	
	//max_partition_size = 0.9 * ((dev.totalGlobalMem - 8*4*num_nodes) / sizeof(E);
	//max_partition_size = 1073741824/4;

	//max_partition_size = (mem_free - (size_t)4665049760)/8;
	//if(max_partition_size < 5000000)
	//	max_partition_size = 5000000;
	//max_partition_size = 10000000;
	//cout<<"size of uint "<<sizeof(uint64_t)<<"\n";
	//cout<<13*(8*num_nodes)<<" "<<num_nodes<<" "<<8*num_nodes<<"\n";
	/*ull ds_to_consider = num_nodes;
	ds_to_consider *= 8;
	ds_to_consider *= 12;
	

	if(mem_free > ds_to_consider)
		max_partition_size = (mem_free - ds_to_consider)/48;
	else
		max_partition_size = 10000000;*/
	
	//max_partition_size = 1073741824;
		//num_nodes = 41652230;
	/*ull ds_to_consider = 94*num_nodes;
	ull degree = 6;
	ull denom = 24/degree + 8;
	ull thrust_call_size = 8*(num_nodes + 10); // 1GB
	ull summ =  thrust_call_size + ds_to_consider;
	//cout<<0.9*mem_free<<"\n";
	//cout<<summ<<"\n";
	if(0.9*mem_free > summ){
		ull numerator = 0.9*mem_free - summ;
		max_partition_size = 0.9*(numerator / denom);
	}
	else
		max_partition_size = 10000000;*/
	max_partition_size = max_size;
	//max_partition_size = 0.3*mem_free;
    cout << "\n Max Partition size : " << max_partition_size << "\n";

	//exit(0);
	if(max_partition_size > DIST_INFINITY)
		max_partition_size = DIST_INFINITY;
	
	//cout << "Max Partition Size: " << max_partition_size << endl;
	
	this->num_nodes = num_nodes;
	this->num_edges = num_edges;
	
	gpuErrorcheck(cudaMallocHost(&activeNodes, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeNodesPointer, (num_nodes+1) * sizeof(uint)));
	//gpuErrorcheck(cudaMallocHost(&activeEdgeList, num_edges * sizeof(E)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodes, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesPointer, (num_nodes+1) * sizeof(unsigned int)));
	//gpuErrorcheck(cudaMalloc(&d_activeEdgeList, (max_partition_size) * sizeof(E)));
}

template class Subgraph<OutEdge>;
template class Subgraph<OutEdgeWeighted>;

// For initialization with one active node
//unsigned int numActiveNodes = 1;
//subgraph.activeNodes[0] = SOURCE_NODE;
//for(unsigned int i=graph.nodePointer[SOURCE_NODE], j=0; i<graph.nodePointer[SOURCE_NODE] + graph.outDegree[SOURCE_NODE]; i++, j++)
//	subgraph.activeEdgeList[j] = graph.edgeList[i];
//subgraph.activeNodesPointer[0] = 0;
//subgraph.activeNodesPointer[1] = graph.outDegree[SOURCE_NODE];
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodes, subgraph.activeNodes, numActiveNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer, subgraph.activeNodesPointer, (numActiveNodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));


