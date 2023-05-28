
#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>


template <class E>
Subgraph<E>::Subgraph(uint num_nodes, uint num_edges)
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

	
	/*size_t mem_free, mem_avail;

    cudaMemGetInfo(&mem_free, &mem_avail);

    cout << "\n Available memory " << mem_avail << " \n";

	cout << "\n Free memory " << mem_free << " \n";

     max_partition_size = 0.5 * ( mem_free  - (6*(1UL<<30))) / sizeof(E);*/

	//max_partition_size = 1e8;

	unsigned long long percentage = 70;
	unsigned long long reserve_memory =(unsigned long long)((24*(1ULL<<30))/100)*percentage;
	reserve_memory += (unsigned long long)(6*(1UL<<30));  
	//max_partition_size = 0.5 * ((dev.totalGlobalMem-reserve_memory) - 8*4*num_nodes) / sizeof(E);
	max_partition_size = 1e8;
    cout << "\n Max Partition size : " << max_partition_size << "\n";
		
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


