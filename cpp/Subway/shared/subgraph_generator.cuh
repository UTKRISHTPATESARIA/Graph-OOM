#ifndef SUBGRAPH_GENERATOR_HPP
#define SUBGRAPH_GENERATOR_HPP


#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"
#include "partitioner.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>
// #include <raft/core/handle.hpp>

template <class E>
class SubgraphGenerator
{
private:

public:
	unsigned int *activeNodesLabeling;
	unsigned int *activeNodesDegree;
	unsigned int *prefixLabeling;
	unsigned int *prefixSumDegrees;
	unsigned int *d_activeNodesLabeling;
	unsigned int *d_activeNodesDegree;
	unsigned int *d_prefixLabeling;
	unsigned int *d_prefixSumDegrees;
	SubgraphGenerator(Graph<E> &graph);
	SubgraphGenerator(GraphPR<E> &graph);
	void generate(Graph<E> &graph, Subgraph<E> &subgraph);
	void generate(GraphPR<E> &graph, Subgraph<E> &subgraph, float acc);
	void generate(Graph<E> &graph, Subgraph<E> &subgraph, int *edgelist);
	void callKernel(Subgraph<OutEdge> &subgraph, Graph<OutEdge> &graph, Partitioner<OutEdge> &partitioner, 
					int i);
	void populate_visited(Subgraph<OutEdge> &subgraph, Graph<OutEdge> &graph, Partitioner<OutEdge> &partitioner,
										int i, int* source, unsigned long int *num_ele);

	void populate_subVertex1(Subgraph<OutEdge> &subgraph, Partitioner<OutEdge> &partitioner,
										int i, int* subVertex, unsigned int val);
	void populate_subVertex2(Subgraph<OutEdge> &subgraph, Partitioner<OutEdge> &partitioner,
										int i, int* subVertex, int val);
	//void copyEdgeList(Subgraph<OutEdge> &subgraph, int *hostList, int n, raft::handle_t const& handle, int*num);
};

#endif	//	SUBGRAPH_GENERATOR_HPP



