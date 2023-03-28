/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */


#include <thrust/fill.h>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>
#include <../Subway/shared/graph.cuh>
#include <../Subway/shared/subgraph.cuh>
#include <../Subway/shared/partitioner.cuh>
#include <../Subway/shared/subgraph_generator.cuh>
#include <../Subway/shared/gpu_error_check.cuh>
#include <../Subway/shared/gpu_kernels.cuh>
#include <../Subway/shared/globals.hpp>
#include <../Subway/shared/timer.hpp>
#include <../Subway/shared/argument_parsing.cuh>
#include <../Subway/shared/subway_utilities.hpp>
#include <../Subway/shared/globals.hpp>
#include <../Subway/shared/timer.hpp>
#include "../Subway/shared/gpu_error_check.cuh"

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>

__device__ int* globalSubVertex;

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t const* offsets,
                   vertex_t const* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = depth;
  std::vector<vertex_t> cur_frontier_rows{source};
  std::vector<vertex_t> new_frontier_rows{};

  while (cur_frontier_rows.size() > 0) {
    for (auto const row : cur_frontier_rows) {
      auto nbr_offset_first = *(offsets + row);
      auto nbr_offset_last  = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr = *(indices + nbr_offset);
        if (*(distances + nbr) == std::numeric_limits<vertex_t>::max()) {
          *(distances + nbr)    = depth + 1;
          *(predecessors + nbr) = row;
          new_frontier_rows.push_back(nbr);
        }
      }
    }
    std::swap(cur_frontier_rows, new_frontier_rows);
    new_frontier_rows.clear();
    ++depth;
    if (depth >= depth_limit) { break; }
  }

  return;
}

/*struct BFS_Usecase {
  size_t source{1};
  bool check_correctness{true};
};*/

template <typename input_usecase_t>
class Tests_BFS : public ::testing::TestWithParam<std::tuple<cugraph::test::BFS_Usecase, input_usecase_t>> {
 public:
  Tests_BFS() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(cugraph::test::BFS_Usecase const& bfs_usecase, input_usecase_t const& input_usecase, bool sub)
  {
    rmm::mr::managed_memory_resource managed_mr;
    rmm::mr::set_current_device_resource(&managed_mr);
   
    //ofstream fout;
    //std::string mid_name = (sub ? "subway_" : "UVM_");
    //std::string file_name = "BFS_" + mid_name + ".txt";
    //fout.open(file_name, ios::app);
    //Timer timer;
    constexpr bool renumber = false;

    using weight_t = float;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    hr_timer.start("ConstructGraph");
    //timer.Start();

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber, true, true);

    hr_timer.stop();
    //float readtime = timer.Finish();
	  //cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
    hr_timer.start("Preprocessing");
    //free(graph);
    auto graph_view = graph.view();
    auto graph_view_copy = graph.view();
    /*rmm::mr::cuda_memory_resource cuda_resource;
      rmm::mr::set_current_device_resource(&cuda_resource);*/
     rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());

    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                  handle.get_stream());
    if(sub){
      cout<<"subway starts\n";
      
      //rmm::mr::cuda_memory_resource cuda_resource;
      //rmm::mr::set_current_device_resource(&cuda_resource);

      auto h_offset_ = cugraph::test::to_host(handle, graph_view.local_edge_partition_view().offsets());
      auto h_indices_ = cugraph::test::to_host(handle, graph_view.local_edge_partition_view().indices());

       std::vector<uint32_t> host_visited_flags((graph_view.local_vertex_partition_range_size() + (sizeof(uint32_t) * 8 - 1)) /
      (sizeof(uint32_t) * 8), 0);

      graph_view.offsets_ = NULL;
      graph_view.indices_ = NULL;
      cudaFree((void*)graph_view.offsets_);
      cudaFree((void*)graph_view.indices_);

      

      Graph<OutEdge> bfs_graph("", false);
      bfs_graph.nodePointer = new unsigned int[h_offset_.size()];
      cudaMallocHost(&bfs_graph.edgeList, (h_indices_.size()) * sizeof(unsigned int));
      for(int i=0;i<h_offset_.size();i++){
        bfs_graph.nodePointer[i] = h_offset_[i];
       // cout<<h_offset_[i]<<" ";
      }
     // cout<<"\n";
      for(int i=0;i<h_indices_.size();i++){
        bfs_graph.edgeList[i].end = h_indices_[i];
      }
      bfs_graph.outDegree = new unsigned int[h_offset_.size()];
      for(int i=1;i<h_offset_.size();i++){
        bfs_graph.outDegree[i-1] = h_offset_[i] - h_offset_[i-1];
      }

      bfs_graph.num_nodes = h_offset_.size();
      bfs_graph.num_edges = h_indices_.size();
      bfs_graph.label1 = new bool[bfs_graph.num_nodes];
      bfs_graph.label2 = new bool[bfs_graph.num_nodes];
      for(unsigned int i=0; i<bfs_graph.num_nodes; i++)
      {
        bfs_graph.label1[i] = true;
        bfs_graph.label2[i] = false;
      }

      cudaMalloc(&bfs_graph.d_outDegree, bfs_graph.num_nodes * sizeof(unsigned int));
      cudaMalloc(&bfs_graph.d_label1, bfs_graph.num_nodes * sizeof(bool));
      cudaMalloc(&bfs_graph.d_label2, bfs_graph.num_nodes * sizeof(bool));

      cudaMemcpy(bfs_graph.d_outDegree, bfs_graph.outDegree, bfs_graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(bfs_graph.d_label1, bfs_graph.label1, bfs_graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice);
      cudaMemcpy(bfs_graph.d_label2, bfs_graph.label2, bfs_graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice);

    Subgraph<OutEdge> subgraph(bfs_graph.num_nodes, bfs_graph.num_edges);
    
    SubgraphGenerator<OutEdge> subgen(bfs_graph);
    
    subgen.generate(bfs_graph, subgraph);
    
    for(unsigned int i=0; i<bfs_graph.num_nodes; i++)
    {
      bfs_graph.label1[i] = false;
    }
    bfs_graph.label1[bfs_usecase.source] = true;

    gpuErrorcheck(cudaMemcpy(bfs_graph.d_label1, bfs_graph.label1, bfs_graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
    Partitioner<OutEdge> partitioner;
    
    unsigned int gItr = 0;
    
    bool finished;
    bool *d_finished;
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    
   
    std::vector<int32_t> vis;

    auto constexpr invalid_distance = std::numeric_limits<vertex_t>::max();
    auto constexpr invalid_vertex   = cugraph::invalid_vertex_id<vertex_t>::value;

    std::vector<vertex_t> host_distance(graph_view.number_of_vertices(), invalid_distance);
    host_distance[bfs_usecase.source] = 0;
    d_distances = cugraph::test::to_device(handle, host_distance);

    std::vector<vertex_t> host_predecessors(graph_view.number_of_vertices(), invalid_vertex);
    d_predecessors = cugraph::test::to_device(handle, host_predecessors);



    auto visited_flags = cugraph::test::to_device(handle, host_visited_flags);

    hr_timer.stop();

    
    bool *host_label1 = new bool[bfs_graph.num_nodes];
    
    hr_timer.start("BFSSubway");
    int* host_sub_vertex = new int[bfs_graph.num_nodes];
    memset(host_sub_vertex, -1, sizeof(host_sub_vertex));
    while (subgraph.numActiveNodes>0)
    {
      gItr++;
      
      partitioner.partition(subgraph, subgraph.numActiveNodes);
      vertex_t *host_ptr_egdeList = new vertex_t[subgraph.num_edges+1];
      for(int i=0;i<subgraph.num_edges;i++){
        host_ptr_egdeList[i] = subgraph.activeEdgeList[i].end;
      }
      std::cout<<"Number of partitions: "<<partitioner.numPartitions<<"\n";
      for(int i=0; i<partitioner.numPartitions; i++)
      { 
        cout<<partitioner.fromNode[i]<<" "<<partitioner.partitionNodeSize[i]<<"\n";
        cout<<partitioner.fromEdge[i]<<" "<<partitioner.partitionEdgeSize[i]<<"\n\n";
        
        rmm::device_uvector<int> device_mapSubvertex(bfs_graph.num_nodes, handle.get_stream());
        for(int j=partitioner.fromNode[i];j<(partitioner.fromNode[i]+partitioner.partitionNodeSize[i]);j++){
          host_sub_vertex[subgraph.activeNodes[j]] = j-partitioner.fromNode[i];
        }
        cudaMemcpy(device_mapSubvertex.data(), host_sub_vertex, bfs_graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        subgen.callKernel(subgraph, bfs_graph, partitioner, i);
        uint itr = 0;

        itr++;
        hr_timer.start("rapidsToSubway");
        
        gpuErrorcheck(cudaFree((void *)graph_view.offsets_));
        gpuErrorcheck(cudaFree((void *)graph_view.indices_));
        gpuErrorcheck((cudaMalloc(&graph_view.offsets_, (partitioner.partitionNodeSize[i])*sizeof(edge_t))));
        gpuErrorcheck((cudaMalloc(&graph_view.indices_, partitioner.partitionEdgeSize[i]*sizeof(vertex_t))));
        gpuErrorcheck(cudaMemcpy((void*)graph_view.offsets_, (void*)(subgraph.activeNodesPointer + partitioner.fromNode[i]), (partitioner.partitionNodeSize[i]) * sizeof(edge_t), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy((void*)( graph_view.indices_), host_ptr_egdeList + partitioner.fromEdge[i], partitioner.partitionEdgeSize[i] * sizeof(vertex_t), cudaMemcpyHostToDevice));
        
        uint* extra_ele;
        
        /*if(i < partitioner.numPartitions-1){
          extra_ele = new uint[1];
          int ele = partitioner.fromNode[i] + partitioner.partitionNodeSize[i] - 1;
          extra_ele[0] = subgraph.activeNodesPointer[ele] + bfs_graph.outDegree[subgraph.activeNodes[ele]];
          gpuErrorcheck(cudaMemcpy((void*)( graph_view.offsets_ + partitioner.partitionNodeSize[i]), (void*)extra_ele, sizeof(vertex_t), cudaMemcpyHostToDevice));
        }*/

        hr_timer.stop();
    
        cudaMemcpy((void*)(host_label1), bfs_graph.d_label1, bfs_graph.num_nodes*sizeof(bool), cudaMemcpyDeviceToHost);
        
        for(int j=0;j<bfs_graph.num_nodes;j++){
          if(host_label1[j])
            vis.push_back(j);
        }
        cout<<"Activate ****************** nodes "<<vis.size()<<" "<<vis[0]<<"\n";
        if(!vis.size()) break;

        auto d_source = cugraph::test::to_device(handle, vis);
        graph_view.set_number_of_vertices((size_t)partitioner.partitionNodeSize[i]);
        graph_view.set_number_of_edges((size_t)partitioner.partitionEdgeSize[i]);
        int tot = bfs_graph.num_nodes;
        cugraph::bfs_subway(handle,
                  graph_view,
                  d_distances.data(),
                  d_predecessors.data(),
                  (vertex_t*)d_source.data(),
                  bfs_graph.d_label1,
                  bfs_graph.d_label2,
                  visited_flags.data(), // change
                  device_mapSubvertex.data(),
                  tot,
                  visited_flags.size(),
                  vis.size(),
                  false,
                  std::numeric_limits<vertex_t>::max()
                  );
        cudaDeviceSynchronize();
        gpuErrorcheck( cudaPeekAtLastError() );
        memset(host_sub_vertex, -1, sizeof(host_sub_vertex));
        vis.clear();
        d_source.release();
        device_mapSubvertex.release();
        
      }
      subgen.generate(bfs_graph, subgraph);

    }

      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);

    }
    else{
    hr_timer.stop();
    hr_timer.start("BFSUVM");

    rmm::device_scalar<vertex_t> const d_source(bfs_usecase.source, handle.get_stream());

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.data(),
                 d_predecessors.data(),
                 d_source.data(),
                 size_t{1},
                 false,
                 std::numeric_limits<vertex_t>::max());
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    hr_timer.stop();
    hr_timer.display_and_clear(std::cout);

    }  

    //fout.close();
    
    /*ASSERT_TRUE(static_cast<vertex_t>(bfs_usecase.source) >= 0 &&
                static_cast<vertex_t>(bfs_usecase.source) < graph_view.number_of_vertices())
      << "Invalid starting source.";*/

    if (bfs_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view_copy;

      auto h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      auto h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      auto unrenumbered_source = static_cast<vertex_t>(bfs_usecase.source);
      if (renumber) {
        auto h_renumber_map_labels = cugraph::test::to_host(handle, *d_renumber_map_labels);
        unrenumbered_source        = h_renumber_map_labels[bfs_usecase.source];
      }

      std::vector<vertex_t> h_reference_distances(unrenumbered_graph_view.number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(unrenumbered_graph_view.number_of_vertices());

      bfs_reference(h_offsets.data(),
                    h_indices.data(),
                    h_reference_distances.data(),
                    h_reference_predecessors.data(),
                    unrenumbered_graph_view.number_of_vertices(),
                    unrenumbered_source,
                    std::numeric_limits<vertex_t>::max());

      std::vector<vertex_t> h_cugraph_distances{};
      std::vector<vertex_t> h_cugraph_predecessors{};
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_predecessors.data(),
                                               d_predecessors.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices(),
                                               true);

        rmm::device_uvector<vertex_t> d_unrenumbered_distances(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_distances) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_distances);
        rmm::device_uvector<vertex_t> d_unrenumbered_predecessors(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_predecessors) =
          cugraph::test::sort_by_key(handle, *d_renumber_map_labels, d_predecessors);
        h_cugraph_distances    = cugraph::test::to_host(handle, d_unrenumbered_distances);
        h_cugraph_predecessors = cugraph::test::to_host(handle, d_unrenumbered_predecessors);
      } else {
        h_cugraph_distances    = cugraph::test::to_host(handle, d_distances);
        h_cugraph_predecessors = cugraph::test::to_host(handle, d_predecessors);
      }

      ASSERT_TRUE(std::equal(
        h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
        << "distances do not match with the reference values.";
      int cnt = 0;
       for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
        auto i = std::distance(h_cugraph_predecessors.begin(), it);
        if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
          cnt++;
          ASSERT_TRUE(h_reference_predecessors[i] == *it)
            << "vertex reachability does not match with the reference.";
            if(cnt==100){
              break;
            }
        } else {
          ASSERT_TRUE(h_reference_distances[*it] + 1 == h_reference_distances[i])
            << "distance to this vertex != distance to the predecessor vertex + 1.";
          bool found{false};
          for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
            if (h_indices[j] == i) {
              found = true;
              break;
            }
          }
          ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
        }
      } 
    }
  }
};

using Tests_BFS_File = Tests_BFS<cugraph::test::File_Usecase>;
using Tests_BFS_Rmat = Tests_BFS<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_BFS_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    override_Source_Node_with_cmd_line_arguments(std::get<0>(param)),
   override_File_Usecase_with_cmd_line_arguments(std::get<1>(param)),
   cugraph::test::g_subway);
}

/*TEST_P(Tests_BFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_BFS_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_BFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}*/

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_BFS_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(cugraph::test::BFS_Usecase{},
                    cugraph::test::File_Usecase("/home/ankit/rapids/cugraph/cugraph/datasets/test_pre.csv"))));
                    //cugraph::test::File_Usecase("/home/ankit/rapids/cugraph/cugraph/datasets/test_pre.csv"))));

/*INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_BFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(BFS_Usecase{0},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, 
  Tests_BFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_pair(BFS_Usecase{0, false},
                   cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));
*/
CUGRAPH_TEST_PROGRAM_MAIN()
