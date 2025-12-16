/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xtensor_allocator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/global_flags.h"
#include "common/macros.h"
#include "distributed_runtime/collective_service.h"
#include "phy_page.h"
#include "phy_page_pool.h"
#include "platform/device.h"
#include "platform/vmm_api.h"
#include "server/xllm_server_registry.h"
#include "xtensor.h"

namespace xllm {

static inline size_t get_v_base_offset(const torch::Tensor& tensor) {
  size_t num_eles = tensor.numel() * tensor.element_size();
  size_t page_size = FLAGS_phy_page_granularity_size;
  CHECK(num_eles % (2 * page_size) == 0)
      << "Invalid tensor size: " << num_eles
      << ", must be a multiple of 2 * page size " << 2 * page_size;
  return num_eles / 2;
}

XTensorAllocator::~XTensorAllocator() {
  if (!initialized_) {
    return;
  }

  // Stop collective server if running
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }

  destroy();
}

void XTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  k_tensors_.clear();
  v_tensors_.clear();
  weight_tensor_.reset();
  zero_page_ = nullptr;  // Not owned, just clear pointer
  xtensor_dist_clients_.clear();
  xtensor_dist_servers_.clear();
  initialized_ = false;
}

void XTensorAllocator::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "XTensorAllocator already initialized, ignoring re-init";
    return;
  }

  dev_ = device;
  init_device_();
  initialized_ = true;
}

void XTensorAllocator::setup_multi_node_xtensor_dist(
    const xtensor::Options& options,
    const std::string& master_node_addr,
    int32_t dp_size) {
  const auto& devices = options.devices();
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  world_size_ = each_node_ranks * FLAGS_nnodes;
  dp_size_ = dp_size;
  tp_size_ = world_size_ / dp_size_;

  CHECK_EQ(world_size_ % dp_size_, 0)
      << "world_size must be divisible by dp_size";

  std::vector<std::atomic<bool>> dones(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    dones[i].store(false, std::memory_order_relaxed);
  }

  // Update collective server name with server index
  collective_server_name_ = "XTensorDistCollectiveServer";

  for (size_t i = 0; i < devices.size(); ++i) {
    // Create XTensor dist server for each device
    xtensor_dist_servers_.emplace_back(std::make_unique<XTensorDistServer>(
        i, master_node_addr, dones[i], devices[i], options));

    // Only rank0 connects to other workers
    if (FLAGS_node_rank == 0) {
      std::shared_ptr<CollectiveService> collective_service =
          std::make_shared<CollectiveService>(
              0, world_size_, devices[0].index());
      XllmServer* collective_server =
          ServerRegistry::get_instance().register_server(
              collective_server_name_);
      if (!collective_server->start(collective_service, master_node_addr)) {
        LOG(ERROR) << "failed to start collective server on address: "
                   << master_node_addr;
        return;
      }

      auto xtensor_dist_addrs_map = collective_service->wait();

      // Initialize DP group clients mapping
      dp_group_clients_.resize(dp_size_);
      for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
        dp_group_clients_[dp_rank].reserve(tp_size_);
      }

      for (int32_t r = 0; r < world_size_; ++r) {
        if (xtensor_dist_addrs_map.find(r) == xtensor_dist_addrs_map.end()) {
          LOG(FATAL) << "Not all xtensor dist servers connect to master node. "
                        "Miss rank is "
                     << r;
          return;
        }
        auto client = std::make_shared<XTensorDistClient>(
            r, xtensor_dist_addrs_map[r], devices[r % each_node_ranks]);

        // Add to flat list
        xtensor_dist_clients_.emplace_back(client);

        // Add to DP group mapping
        // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1,
        // ...]
        int32_t dp_rank = r / tp_size_;
        dp_group_clients_[dp_rank].emplace_back(client);
      }

      LOG(INFO) << "XTensor dist setup: world_size=" << world_size_
                << ", dp_size=" << dp_size_ << ", tp_size=" << tp_size_;
    }

    // Wait for all servers to be ready
    for (size_t idx = 0; idx < dones.size(); ++idx) {
      while (!dones[idx].load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }
}

int64_t XTensorAllocator::init_phy_page_pools(double max_memory_utilization,
                                              int64_t max_cache_size) {
  if (world_size_ <= 1) {
    // Single process single GPU, initialize locally
    Device device(dev_);
    device.set_device();

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    int64_t cache_size = available_memory;
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      cache_size -= buffer_memory;
    }
    if (max_cache_size > 0) {
      cache_size = std::min(cache_size, max_cache_size);
    }

    int64_t num_pages = cache_size / FLAGS_phy_page_granularity_size;
    LOG(INFO) << "init_phy_page_pools (local): available_memory="
              << available_memory << ", total_memory=" << total_memory
              << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

    PhyPagePool::get_instance().init(dev_, num_pages);
    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    memory_futures.push_back(client->get_memory_info_async());
  }

  // Wait for all memory info responses
  auto memory_results = folly::collectAll(memory_futures).get();

  int64_t min_available_memory = std::numeric_limits<int64_t>::max();
  int64_t min_total_memory = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < memory_results.size(); ++i) {
    if (!memory_results[i].hasValue()) {
      LOG(ERROR) << "Failed to get memory info from worker: " << i;
      return 0;
    }
    auto& info = memory_results[i].value();
    if (info.available_memory == 0 && info.total_memory == 0) {
      LOG(ERROR) << "Worker " << i << " returned invalid memory info";
      return 0;
    }

    LOG(INFO) << "Worker #" << i
              << ": available_memory=" << info.available_memory
              << ", total_memory=" << info.total_memory;

    min_available_memory =
        std::min(min_available_memory, info.available_memory);
    min_total_memory = std::min(min_total_memory, info.total_memory);
  }

  // Step 2: Calculate num_pages based on min available memory
  int64_t cache_size = min_available_memory;
  if (max_memory_utilization < 1.0) {
    const int64_t buffer_memory =
        min_total_memory * (1.0 - max_memory_utilization);
    cache_size -= buffer_memory;
  }
  if (max_cache_size > 0) {
    cache_size = std::min(cache_size, max_cache_size);
  }

  int64_t num_pages = cache_size / FLAGS_phy_page_granularity_size;
  LOG(INFO) << "init_phy_page_pools: min_available_memory="
            << min_available_memory << ", min_total_memory=" << min_total_memory
            << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

  if (num_pages <= 0) {
    LOG(ERROR) << "Insufficient memory for PhyPagePool";
    return 0;
  }

  // Step 3: Broadcast InitPhyPagePool to all workers
  std::vector<folly::SemiFuture<bool>> init_futures;
  init_futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    init_futures.push_back(client->init_phy_page_pool_async(num_pages));
  }

  // Wait for all init responses
  auto init_results = folly::collectAll(init_futures).get();
  for (size_t i = 0; i < init_results.size(); ++i) {
    if (!init_results[i].hasValue() || !init_results[i].value()) {
      LOG(ERROR) << "Failed to init PhyPagePool on worker: " << i;
      return 0;
    }
  }

  LOG(INFO) << "Successfully initialized PhyPagePool on all " << world_size_
            << " workers with " << num_pages << " pages each";
  return num_pages;
}

bool XTensorAllocator::broadcast_map_to_kv_tensors(
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just map locally
    return map_to_kv_tensors(offsets);
  }

  // Get clients for the specified DP group
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  const auto& clients = dp_group_clients_[dp_rank];

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(clients.size());
  for (auto& client : clients) {
    futures.push_back(client->map_to_kv_tensors_async(offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool XTensorAllocator::broadcast_unmap_from_kv_tensors(
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just unmap locally
    return unmap_from_kv_tensors(offsets);
  }

  // Get clients for the specified DP group
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  const auto& clients = dp_group_clients_[dp_rank];

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(clients.size());
  for (auto& client : clients) {
    futures.push_back(client->unmap_from_kv_tensors_async(offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool XTensorAllocator::broadcast_map_weight_tensor(int64_t num_pages) {
  if (world_size_ <= 1) {
    // Single process single GPU, just map locally
    return map_weight_tensor(num_pages);
  }

  // Broadcast to all workers via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    futures.push_back(client->map_weight_tensor_async(num_pages));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool XTensorAllocator::broadcast_unmap_weight_tensor() {
  if (world_size_ <= 1) {
    // Single process single GPU, just unmap locally
    return unmap_weight_tensor();
  }

  // Broadcast to all workers via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    futures.push_back(client->unmap_weight_tensor_async());
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

std::vector<torch::Tensor> XTensorAllocator::create_k_tensors(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(dims, dtype, num_layers, k_tensors_, "K");
}

std::vector<torch::Tensor> XTensorAllocator::create_v_tensors(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(dims, dtype, num_layers, v_tensors_, "V");
}

std::vector<torch::Tensor> XTensorAllocator::create_kv_tensors_impl_(
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    std::vector<std::unique_ptr<XTensor>>& tensors_out,
    const char* name) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(num_layers_ == 0 || num_layers_ == num_layers)
      << "Number of layers mismatch";
  CHECK(tensors_out.empty()) << name << " tensors already created";
  CHECK(!dims.empty()) << name << " tensor dims cannot be empty";

  // Calculate size from dims and dtype
  size_t size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  for (auto dim : dims) {
    size *= dim;
  }

  size_t page_size = FLAGS_phy_page_granularity_size;
  // Align size to page size (round up)
  if (size % page_size != 0) {
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    LOG(WARNING) << name << " tensor size " << size
                 << " is not aligned to page size " << page_size
                 << ", aligning to " << aligned_size;
    size = aligned_size;
  }

  num_layers_ = num_layers;
  kv_tensor_size_per_layer_ = size;

  if (!zero_page_) {
    zero_page_ = PhyPagePool::get_instance().get_zero_page();
  }

  return create_tensors_internal_(size, dims, dtype, num_layers, tensors_out);
}

bool XTensorAllocator::map_to_kv_tensors(const std::vector<offset_t>& offsets) {
  std::unique_lock<std::mutex> lock(mtx_);

  if (k_tensors_.empty() || v_tensors_.empty()) {
    LOG(ERROR) << "KV tensors not created";
    return false;
  }

  // Per-layer mapping for K and V tensors separately
  for (int64_t i = 0; i < num_layers_; i++) {
    auto k_xtensor = k_tensors_[i].get();
    auto v_xtensor = v_tensors_[i].get();
    for (auto offset : offsets) {
      k_xtensor->map(offset);
      v_xtensor->map(offset);
    }
  }
  return true;
}

bool XTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t>& offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (k_tensors_.empty() || v_tensors_.empty()) {
    LOG(ERROR)
        << "try to unmap from KV tensors when KV tensors are not created";
    return false;
  }

  // Per-layer unmapping for K and V tensors separately
  for (int64_t i = 0; i < num_layers_; i++) {
    auto k_xtensor = k_tensors_[i].get();
    auto v_xtensor = v_tensors_[i].get();
    for (auto offset : offsets) {
      k_xtensor->unmap(offset);
      v_xtensor->unmap(offset);
    }
  }
  return true;
}

bool XTensorAllocator::map_weight_tensor(int64_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Create weight tensor if not exists
  if (!weight_tensor_) {
    size_t page_size = FLAGS_phy_page_granularity_size;
    size_t size = num_pages * page_size;

    // Get zero page from pool if not exists
    if (!zero_page_) {
      zero_page_ = PhyPagePool::get_instance().get_zero_page();
    }

    weight_tensor_ =
        std::make_unique<XTensor>(size, torch::kByte, dev_, zero_page_);
    LOG(INFO) << "Created weight XTensor: num_pages=" << num_pages
              << ", page_size=" << page_size << ", total_size=" << size;
  }

  return weight_tensor_->map_all();
}

bool XTensorAllocator::unmap_weight_tensor() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!weight_tensor_) {
    LOG(ERROR) << "Weight tensor not created";
    return false;
  }
  return weight_tensor_->unmap_all();
}

bool XTensorAllocator::allocate_weight(void*& ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!weight_tensor_) {
    LOG(ERROR) << "Weight tensor not created, call map_weight_tensor first";
    return false;
  }
  return weight_tensor_->allocate(ptr, size);
}

std::vector<torch::Tensor> XTensorAllocator::create_tensors_internal_(
    size_t size,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    std::vector<std::unique_ptr<XTensor>>& tensors_out) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(num_layers);
  tensors_out.reserve(num_layers);

  for (int64_t i = 0; i < num_layers; i++) {
    auto xtensor = std::make_unique<XTensor>(size, dtype, dev_, zero_page_);
    tensors.push_back(xtensor->to_torch_tensor(0, dims));
    tensors_out.push_back(std::move(xtensor));
  }
  return tensors;
}

void XTensorAllocator::init_device_() {
  Device device(dev_);
  device.set_device();
  device.init_device_context();

  // Create a dummy PhyPage to initialize the granularity size
  // This will set FLAGS_phy_page_granularity_size
  auto dummy_page = std::make_shared<PhyPage>(dev_);

  size_t chunk_sz = FLAGS_phy_page_granularity_size;
  LOG(INFO) << "Device initialized with granularity size: " << chunk_sz
            << " bytes";
}

}  // namespace xllm
