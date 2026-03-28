/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <cstddef>
#include <cstdint>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace layer {

struct PinnedHostMemoryWeightSlice {
  uint64_t offset = 0;
  uint64_t bytes = 0;
  std::vector<int64_t> sizes;
  torch::ScalarType dtype = torch::kFloat16;
};

struct PinnedHostMemoryEntry {
  std::string cache_key;
  void* host_pinned_storage = nullptr;
  uint64_t storage_size = 0;
  std::vector<PinnedHostMemoryWeightSlice> weight_slices;
  bool loading = false;
  bool ready = false;
  size_t ref_count = 0;
  std::mutex mutex;
  std::condition_variable cv;
};

class PinnedHostMemoryCache {
 public:
  static PinnedHostMemoryCache& get_instance() {
    static PinnedHostMemoryCache instance;
    return instance;
  }

  std::shared_ptr<PinnedHostMemoryEntry> acquire_or_create(
      const std::string& cache_key,
      bool* cache_hit);

  void* allocate_host_storage(const std::shared_ptr<PinnedHostMemoryEntry>& entry,
                              uint64_t storage_size);

  void publish(const std::shared_ptr<PinnedHostMemoryEntry>& entry,
               uint64_t storage_size,
               const std::vector<PinnedHostMemoryWeightSlice>& weight_slices);

  void release(const std::shared_ptr<PinnedHostMemoryEntry>& entry);

 private:
  PinnedHostMemoryCache() = default;

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<PinnedHostMemoryEntry>>
      entries_;
};

}  // namespace layer
}  // namespace xllm
