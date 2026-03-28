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

#include "pinned_host_memory_cache.h"

#include <acl/acl_rt.h>
#include <glog/logging.h>

namespace xllm {
namespace layer {

std::shared_ptr<PinnedHostMemoryEntry> PinnedHostMemoryCache::acquire_or_create(
    const std::string& cache_key,
    bool* cache_hit) {
  CHECK(cache_hit != nullptr) << "cache_hit must not be null";

  std::shared_ptr<PinnedHostMemoryEntry> entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(cache_key);
    if (it == entries_.end()) {
      entry = std::make_shared<PinnedHostMemoryEntry>();
      entry->cache_key = cache_key;
      entry->loading = true;
      entry->ref_count = 1;
      entries_.emplace(cache_key, entry);
      *cache_hit = false;
      return entry;
    }

    entry = it->second;
    ++entry->ref_count;
  }

  std::unique_lock<std::mutex> entry_lock(entry->mutex);
  entry->cv.wait(entry_lock, [&entry]() { return !entry->loading; });
  CHECK(entry->ready) << "Pinned host cache entry is not ready for key="
                      << cache_key;
  *cache_hit = true;
  return entry;
}

void* PinnedHostMemoryCache::allocate_host_storage(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry,
    uint64_t storage_size) {
  CHECK(entry != nullptr) << "Pinned host cache entry is null";
  CHECK_GT(storage_size, 0) << "Pinned host cache storage_size must be > 0";

  std::lock_guard<std::mutex> entry_lock(entry->mutex);
  if (entry->host_pinned_storage == nullptr) {
    auto ret = aclrtMallocHost(&entry->host_pinned_storage, storage_size);
    CHECK_EQ(ret, ACL_SUCCESS)
        << "Failed to allocate cached pinned host storage size="
        << storage_size;
    entry->storage_size = storage_size;
  } else {
    CHECK_EQ(entry->storage_size, storage_size)
        << "Pinned host cache storage_size mismatch, key=" << entry->cache_key
        << ", cached=" << entry->storage_size
        << ", requested=" << storage_size;
  }
  return entry->host_pinned_storage;
}

void PinnedHostMemoryCache::publish(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry,
    uint64_t storage_size,
    const std::vector<PinnedHostMemoryWeightSlice>& weight_slices) {
  CHECK(entry != nullptr) << "Pinned host cache entry is null";

  std::lock_guard<std::mutex> entry_lock(entry->mutex);
  CHECK(entry->host_pinned_storage != nullptr)
      << "Pinned host cache entry has no host storage for key="
      << entry->cache_key;
  entry->storage_size = storage_size;
  entry->weight_slices = weight_slices;
  entry->ready = true;
  entry->loading = false;
  entry->cv.notify_all();
}

void PinnedHostMemoryCache::release(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry) {
  if (entry == nullptr) {
    return;
  }

  void* host_pinned_storage = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(entry->cache_key);
    if (it == entries_.end()) {
      return;
    }

    auto cached_entry = it->second;
    CHECK_EQ(cached_entry.get(), entry.get())
        << "Pinned host cache entry mismatch for key=" << entry->cache_key;
    CHECK_GT(cached_entry->ref_count, 0)
        << "Pinned host cache ref_count is already zero for key="
        << entry->cache_key;
    --cached_entry->ref_count;
    if (cached_entry->ref_count == 0 && !cached_entry->loading) {
      {
        std::lock_guard<std::mutex> entry_lock(cached_entry->mutex);
        host_pinned_storage = cached_entry->host_pinned_storage;
        cached_entry->host_pinned_storage = nullptr;
        cached_entry->storage_size = 0;
        cached_entry->weight_slices.clear();
        cached_entry->ready = false;
      }
      entries_.erase(it);
    }
  }

  if (host_pinned_storage != nullptr) {
    auto ret = aclrtFreeHost(host_pinned_storage);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Failed to free cached pinned host storage, ret=" << ret;
    }
  }
}

}  // namespace layer
}  // namespace xllm
