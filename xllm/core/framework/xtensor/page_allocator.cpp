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

#include "page_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>

#include "common/global_flags.h"
#include "xtensor_allocator.h"

namespace xllm {

void PageAllocator::init(int64_t num_layers,
                         size_t num_phy_pages,
                         int32_t dp_size,
                         bool enable_page_prealloc) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PageAllocator already initialized, ignoring re-init";
    return;
  }

  num_layers_ = num_layers;
  dp_size_ = dp_size;
  page_size_ = FLAGS_phy_page_granularity_size;
  enable_page_prealloc_ = enable_page_prealloc;

  // Set total physical pages from parameter
  num_total_phy_pages_ = num_phy_pages;
  num_free_phy_pages_ = num_total_phy_pages_;

  // Calculate virtual pages based on single-layer memory size
  // Virtual pages are per-DP group, so divide by dp_size
  num_total_virt_pages_ = num_total_phy_pages_ / num_layers_;

  // Calculate physical pages consumed per virtual page allocation
  // Each virt_page needs to map on all K and V XTensors
  // page_size is the same for both virt and phy pages
  // So each XTensor map consumes 1 phy_page
  // Total = 2 * num_layers (for K and V of each layer)
  phy_pages_per_virt_page_ = 2 * num_layers_;

  CHECK(phy_pages_per_virt_page_ > 0) << "phy_pages_per_virt_page must be > 0";

  // Initialize per-DP group page lists
  dp_group_pages_.resize(dp_size_);
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    auto& dp_pages = dp_group_pages_[dp_rank];
    dp_pages.num_free_virt_pages = num_total_virt_pages_;
    for (size_t i = 0; i < num_total_virt_pages_; ++i) {
      dp_pages.free_virt_page_list.push_back(static_cast<int64_t>(i));
    }
  }

  initialized_ = true;

  LOG(INFO) << "Init PageAllocator: "
            << "num_layers=" << num_layers << ", dp_size=" << dp_size_
            << ", page_size=" << page_size_ / (1024 * 1024) << "MB"
            << ", num_total_virt_pages=" << num_total_virt_pages_
            << " (per dp_group)"
            << ", num_total_phy_pages=" << num_total_phy_pages_
            << ", phy_pages_per_virt_page=" << phy_pages_per_virt_page_
            << ", enable_prealloc=" << enable_page_prealloc;
}

PageAllocator::~PageAllocator() {
  try {
    if (enable_page_prealloc_ && prealloc_thd_ != nullptr) {
      stop_prealloc_thread(PREALLOC_THREAD_TIMEOUT);
    }
  } catch (...) {
    // Silently ignore exceptions during cleanup
  }
}

void PageAllocator::start_prealloc_thread() {
  if (enable_page_prealloc_) {
    start_prealloc_thread_internal();
  }
}

bool PageAllocator::has_enough_phy_pages(size_t num_virt_pages) const {
  // Note: Caller must hold mtx_
  size_t needed_phy = num_virt_pages * phy_pages_per_virt_page_;
  return num_free_phy_pages_ >= needed_phy;
}

void PageAllocator::consume_phy_pages(size_t count) {
  // Note: Caller must hold mtx_
  size_t needed = count * phy_pages_per_virt_page_;
  CHECK(num_free_phy_pages_ >= needed)
      << "Not enough physical pages: need " << needed << ", available "
      << num_free_phy_pages_;
  num_free_phy_pages_ -= needed;
}

void PageAllocator::release_phy_pages(size_t count) {
  size_t released = count * phy_pages_per_virt_page_;
  num_free_phy_pages_ += released;
}

std::unique_ptr<VirtPage> PageAllocator::alloc_kv_cache_page(int32_t dp_rank) {
  std::unique_lock<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PageAllocator not initialized";
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  auto& dp_pages = dp_group_pages_[dp_rank];
  std::optional<int64_t> virt_page_id;

  while (!virt_page_id.has_value()) {
    // Fast path: allocate from reserved pages (already mapped)
    if (!dp_pages.reserved_virt_page_list.empty()) {
      virt_page_id = dp_pages.reserved_virt_page_list.front();
      dp_pages.reserved_virt_page_list.pop_front();
      dp_pages.num_free_virt_pages--;
      // Physical pages already consumed when reserved

      // Trigger preallocation to refill reserved pool if getting low
      if (dp_pages.reserved_virt_page_list.size() <
          static_cast<size_t>(min_reserved_pages_)) {
        prealloc_needed_ = true;
        cond_.notify_all();
      }

      update_memory_usage();
      return std::make_unique<VirtPage>(*virt_page_id, page_size_);
    }

    // Slow path: allocate from free pages (need to map)
    if (!dp_pages.free_virt_page_list.empty() && has_enough_phy_pages(1)) {
      virt_page_id = dp_pages.free_virt_page_list.front();
      dp_pages.free_virt_page_list.pop_front();
      dp_pages.num_free_virt_pages--;
      consume_phy_pages(1);
      break;
    }

    // Check if we're out of resources
    if (dp_pages.free_virt_page_list.empty()) {
      throw std::runtime_error("No free virtual pages left for dp_rank " +
                               std::to_string(dp_rank));
    }
    if (!has_enough_phy_pages(1)) {
      throw std::runtime_error("No free physical pages left");
    }

    if (!enable_page_prealloc_) {
      throw std::runtime_error(
          "Inconsistent page allocator state: no pages available");
    }

    // Wait for background preallocation or page freeing
    cond_.wait(lock);
  }

  CHECK(virt_page_id.has_value()) << "Virtual page ID should be set";

  // Release lock before mapping (slow path)
  lock.unlock();

  try {
    map_virt_pages(dp_rank, {*virt_page_id});
  } catch (const std::exception& e) {
    // If mapping fails, return page to free list and restore phy pages
    std::lock_guard<std::mutex> guard(mtx_);
    dp_pages.free_virt_page_list.push_front(*virt_page_id);
    dp_pages.num_free_virt_pages++;
    release_phy_pages(1);
    cond_.notify_all();
    throw std::runtime_error("Failed to map virtual page " +
                             std::to_string(*virt_page_id) + ": " + e.what());
  }

  if (enable_page_prealloc_) {
    trigger_preallocation();
  }

  update_memory_usage();
  return std::make_unique<VirtPage>(*virt_page_id, page_size_);
}

void PageAllocator::free_kv_cache_page(int32_t dp_rank, int64_t virt_page_id) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  auto& dp_pages = dp_group_pages_[dp_rank];

  {
    std::lock_guard<std::mutex> lock(mtx_);

    dp_pages.num_free_virt_pages++;
    if (dp_pages.reserved_virt_page_list.size() <
        static_cast<size_t>(max_reserved_pages_)) {
      // Fast path: keep page mapped for reuse (don't release phy pages)
      dp_pages.reserved_virt_page_list.push_back(virt_page_id);
      update_memory_usage();
      cond_.notify_all();
      return;
    }
  }

  // Slow path: unmap physical pages and add to free list
  unmap_virt_pages(dp_rank, {virt_page_id});
  {
    std::lock_guard<std::mutex> lock(mtx_);
    dp_pages.free_virt_page_list.push_back(virt_page_id);
    release_phy_pages(1);
    update_memory_usage();
    cond_.notify_all();
  }
}

void PageAllocator::free_kv_cache_pages(
    int32_t dp_rank,
    const std::vector<int64_t>& virt_page_ids) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  auto& dp_pages = dp_group_pages_[dp_rank];
  std::vector<int64_t> pages_to_unmap;

  {
    std::lock_guard<std::mutex> lock(mtx_);

    dp_pages.num_free_virt_pages += virt_page_ids.size();
    size_t num_to_reserve =
        max_reserved_pages_ - dp_pages.reserved_virt_page_list.size();

    if (num_to_reserve > 0) {
      // Fast path: keep some pages mapped for reuse
      size_t actual_reserve = std::min(num_to_reserve, virt_page_ids.size());
      for (size_t i = 0; i < actual_reserve; ++i) {
        dp_pages.reserved_virt_page_list.push_back(virt_page_ids[i]);
      }
      cond_.notify_all();

      // Remaining pages need to be unmapped
      for (size_t i = actual_reserve; i < virt_page_ids.size(); ++i) {
        pages_to_unmap.push_back(virt_page_ids[i]);
      }
    } else {
      pages_to_unmap = virt_page_ids;
    }
  }

  if (pages_to_unmap.empty()) {
    update_memory_usage();
    return;
  }

  // Slow path: unmap physical pages
  unmap_virt_pages(dp_rank, pages_to_unmap);
  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int64_t virt_page_id : pages_to_unmap) {
      dp_pages.free_virt_page_list.push_back(virt_page_id);
    }
    release_phy_pages(pages_to_unmap.size());
    update_memory_usage();
    cond_.notify_all();
  }
}

void PageAllocator::trim_kv_cache(int32_t dp_rank) {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";

  auto& dp_pages = dp_group_pages_[dp_rank];
  std::vector<int64_t> pages_to_unmap;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    pages_to_unmap.assign(dp_pages.reserved_virt_page_list.begin(),
                          dp_pages.reserved_virt_page_list.end());
    dp_pages.reserved_virt_page_list.clear();

    if (pages_to_unmap.empty()) {
      update_memory_usage();
      return;
    }
  }

  unmap_virt_pages(dp_rank, pages_to_unmap);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int64_t virt_page_id : pages_to_unmap) {
      dp_pages.free_virt_page_list.push_back(virt_page_id);
    }
    release_phy_pages(pages_to_unmap.size());
    update_memory_usage();
  }
}

bool PageAllocator::alloc_weight_pages(size_t num_pages) {
  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    // Allocate physical pages directly for weight tensor
    // All-or-nothing: either allocate all requested pages or fail
    if (num_free_phy_pages_ < num_pages) {
      LOG(ERROR) << "Not enough physical pages for weight allocation: "
                 << "requested " << num_pages << ", available "
                 << num_free_phy_pages_;
      return false;
    }

    num_free_phy_pages_ -= num_pages;
    update_memory_usage();
  }

  // Map weight tensor (full map)
  try {
    auto& allocator = XTensorAllocator::get_instance();
    allocator.broadcast_map_weight_tensor(num_pages);
  } catch (const std::exception& e) {
    // Rollback on failure
    std::lock_guard<std::mutex> lock(mtx_);
    num_free_phy_pages_ += num_pages;
    update_memory_usage();
    LOG(ERROR) << "Failed to map weight tensor: " << e.what();
    return false;
  }

  VLOG(1) << "Allocated and mapped " << num_pages
          << " physical pages for weight tensor";
  return true;
}

void PageAllocator::free_weight_pages(size_t num_pages) {
  // Unmap weight tensor first (full unmap)
  try {
    auto& allocator = XTensorAllocator::get_instance();
    allocator.broadcast_unmap_weight_tensor();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to unmap weight tensor: " << e.what();
    // Continue to release pages anyway
  }

  {
    std::lock_guard<std::mutex> lock(mtx_);

    CHECK(initialized_) << "PageAllocator not initialized";

    num_free_phy_pages_ += num_pages;
    update_memory_usage();
    cond_.notify_all();
  }

  VLOG(1) << "Unmapped and freed " << num_pages
          << " physical pages from weight tensor";
}

size_t PageAllocator::get_num_free_virt_pages(int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  return dp_group_pages_[dp_rank].num_free_virt_pages;
}

size_t PageAllocator::get_num_inuse_virt_pages(int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  return num_total_virt_pages_ - dp_group_pages_[dp_rank].num_free_virt_pages;
}

size_t PageAllocator::get_num_total_virt_pages() const {
  return num_total_virt_pages_;
}

size_t PageAllocator::get_num_reserved_virt_pages(int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, dp_size_) << "dp_rank must be < dp_size";
  std::lock_guard<std::mutex> lock(mtx_);
  return dp_group_pages_[dp_rank].reserved_virt_page_list.size();
}

size_t PageAllocator::get_num_free_phy_pages() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_free_phy_pages_;
}

size_t PageAllocator::get_num_total_phy_pages() const {
  return num_total_phy_pages_;
}

int64_t PageAllocator::get_virt_page_id(int64_t block_id,
                                        size_t block_mem_size) const {
  return block_id * block_mem_size / page_size_;
}

offset_t PageAllocator::get_offset(int64_t virt_page_id) const {
  // Offset for single-layer XTensor map/unmap
  return virt_page_id * page_size_;
}

void PageAllocator::prealloc_worker() {
  while (prealloc_running_.load()) {
    // Per-DP group pages to reserve
    std::vector<std::pair<int32_t, std::vector<int64_t>>> dp_pages_to_reserve;

    {
      std::unique_lock<std::mutex> lock(mtx_);

      // Wait until preallocation is needed or thread is stopped
      cond_.wait(lock, [this] {
        return prealloc_needed_.load() || !prealloc_running_.load();
      });

      if (!prealloc_running_.load()) {
        break;
      }

      prealloc_needed_ = false;

      // Check each DP group for preallocation needs
      for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
        auto& dp_pages = dp_group_pages_[dp_rank];

        size_t current_reserved = dp_pages.reserved_virt_page_list.size();
        size_t to_reserve = 0;
        if (current_reserved < static_cast<size_t>(min_reserved_pages_)) {
          to_reserve = min_reserved_pages_ - current_reserved;
        }

        // Limit by available free virtual pages
        to_reserve = std::min(to_reserve, dp_pages.free_virt_page_list.size());

        // Limit by available physical pages (shared resource)
        size_t max_by_phy = num_free_phy_pages_ / phy_pages_per_virt_page_;
        to_reserve = std::min(to_reserve, max_by_phy);

        if (to_reserve == 0) {
          continue;
        }

        // Get pages from free list and consume physical pages
        std::vector<int64_t> pages_to_reserve;
        for (size_t i = 0;
             i < to_reserve && !dp_pages.free_virt_page_list.empty();
             ++i) {
          pages_to_reserve.push_back(dp_pages.free_virt_page_list.front());
          dp_pages.free_virt_page_list.pop_front();
        }
        consume_phy_pages(pages_to_reserve.size());
        dp_pages_to_reserve.emplace_back(dp_rank, std::move(pages_to_reserve));
      }
    }

    // Map pages for each DP group
    for (auto& [dp_rank, pages_to_reserve] : dp_pages_to_reserve) {
      if (pages_to_reserve.empty()) {
        continue;
      }

      auto& dp_pages = dp_group_pages_[dp_rank];

      try {
        map_virt_pages(dp_rank, pages_to_reserve);
        {
          std::lock_guard<std::mutex> lock(mtx_);
          for (int64_t virt_page_id : pages_to_reserve) {
            dp_pages.reserved_virt_page_list.push_back(virt_page_id);
          }
          update_memory_usage();
          cond_.notify_all();
        }
        VLOG(1) << "Preallocated " << pages_to_reserve.size()
                << " virtual pages for dp_rank=" << dp_rank
                << ", reserved=" << dp_pages.reserved_virt_page_list.size();
      } catch (const std::exception& e) {
        // If mapping fails, return pages to free list and release phy pages
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto it = pages_to_reserve.rbegin(); it != pages_to_reserve.rend();
             ++it) {
          dp_pages.free_virt_page_list.push_front(*it);
        }
        release_phy_pages(pages_to_reserve.size());
        cond_.notify_all();
        LOG(ERROR) << "Failed to preallocate " << pages_to_reserve.size()
                   << " virtual pages for dp_rank=" << dp_rank << ": "
                   << e.what();
      }
    }
  }
}

void PageAllocator::start_prealloc_thread_internal() {
  if (prealloc_thd_ == nullptr) {
    prealloc_running_ = true;
    prealloc_thd_ =
        std::make_unique<std::thread>(&PageAllocator::prealloc_worker, this);

    // Initial preallocation trigger
    trigger_preallocation();
  }
}

void PageAllocator::stop_prealloc_thread(double timeout) {
  if (prealloc_thd_ != nullptr) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      prealloc_running_ = false;
      cond_.notify_all();
    }

    if (prealloc_thd_->joinable()) {
      auto future =
          std::async(std::launch::async, [this]() { prealloc_thd_->join(); });

      if (future.wait_for(std::chrono::duration<double>(timeout)) ==
          std::future_status::timeout) {
        LOG(WARNING) << "Preallocation thread did not stop within timeout";
      }
    }
    prealloc_thd_.reset();
    VLOG(1) << "Stopped page preallocation thread";
  }
}

void PageAllocator::trigger_preallocation() {
  std::lock_guard<std::mutex> lock(mtx_);
  prealloc_needed_ = true;
  cond_.notify_all();
}

void PageAllocator::map_virt_pages(int32_t dp_rank,
                                   const std::vector<int64_t>& virt_page_ids) {
  // Convert virtual page IDs to offsets (for single-layer XTensor)
  std::vector<offset_t> offsets;
  offsets.reserve(virt_page_ids.size());

  for (int64_t virt_page_id : virt_page_ids) {
    offsets.push_back(get_offset(virt_page_id));
  }

  // Broadcast to workers in this DP group
  auto& allocator = XTensorAllocator::get_instance();
  allocator.broadcast_map_to_kv_tensors(dp_rank, offsets);
}

void PageAllocator::unmap_virt_pages(
    int32_t dp_rank,
    const std::vector<int64_t>& virt_page_ids) {
  // Convert virtual page IDs to offsets (for single-layer XTensor)
  std::vector<offset_t> offsets;
  offsets.reserve(virt_page_ids.size());

  for (int64_t virt_page_id : virt_page_ids) {
    offsets.push_back(get_offset(virt_page_id));
  }

  // Broadcast to workers in this DP group
  auto& allocator = XTensorAllocator::get_instance();
  allocator.broadcast_unmap_from_kv_tensors(dp_rank, offsets);
}

void PageAllocator::update_memory_usage() {
  // Note: Caller must hold mtx_
  size_t free_phy_pages = num_free_phy_pages_;

  // Calculate physical memory usage
  size_t used_phy_mem = (num_total_phy_pages_ - free_phy_pages) * page_size_;

  VLOG(2) << "Memory usage: "
          << "dp_size=" << dp_size_ << ", free_phy_pages=" << free_phy_pages
          << ", used_phy_mem=" << used_phy_mem / (1024 * 1024) << "MB";
}

}  // namespace xllm
