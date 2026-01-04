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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "virt_page.h"
#include "xtensor.h"  // For offset_t type definition

namespace xllm {

// Configuration constants
constexpr int32_t MIN_RESERVED_PAGES = 8;
constexpr int32_t MAX_RESERVED_PAGES = 32;
constexpr bool PAGE_PREALLOC_ENABLED = true;
constexpr double PREALLOC_THREAD_TIMEOUT = 2.0;  // seconds

/**
 * PageAllocator manages virtual page allocation for KV cache.
 *
 * Key concepts:
 * - VirtPage: Logical page for KV cache indexing, based on single-layer memory
 * - PhyPage: Physical memory page (2MB), managed by PhyPagePool
 *
 * Memory layout:
 * - For non-contiguous: each layer has its own K and V XTensor
 *   - mem_size_per_layer = total_phy_mem / (2 * num_layers)
 *   - num_virt_pages = mem_size_per_layer / virt_page_size
 *   - Allocating 1 virt_page consumes (2 * num_layers) phy_pages
 *
 * - For contiguous: all layers share one K and one V XTensor
 *   - contiguous_page_size = virt_page_size * num_layers
 *   - num_virt_pages = total_phy_mem / 2 / contiguous_page_size
 *   - Allocating 1 virt_page consumes (2 * contiguous_page_size /
 * phy_page_size) phy_pages
 *
 * Offset calculation:
 * - offset = virt_page_id * virt_page_size (for single-layer XTensor)
 * - This offset is used for XTensor::map/unmap operations
 *
 * This is a singleton class shared by all XTensorBlockManagerImpl instances.
 */
class PageAllocator {
 public:
  // Get the global singleton instance
  static PageAllocator& get_instance() {
    static PageAllocator allocator;
    return allocator;
  }

  // Initialize the allocator
  // num_layers: number of transformer layers
  // num_phy_pages: total number of physical pages from PhyPagePool
  // dp_size: number of data parallel groups
  // enable_page_prealloc: whether to enable background preallocation
  void init(int64_t num_layers,
            size_t num_phy_pages,
            int32_t dp_size = 1,
            bool enable_page_prealloc = PAGE_PREALLOC_ENABLED);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // Start preallocation thread (called after reserving null block)
  void start_prealloc_thread();

  // ============ KV Cache Page Allocation ============
  // Allocate a virtual page for KV cache
  // dp_rank: which DP group this allocation is for
  // Consumes phy_pages_per_virt_page_ physical pages
  // Returns nullptr if no physical pages available
  std::unique_ptr<VirtPage> alloc_kv_cache_page(int32_t dp_rank);

  // Free a single KV cache virtual page
  void free_kv_cache_page(int32_t dp_rank, int64_t virt_page_id);

  // Free multiple KV cache virtual pages
  void free_kv_cache_pages(int32_t dp_rank,
                           const std::vector<int64_t>& virt_page_ids);

  // Trim reserved KV cache pages (unmap physical pages)
  void trim_kv_cache(int32_t dp_rank);

  // ============ Weight Page Allocation ============
  // Allocate physical pages for weight tensor (full map)
  // num_pages: number of physical pages (aligned up from weight size)
  // All-or-nothing: returns true if all pages allocated, false otherwise
  bool alloc_weight_pages(size_t num_pages);

  // Free physical pages from weight tensor
  // num_pages: same count used in alloc_weight_pages
  void free_weight_pages(size_t num_pages);

  // Virtual page getters (for specific DP group)
  size_t get_num_free_virt_pages(int32_t dp_rank) const;
  size_t get_num_inuse_virt_pages(int32_t dp_rank) const;
  size_t get_num_total_virt_pages() const;
  size_t get_num_reserved_virt_pages(int32_t dp_rank) const;

  // Physical page getters
  size_t get_num_free_phy_pages() const;
  size_t get_num_total_phy_pages() const;

  // Convert block_id to virt_page_id
  int64_t get_virt_page_id(int64_t block_id, size_t block_mem_size) const;

  // Get offset for XTensor map/unmap (based on single-layer)
  offset_t get_offset(int64_t virt_page_id) const;

  // Get configuration
  size_t page_size() const { return page_size_; }
  int64_t num_layers() const { return num_layers_; }

  // Get number of physical pages consumed per virtual page allocation
  size_t phy_pages_per_virt_page() const { return phy_pages_per_virt_page_; }

 private:
  PageAllocator() = default;
  ~PageAllocator();
  PageAllocator(const PageAllocator&) = delete;
  PageAllocator& operator=(const PageAllocator&) = delete;

  // Check if enough physical pages available for allocation
  bool has_enough_phy_pages(size_t num_virt_pages) const;

  // Consume/release physical pages (update tracking)
  void consume_phy_pages(size_t count);
  void release_phy_pages(size_t count);

  // Preallocation worker thread function
  void prealloc_worker();

  // Start/stop preallocation thread
  void start_prealloc_thread_internal();
  void stop_prealloc_thread(double timeout = PREALLOC_THREAD_TIMEOUT);

  // Trigger preallocation
  void trigger_preallocation();

  // Map/unmap virtual pages (broadcasts to workers in dp_rank group)
  void map_virt_pages(int32_t dp_rank,
                      const std::vector<int64_t>& virt_page_ids);
  void unmap_virt_pages(int32_t dp_rank,
                        const std::vector<int64_t>& virt_page_ids);

  // Update memory usage tracking
  void update_memory_usage();

  // Initialization state
  bool initialized_ = false;

  // Configuration
  int64_t num_layers_ = 0;
  int32_t dp_size_ = 1;
  size_t page_size_ = 0;  // Page size (from FLAGS_phy_page_granularity_size)
  bool enable_page_prealloc_ = PAGE_PREALLOC_ENABLED;

  // Derived values
  size_t num_total_virt_pages_ =
      0;  // Total virtual pages (based on single layer)
  size_t num_total_phy_pages_ = 0;  // Total physical pages from PhyPagePool
  size_t phy_pages_per_virt_page_ =
      0;  // Physical pages consumed per virt_page alloc

  // Per-DP group virtual page tracking
  // Each DP group has its own free_virt_page_list and reserved_virt_page_list
  struct DpGroupPages {
    size_t num_free_virt_pages{0};            // Protected by mtx_
    std::deque<int64_t> free_virt_page_list;  // Unmapped virtual pages
    std::deque<int64_t>
        reserved_virt_page_list;  // Mapped virtual pages ready for use
  };
  std::vector<DpGroupPages> dp_group_pages_;

  // Physical page tracking (shared across all DP groups)
  size_t num_free_phy_pages_ = 0;  // Available physical pages

  // Reserved page limits
  int32_t min_reserved_pages_ = MIN_RESERVED_PAGES;
  int32_t max_reserved_pages_ = MAX_RESERVED_PAGES;

  // Threading
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  std::atomic<bool> prealloc_running_{false};
  std::atomic<bool> prealloc_needed_{false};
  std::unique_ptr<std::thread> prealloc_thd_;
};

}  // namespace xllm
