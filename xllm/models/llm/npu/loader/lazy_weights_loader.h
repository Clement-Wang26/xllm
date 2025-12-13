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

#include <acl/acl.h>
#include <torch_npu/torch_npu.h>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "core/platform/stream.h"
#include "core/util/threadpool.h"

namespace xllm {

/**
 * LazyWeightsLoader - Elegant abstraction for on-demand layer weight loading
 *
 * Design principles:
 * - Defers layer weights loading until first forward pass
 * - Sequential layer loading on dedicated stream
 * - Per-layer ACL events for fine-grained synchronization
 */
class LazyWeightsLoader {
 public:
  using LayerLoaderFunc = std::function<void(int32_t layer_idx)>;

  LazyWeightsLoader(int32_t num_layers, int32_t device_id);

  ~LazyWeightsLoader();

  LazyWeightsLoader(const LazyWeightsLoader&) = delete;
  LazyWeightsLoader& operator=(const LazyWeightsLoader&) = delete;
  LazyWeightsLoader(LazyWeightsLoader&&) = delete;
  LazyWeightsLoader& operator=(LazyWeightsLoader&&) = delete;

  void reset_events();

  void load_weights_to_host(LayerLoaderFunc loader_func);

  void load_weights_to_device(LayerLoaderFunc loader_func);

  void offload_weights_to_host(LayerLoaderFunc loader_func);

  void set_loaded_to_device(bool loaded) { is_loaded_to_device_ = loaded; }

  void set_loaded_to_host(bool loaded) { is_loaded_to_host_ = loaded; }

  bool is_loaded_to_device() const { return is_loaded_to_device_; }

  bool is_loaded_to_host() const { return is_loaded_to_host_; }

  void wait_for_layer(int32_t layer_idx);

 private:
  const int32_t num_layers_;
  const int32_t device_id_;

  void load_layer_to_device(int32_t layer_idx, LayerLoaderFunc loader_func);
  bool is_loaded_to_device_{false};
  bool is_loaded_to_host_{false};

  c10_npu::NPUStream load_stream_;
  std::unique_ptr<ThreadPool> threadpool_;
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_recorded_;
};

}  // namespace xllm