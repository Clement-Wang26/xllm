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

#include "lazy_layer_loader.h"

#include <glog/logging.h>

#include <thread>

namespace xllm {

LazyLayerLoader::LazyLayerLoader(int32_t num_layers, int32_t device_id)
    : num_layers_(num_layers),
      device_id_(device_id),
      events_(num_layers, nullptr),
      event_recorded_(num_layers) {
  uint32_t flags = ACL_EVENT_SYNC;
  threadpool_ = std::make_unique<ThreadPool>(num_layers);

  for (int32_t i = 0; i < num_layers; ++i) {
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK_EQ(ret, ACL_SUCCESS) << "Failed to create event for layer " << i;
    load_streams_.emplace_back(c10_npu::getNPUStreamFromPool());
    event_recorded_[i].store(false, std::memory_order_relaxed);
  }

  LOG(INFO) << "LazyLayerLoader initialized for " << num_layers << " layers";
}

LazyLayerLoader::~LazyLayerLoader() {
  threadpool_.reset();

  for (int i = 0; i < events_.size(); i++) {
    if (events_[i] != nullptr) {
      aclrtDestroyEvent(events_[i]);
    }
  }
}

void LazyLayerLoader::start_async_loading(LayerLoader handle) {
  bool expected = false;
  if (!started_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }

  LOG(INFO) << "Starting asynchronous layer loading for " << num_layers_
            << " layers";

  for (int32_t i = 0; i < num_layers_; ++i) {
    threadpool_->schedule([this, i, handle]() { this->load_layer(i, handle); });
  }
}

void LazyLayerLoader::load_layer(int32_t layer_idx, LayerLoader handle) {
  c10_npu::SetDevice(device_id_);
  auto stream_guard = c10::StreamGuard(load_streams_[layer_idx % 8].unwrap());
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  LOG(INFO) << "handle layer " << layer_idx << " (load + verify + merge) "
            << stream;

  while (layer_idx >= 8 &&
         !event_recorded_[layer_idx - 8].load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  handle(layer_idx);

  // Capture current layer load task.
  auto ret =
      aclrtRecordEvent(events_[layer_idx], load_streams_[layer_idx].stream());
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to record event for layer " << layer_idx;

  event_recorded_[layer_idx].store(true, std::memory_order_release);

  LOG(INFO) << "Layer " << layer_idx << " fully loaded";
}

void LazyLayerLoader::wait_for_layer(int32_t layer_idx) {
  while (!event_recorded_[layer_idx].load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  auto ret = aclrtStreamWaitEvent(stream, events_[layer_idx]);
  CHECK_EQ(ret, ACL_SUCCESS) << "Failed to sync layer " << layer_idx;
}
}  // namespace xllm
