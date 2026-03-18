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

#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>

#include "json_reader.h"

namespace xllm {
namespace util {

inline std::optional<std::string> try_get_model_type_from_config(
    const std::filesystem::path& model_path) {
  const auto config_json_path = model_path / "config.json";
  JsonReader reader;
  try {
    if (!reader.parse(config_json_path.string())) {
      return std::nullopt;
    }
  } catch (const std::exception&) {
    return std::nullopt;
  }

  auto model_type = reader.value<std::string>("model_type");
  if (!model_type.has_value()) {
    model_type = reader.value<std::string>("model_name");
  }
  return model_type;
}

inline bool is_mla_model_type(const std::string& model_type) {
  static const std::unordered_set<std::string> kMlaModelTypes = {
      "deepseek_v2",       "deepseek_v3",      "deepseek_v32",
      "deepseek_v3_mtp",   "deepseek_v32_mtp", "kimi_k2",
      "glm4_moe_lite",     "glm5_moe",         "glm_moe_dsa",
      "joyai_llm_flash"};
  return kMlaModelTypes.find(model_type) != kMlaModelTypes.end();
}

}  // namespace util
}  // namespace xllm
