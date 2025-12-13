include(cc_library)

# Define the library
cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
    llm/npu/loader/lazy_weights_loader.h
  SRCS
    model_registry.cpp
    llm/npu/loader/lazy_weights_loader.cpp
  DEPS
    :model
)
