#pragma once

#include "model.h"

#include <cstddef>
#include <vector>

std::vector<float> infer(const Model &m, const std::vector<float> &raw,
                         int threads);
std::vector<float> infer_streaming(const Model &m,
                                   const std::vector<float> &raw, int threads,
                                   size_t block_size);
