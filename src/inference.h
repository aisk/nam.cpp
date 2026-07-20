#pragma once

#include "model.h"

#include <cstddef>
#include <memory>
#include <vector>

struct GraphSession;

// Stateful block-by-block inference for online/real-time use. The session is
// prewarmed on construction, so the first processed block is already past the
// zero-history transient.
class StreamSession {
public:
  StreamSession(const Model &m, size_t block_size, int threads);
  ~StreamSession();
  StreamSession(const StreamSession &) = delete;
  StreamSession &operator=(const StreamSession &) = delete;

  size_t block_size() const { return block; }

  // Consumes exactly block_size input samples and writes block_size output
  // samples. Feeding anything but real consecutive samples (e.g. zero
  // padding) pollutes the internal state, so only pad the final block of a
  // stream and discard the padded tail of its output.
  void process(const float *in, float *out);

private:
  std::unique_ptr<GraphSession> impl;
  size_t block;
};

std::vector<float> infer(const Model &m, const std::vector<float> &raw,
                         int threads);
std::vector<float> infer_streaming(const Model &m,
                                   const std::vector<float> &raw, int threads,
                                   size_t block_size);
