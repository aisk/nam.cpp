#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Wav {
  uint32_t rate;
  std::vector<float> samples;

  void write(const std::string &path) const;
};

Wav read_wav(const std::string &path);
