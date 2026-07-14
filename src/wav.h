#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Wav {
  uint32_t rate;
  std::vector<float> samples;
};

Wav read_wav(const std::string &path);
void write_wav(const std::string &path, const Wav &w);
