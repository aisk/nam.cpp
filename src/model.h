#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Conv {
  int in, out, kernel, dilation;
  bool bias;
  size_t offset;
};

struct Layer {
  Conv conv, mixer, layer1x1;
  std::string activation;
};

struct Array {
  Conv rechannel, head;
  std::vector<Layer> layers;
  int receptive = 1;
};

struct Model {
  std::vector<Array> arrays;
  std::vector<float> weights;
  float head_scale = 1;
  int receptive = 1;
};

Model load_model(const std::string &path);
