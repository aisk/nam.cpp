#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <argparse/argparse.hpp>
#include <json-c/json.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

static json_object *get(json_object *o, const char *key) {
  json_object *v = nullptr;
  if (!json_object_object_get_ex(o, key, &v)) {
    throw std::runtime_error(std::string("Missing field: ") + key);
  }
  return v;
}

static int ji(json_object *o, const char *key, int fallback = -1) {
  json_object *v = nullptr;
  return json_object_object_get_ex(o, key, &v) ? json_object_get_int(v)
                                               : fallback;
}

static bool jb(json_object *o, const char *key, bool fallback) {
  json_object *v = nullptr;
  return json_object_object_get_ex(o, key, &v) ? json_object_get_boolean(v)
                                               : fallback;
}

static std::string js(json_object *v) {
  if (json_object_is_type(v, json_type_string)) {
    return json_object_get_string(v);
  }
  throw std::runtime_error("Only string activation values are supported");
}

static void take(Conv &c, size_t &p) {
  c.offset = p;
  p += size_t(c.in) * c.out * c.kernel + (c.bias ? c.out : 0);
}

static Model load_model(const std::string &path) {
  json_object *root = json_object_from_file(path.c_str());
  if (!root) {
    throw std::runtime_error("Failed to parse NAM JSON: " + path);
  }

  struct Guard {
    json_object *p;

    ~Guard() { json_object_put(p); }
  } guard{root};

  if (js(get(root, "architecture")) != "WaveNet") {
    throw std::runtime_error("Only the A1 WaveNet architecture is supported");
  }
  json_object *cfg = get(root, "config"), *wa = get(root, "weights"),
              *as = get(cfg, "layers");
  Model m;
  m.weights.reserve(json_object_array_length(wa));
  for (size_t i = 0; i < json_object_array_length(wa); ++i) {
    m.weights.push_back(
        float(json_object_get_double(json_object_array_get_idx(wa, i))));
  }
  size_t p = 0;
  for (size_t ai = 0; ai < json_object_array_length(as); ++ai) {
    json_object *a = json_object_array_get_idx(as, ai);
    json_object *ds = get(a, "dilations");
    json_object *hc = nullptr;
    json_object_object_get_ex(a, "head", &hc);
    int input = ji(a, "input_size"), cond = ji(a, "condition_size"),
        ch = ji(a, "channels"), bottleneck = ji(a, "bottleneck", ch);
    if (cond != 1 || ji(a, "groups_input", 1) != 1 ||
        ji(a, "groups_input_mixin", 1) != 1) {
      throw std::runtime_error(
          "Only mono conditioning and groups=1 are supported");
    }
    json_object *act = get(a, "activation");
    if (json_object_is_type(act, json_type_array)) {
      throw std::runtime_error("Per-layer activation lists are not supported");
    }
    std::string activation = js(act);
    if (activation != "ReLU" && activation != "Tanh") {
      throw std::runtime_error("Only ReLU and Tanh activations are supported");
    }
    if (jb(a, "gated", false)) {
      throw std::runtime_error("Gated activations are not supported");
    }
    Array ar;
    ar.rechannel = {input, ch, 1, 1, false, 0};
    take(ar.rechannel, p);
    int common_k = ji(a, "kernel_size", -1);
    json_object *ks = nullptr;
    json_object_object_get_ex(a, "kernel_sizes", &ks);
    for (size_t li = 0; li < json_object_array_length(ds); ++li) {
      int d = json_object_get_int(json_object_array_get_idx(ds, li));
      int k = ks ? json_object_get_int(json_object_array_get_idx(ks, li))
                 : common_k;
      Layer l{{ch, bottleneck, k, d, true, 0},
              {cond, bottleneck, 1, 1, false, 0},
              {bottleneck, ch, 1, 1, true, 0},
              activation};
      take(l.conv, p);
      take(l.mixer, p);
      take(l.layer1x1, p);
      ar.layers.push_back(l);
      ar.receptive += (k - 1) * d;
    }
    if (hc != nullptr) {
      ar.head = {bottleneck, ji(hc, "out_channels"), ji(hc, "kernel_size"),
                 1,          jb(hc, "bias", false),  0};
    } else {
      ar.head = {bottleneck, ji(a, "head_size"),        1,
                 1,          jb(a, "head_bias", false), 0};
    }
    take(ar.head, p);
    ar.receptive += ar.head.kernel - 1;
    m.receptive += ar.receptive - 1;
    m.arrays.push_back(ar);
  }
  if (p + 1 != m.weights.size()) {
    throw std::runtime_error(
        "Weight count does not match the classic A1 layout "
        "(expected " +
        std::to_string(p + 1) + ", got " + std::to_string(m.weights.size()) +
        ")");
  }
  m.head_scale = m.weights[p];
  return m;
}

static uint32_t u32(std::istream &f) {
  uint32_t v;
  f.read(reinterpret_cast<char *>(&v), 4);
  return v;
}

static uint16_t u16(std::istream &f) {
  uint16_t v;
  f.read(reinterpret_cast<char *>(&v), 2);
  return v;
}

struct Wav {
  uint32_t rate;
  std::vector<float> samples;
};

static Wav read_wav(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  char id[4];
  f.read(id, 4);
  (void)u32(f);
  char wave[4];
  f.read(wave, 4);
  if (!f || std::memcmp(id, "RIFF", 4) || std::memcmp(wave, "WAVE", 4)) {
    throw std::runtime_error("Input is not a RIFF/WAVE file");
  }
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  std::vector<char> data;
  while (f.read(id, 4)) {
    uint32_t n = u32(f);
    std::streampos next = f.tellg();
    next += n + (n & 1);
    if (!std::memcmp(id, "fmt ", 4)) {
      format = u16(f);
      channels = u16(f);
      rate = u32(f);
      (void)u32(f);
      (void)u16(f);
      bits = u16(f);
      if (format == 0xfffe && n >= 40) {
        (void)u16(f);
        (void)u16(f);
        (void)u32(f);
        format = u16(f);
      }
    } else if (!std::memcmp(id, "data", 4)) {
      data.resize(n);
      f.read(data.data(), n);
    }
    f.seekg(next);
  }
  if (format != 3 || channels != 1 || bits != 32 || data.empty()) {
    throw std::runtime_error("Input must be a mono IEEE float32 WAV file");
  }

  Wav w{rate, std::vector<float>(data.size() / 4)};
  std::memcpy(w.samples.data(), data.data(), data.size());
  return w;
}

static void write_wav(const std::string &path, const Wav &w) {
  std::ofstream f(path, std::ios::binary);
  uint32_t bytes = uint32_t(w.samples.size() * 4), riff = 36 + bytes;
  uint16_t one = 1, fmt = 3, bits = 32, align = 4;
  uint32_t br = w.rate * 4, fs = 16;
  f.write("RIFF", 4);
  f.write((char *)&riff, 4);
  f.write("WAVEfmt ", 8);
  f.write((char *)&fs, 4);
  f.write((char *)&fmt, 2);
  f.write((char *)&one, 2);
  f.write((char *)&w.rate, 4);
  f.write((char *)&br, 4);
  f.write((char *)&align, 2);
  f.write((char *)&bits, 2);
  f.write("data", 4);
  f.write((char *)&bytes, 4);
  f.write((char *)w.samples.data(), bytes);
}

static ggml_tensor *right(ggml_context *c, ggml_tensor *x, int64_t n) {
  if (x->ne[0] == n) {
    return x;
  }
  return ggml_view_2d(c, x, n, x->ne[1], x->nb[1],
                      size_t(x->ne[0] - n) * sizeof(float));
}

static ggml_tensor *
make_weight(ggml_context *c, const Conv &v,
            std::vector<std::pair<ggml_tensor *, const float *>> &loads) {
  auto *t = ggml_new_tensor_3d(c, GGML_TYPE_F32, v.kernel, v.in, v.out);
  loads.push_back({t, nullptr});
  return t;
}

static ggml_tensor *
conv(ggml_context *c, ggml_tensor *x, const Conv &v, const Model &m,
     std::vector<std::pair<ggml_tensor *, const float *>> &loads) {
  auto *w = make_weight(c, v, loads);
  loads.back().second = m.weights.data() + v.offset;
  auto *y = ggml_conv_1d(c, w, x, 1, 0, v.dilation);
  if (v.bias) {
    auto *b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, v.out);
    loads.push_back(
        {b, m.weights.data() + v.offset + size_t(v.in) * v.out * v.kernel});
    y = ggml_add(c, y, b);
  }
  return y;
}

class GraphSession {
public:
  GraphSession(const Model &m, size_t input_size, int threads) {
    backend_ = ggml_backend_cpu_init();
    if (!backend_) {
      throw std::runtime_error("Failed to initialize the ggml CPU backend");
    }
    ggml_backend_cpu_set_n_threads(backend_, threads);
    ggml_init_params ip{64 * 1024 * 1024, nullptr, true};
    ctx_ = ggml_init(ip);
    if (!ctx_) {
      throw std::runtime_error("ggml_init failed");
    }

    input_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, input_size, 1);
    ggml_tensor *x = input_, *condition = input_, *head_sum = nullptr;
    for (const auto &a : m.arrays) {
      x = conv(ctx_, x, a.rechannel, m, loads_);
      int out_no_head = int(std::min(x->ne[0], condition->ne[0]) -
                            (a.receptive - a.head.kernel));
      for (const auto &l : a.layers) {
        auto *z = conv(ctx_, x, l.conv, m, loads_);
        auto *mix =
            right(ctx_, conv(ctx_, condition, l.mixer, m, loads_), z->ne[0]);
        z = ggml_add(ctx_, z, mix);
        z = (l.activation == "ReLU") ? ggml_relu(ctx_, z)
                                      : ggml_tanh(ctx_, z);
        auto *term = right(ctx_, z, out_no_head);
        head_sum = head_sum
                       ? ggml_add(ctx_, right(ctx_, head_sum, out_no_head), term)
                       : term;
        auto *res = conv(ctx_, z, l.layer1x1, m, loads_);
        x = ggml_add(ctx_, right(ctx_, x, res->ne[0]), res);
      }
      head_sum = conv(ctx_, head_sum, a.head, m, loads_);
      x = right(ctx_, x, head_sum->ne[0]);
    }
    output_ = ggml_scale(ctx_, head_sum, m.head_scale);
    graph_ = ggml_new_graph_custom(ctx_, 4096, false);
    ggml_build_forward_expand(graph_, output_);
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buffer_) {
      throw std::runtime_error("Failed to allocate the ggml backend buffer");
    }
    for (const auto &p : loads_) {
      ggml_backend_tensor_set(p.first, p.second, 0, ggml_nbytes(p.first));
    }
  }

  GraphSession(const GraphSession &) = delete;
  GraphSession &operator=(const GraphSession &) = delete;

  ~GraphSession() {
    if (buffer_) {
      ggml_backend_buffer_free(buffer_);
    }
    if (ctx_) {
      ggml_free(ctx_);
    }
    if (backend_) {
      ggml_backend_free(backend_);
    }
  }

  void compute(const float *input, size_t input_size, float *output,
               size_t output_size) {
    if (input_size != size_t(input_->ne[0]) ||
        output_size > size_t(output_->ne[0])) {
      throw std::runtime_error("GraphSession buffer size mismatch");
    }
    ggml_backend_tensor_set(input_, input, 0, input_size * sizeof(float));
    if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("ggml graph computation failed");
    }
    ggml_backend_tensor_get(output_, output, 0, output_size * sizeof(float));
  }

private:
  ggml_backend_t backend_ = nullptr;
  ggml_context *ctx_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_tensor *input_ = nullptr;
  ggml_tensor *output_ = nullptr;
  std::vector<std::pair<ggml_tensor *, const float *>> loads_;
};

static std::vector<float> infer(const Model &m, const std::vector<float> &raw,
                                int threads) {
  std::vector<float> padded(size_t(m.receptive - 1) + raw.size(), 0);
  std::copy(raw.begin(), raw.end(), padded.begin() + m.receptive - 1);
  GraphSession session(m, padded.size(), threads);
  std::vector<float> out(raw.size());
  session.compute(padded.data(), padded.size(), out.data(), out.size());
  return out;
}

static std::vector<float> infer_streaming(const Model &m,
                                          const std::vector<float> &raw,
                                          int threads, size_t block_size) {
  const size_t history_size = size_t(m.receptive - 1);
  std::vector<float> history(history_size, 0.0f);
  std::vector<float> input(history_size + block_size, 0.0f);
  std::vector<float> out(raw.size());
  GraphSession session(m, input.size(), threads);

  for (size_t offset = 0; offset < raw.size(); offset += block_size) {
    const size_t n = std::min(block_size, raw.size() - offset);
    std::copy(history.begin(), history.end(), input.begin());
    std::copy_n(raw.data() + offset, n, input.data() + history_size);
    std::fill(input.begin() + history_size + n, input.end(), 0.0f);
    session.compute(input.data(), input.size(), out.data() + offset, n);

    if (history_size == 0) {
      continue;
    }
    if (n >= history_size) {
      std::copy_n(raw.data() + offset + n - history_size, history_size,
                  history.data());
    } else {
      std::move(history.begin() + n, history.end(), history.begin());
      std::copy_n(raw.data() + offset, n, history.end() - n);
    }
  }
  return out;
}

int main(int argc, char **argv) {
  try {
    argparse::ArgumentParser program("nam");
    program.add_description("Experimental offline NAM inference using ggml");
    program.add_argument("model").help("NAM model file");
    program.add_argument("input").help("Mono float32 WAV input");
    program.add_argument("output").help("Mono float32 WAV output");
    program.add_argument("-t", "--threads")
        .help("Number of CPU threads")
        .scan<'i', int>()
        .default_value(1);
    program.add_argument("--stream-block")
        .help("Process using a reusable streaming graph with this block size; "
              "0 keeps whole-file offline inference")
        .scan<'i', int>()
        .default_value(0);
    program.parse_args(argc, argv);

    int threads = program.get<int>("--threads");
    if (threads < 1) {
      throw std::runtime_error("Thread count must be at least 1");
    }
    int stream_block = program.get<int>("--stream-block");
    if (stream_block < 0) {
      throw std::runtime_error("Streaming block size must not be negative");
    }

    Model m = load_model(program.get<std::string>("model"));
    Wav w = read_wav(program.get<std::string>("input"));
    w.samples = stream_block == 0
                    ? infer(m, w.samples, threads)
                    : infer_streaming(m, w.samples, threads,
                                      size_t(stream_block));
    write_wav(program.get<std::string>("output"), w);
    std::cout << "Done: " << w.samples.size() << " samples, receptive field "
              << m.receptive << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
