#include "inference.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

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

struct GraphSession {
  ggml_backend_t backend = nullptr;
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *output = nullptr;
  std::vector<std::pair<ggml_tensor *, const float *>> loads;

  GraphSession(const Model &m, size_t input_size, int threads) {
    backend = ggml_backend_cpu_init();
    if (!backend) {
      std::fprintf(stderr, "Failed to initialize the ggml CPU backend\n");
      std::abort();
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_init_params ip{64 * 1024 * 1024, nullptr, true};
    ctx = ggml_init(ip);
    if (!ctx) {
      std::fprintf(stderr, "ggml_init failed\n");
      std::abort();
    }

    input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, 1);
    ggml_tensor *x = input, *condition = input, *head_sum = nullptr;
    for (const auto &a : m.arrays) {
      x = conv(ctx, x, a.rechannel, m, loads);
      int out_no_head = int(std::min(x->ne[0], condition->ne[0]) -
                            (a.receptive - a.head.kernel));
      for (const auto &l : a.layers) {
        auto *z = conv(ctx, x, l.conv, m, loads);
        auto *mix =
            right(ctx, conv(ctx, condition, l.mixer, m, loads), z->ne[0]);
        z = ggml_add(ctx, z, mix);
        z = (l.activation == "ReLU") ? ggml_relu(ctx, z) : ggml_tanh(ctx, z);
        auto *term = right(ctx, z, out_no_head);
        head_sum = head_sum
                       ? ggml_add(ctx, right(ctx, head_sum, out_no_head), term)
                       : term;
        auto *res = conv(ctx, z, l.layer1x1, m, loads);
        x = ggml_add(ctx, right(ctx, x, res->ne[0]), res);
      }
      head_sum = conv(ctx, head_sum, a.head, m, loads);
      x = right(ctx, x, head_sum->ne[0]);
    }
    output = ggml_scale(ctx, head_sum, m.head_scale);
    graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, output);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
      std::fprintf(stderr, "Failed to allocate the ggml backend buffer\n");
      std::abort();
    }
    for (const auto &p : loads) {
      ggml_backend_tensor_set(p.first, p.second, 0, ggml_nbytes(p.first));
    }
  }

  GraphSession(const GraphSession &) = delete;
  GraphSession &operator=(const GraphSession &) = delete;

  void compute(const float *in, size_t input_size, float *out,
               size_t output_size) {
    if (input_size != size_t(input->ne[0]) ||
        output_size > size_t(output->ne[0])) {
      std::fprintf(stderr, "GraphSession buffer size mismatch\n");
      std::abort();
    }
    ggml_backend_tensor_set(input, in, 0, input_size * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "ggml graph computation failed\n");
      std::abort();
    }
    ggml_backend_tensor_get(output, out, 0, output_size * sizeof(float));
  }

  ~GraphSession() {
    if (buffer) {
      ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
      ggml_free(ctx);
    }
    if (backend) {
      ggml_backend_free(backend);
    }
  }
};

std::vector<float> infer(const Model &m, const std::vector<float> &raw,
                         int threads) {
  std::vector<float> padded(size_t(m.receptive - 1) + raw.size(), 0);
  std::copy(raw.begin(), raw.end(), padded.begin() + m.receptive - 1);
  GraphSession session(m, padded.size(), threads);
  std::vector<float> out(raw.size());
  session.compute(padded.data(), padded.size(), out.data(), out.size());
  return out;
}

std::vector<float> infer_streaming(const Model &m,
                                   const std::vector<float> &raw, int threads,
                                   size_t block_size) {
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
