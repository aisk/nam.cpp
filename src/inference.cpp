#include "inference.h"

#include "ggml-alloc.h"
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
make_weight(ggml_context *cw, const Conv &v,
            std::vector<std::pair<ggml_tensor *, const float *>> &loads) {
  auto *t = ggml_new_tensor_3d(cw, GGML_TYPE_F32, v.kernel, v.in, v.out);
  loads.push_back({t, nullptr});
  return t;
}

static ggml_tensor *
conv(ggml_context *cw, ggml_context *cg, ggml_tensor *x, const Conv &v,
     const Model &m,
     std::vector<std::pair<ggml_tensor *, const float *>> &loads) {
  auto *w = make_weight(cw, v, loads);
  loads.back().second = m.weights.data() + v.offset;
  auto *y = ggml_conv_1d(cg, w, x, 1, 0, v.dilation);
  if (v.bias) {
    auto *b = ggml_new_tensor_2d(cw, GGML_TYPE_F32, 1, v.out);
    loads.push_back(
        {b, m.weights.data() + v.offset + size_t(v.in) * v.out * v.kernel});
    y = ggml_add(cg, y, b);
  }
  return y;
}

struct GraphSession {
  ggml_backend_t backend = nullptr;
  ggml_context *ctx_w = nullptr;
  ggml_context *ctx_g = nullptr;
  ggml_backend_buffer_t weight_buffer = nullptr;
  ggml_gallocr_t galloc = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *output = nullptr;
  std::vector<std::pair<ggml_tensor *, const float *>> loads;

  struct Cache {
    ggml_tensor *in, *out;
    std::vector<float> host;
  };
  std::vector<Cache> caches;

  // Prepends this conv's cached input history and registers the tail of the
  // combined sequence for readback, so the next block resumes seamlessly.
  ggml_tensor *with_cache(ggml_tensor *x, const Conv &v) {
    int64_t cols = int64_t(v.kernel - 1) * v.dilation;
    if (cols == 0) {
      return x;
    }
    auto *cache = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, cols, x->ne[1]);
    ggml_set_input(cache);
    auto *xin = ggml_concat(ctx_g, cache, x, 0);
    auto *tail = ggml_cont(ctx_g, right(ctx_g, xin, cols));
    ggml_set_output(tail);
    caches.push_back({cache, tail, std::vector<float>(cols * x->ne[1], 0.0f)});
    return xin;
  }

  GraphSession(const Model &m, size_t input_size, int threads,
               bool streaming) {
    backend = ggml_backend_cpu_init();
    if (!backend) {
      std::fprintf(stderr, "Failed to initialize the ggml CPU backend\n");
      std::abort();
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_init_params ip{64 * 1024 * 1024, nullptr, true};
    ctx_w = ggml_init(ip);
    ctx_g = ggml_init(ip);
    if (!ctx_w || !ctx_g) {
      std::fprintf(stderr, "ggml_init failed\n");
      std::abort();
    }

    input = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, input_size, 1);
    ggml_set_input(input);
    ggml_tensor *x = input, *condition = input, *head_sum = nullptr;
    if (streaming) {
      // Per-conv input caches keep every tensor exactly input_size long, so
      // no cropping or history recomputation is needed between blocks.
      for (const auto &a : m.arrays) {
        x = conv(ctx_w, ctx_g, x, a.rechannel, m, loads);
        for (const auto &l : a.layers) {
          auto *z = conv(ctx_w, ctx_g, with_cache(x, l.conv), l.conv, m, loads);
          auto *mix = conv(ctx_w, ctx_g, condition, l.mixer, m, loads);
          z = ggml_add(ctx_g, z, mix);
          z = (l.activation == "ReLU") ? ggml_relu(ctx_g, z)
                                       : ggml_tanh(ctx_g, z);
          head_sum = head_sum ? ggml_add(ctx_g, head_sum, z) : z;
          auto *res = conv(ctx_w, ctx_g, z, l.layer1x1, m, loads);
          x = ggml_add(ctx_g, x, res);
        }
        head_sum =
            conv(ctx_w, ctx_g, with_cache(head_sum, a.head), a.head, m, loads);
      }
    } else {
      for (const auto &a : m.arrays) {
        x = conv(ctx_w, ctx_g, x, a.rechannel, m, loads);
        int out_no_head = int(std::min(x->ne[0], condition->ne[0]) -
                              (a.receptive - a.head.kernel));
        for (const auto &l : a.layers) {
          auto *z = conv(ctx_w, ctx_g, x, l.conv, m, loads);
          auto *mix =
              right(ctx_g, conv(ctx_w, ctx_g, condition, l.mixer, m, loads),
                    z->ne[0]);
          z = ggml_add(ctx_g, z, mix);
          z = (l.activation == "ReLU") ? ggml_relu(ctx_g, z)
                                       : ggml_tanh(ctx_g, z);
          auto *term = right(ctx_g, z, out_no_head);
          head_sum =
              head_sum
                  ? ggml_add(ctx_g, right(ctx_g, head_sum, out_no_head), term)
                  : term;
          auto *res = conv(ctx_w, ctx_g, z, l.layer1x1, m, loads);
          x = ggml_add(ctx_g, right(ctx_g, x, res->ne[0]), res);
        }
        head_sum = conv(ctx_w, ctx_g, head_sum, a.head, m, loads);
        x = right(ctx_g, x, head_sum->ne[0]);
      }
    }
    output = ggml_scale(ctx_g, head_sum, m.head_scale);
    ggml_set_output(output);
    graph = ggml_new_graph_custom(ctx_g, 4096, false);
    ggml_build_forward_expand(graph, output);
    for (const auto &c : caches) {
      ggml_build_forward_expand(graph, c.out);
    }
    weight_buffer = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!weight_buffer) {
      std::fprintf(stderr, "Failed to allocate the ggml weight buffer\n");
      std::abort();
    }
    for (const auto &p : loads) {
      ggml_backend_tensor_set(p.first, p.second, 0, ggml_nbytes(p.first));
    }
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, graph)) {
      std::fprintf(stderr, "Failed to allocate the ggml compute graph\n");
      std::abort();
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
    for (const auto &c : caches) {
      ggml_backend_tensor_set(c.in, c.host.data(), 0, ggml_nbytes(c.in));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "ggml graph computation failed\n");
      std::abort();
    }
    ggml_backend_tensor_get(output, out, 0, output_size * sizeof(float));
    for (auto &c : caches) {
      ggml_backend_tensor_get(c.out, c.host.data(), 0, ggml_nbytes(c.out));
    }
  }

  ~GraphSession() {
    if (galloc) {
      ggml_gallocr_free(galloc);
    }
    if (weight_buffer) {
      ggml_backend_buffer_free(weight_buffer);
    }
    if (ctx_w) {
      ggml_free(ctx_w);
    }
    if (ctx_g) {
      ggml_free(ctx_g);
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
  GraphSession session(m, padded.size(), threads, false);
  std::vector<float> out(raw.size());
  session.compute(padded.data(), padded.size(), out.data(), out.size());
  return out;
}

std::vector<float> infer_streaming(const Model &m,
                                   const std::vector<float> &raw, int threads,
                                   size_t block_size) {
  GraphSession session(m, block_size, threads, true);
  std::vector<float> block(block_size, 0.0f);
  std::vector<float> out(raw.size());

  // Deep activations of an all-zero history are nonzero (biases), so warm the
  // caches by streaming receptive-1 zeros; this matches the offline padding.
  std::vector<float> discard(block_size);
  for (size_t warmed = 0; warmed < size_t(m.receptive - 1);
       warmed += block_size) {
    session.compute(block.data(), block_size, discard.data(), block_size);
  }

  for (size_t offset = 0; offset < raw.size(); offset += block_size) {
    const size_t n = std::min(block_size, raw.size() - offset);
    std::copy_n(raw.data() + offset, n, block.begin());
    std::fill(block.begin() + n, block.end(), 0.0f);
    session.compute(block.data(), block_size, out.data() + offset, n);
  }
  return out;
}
