#include "model.h"

#include <json-c/json.h>

#include <cstdio>
#include <cstdlib>

static json_object *get(json_object *o, const char *key) {
  json_object *v = nullptr;
  if (!json_object_object_get_ex(o, key, &v)) {
    std::fprintf(stderr, "Missing field: %s\n", key);
    std::abort();
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
  std::fprintf(stderr, "Only string activation values are supported\n");
  std::abort();
}

static void take(Conv &c, size_t &p) {
  c.offset = p;
  p += size_t(c.in) * c.out * c.kernel + (c.bias ? c.out : 0);
}

Model load_model(const std::string &path) {
  json_object *root = json_object_from_file(path.c_str());
  if (!root) {
    std::fprintf(stderr, "Failed to parse NAM JSON: %s\n", path.c_str());
    std::abort();
  }

  struct Guard {
    json_object *p;

    ~Guard() { json_object_put(p); }
  } guard{root};

  if (js(get(root, "architecture")) != "WaveNet") {
    std::fprintf(stderr, "Only the A1 WaveNet architecture is supported\n");
    std::abort();
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
      std::fprintf(stderr,
                   "Only mono conditioning and groups=1 are supported\n");
      std::abort();
    }
    json_object *act = get(a, "activation");
    if (json_object_is_type(act, json_type_array)) {
      std::fprintf(stderr, "Per-layer activation lists are not supported\n");
      std::abort();
    }
    std::string activation = js(act);
    if (activation != "ReLU" && activation != "Tanh") {
      std::fprintf(stderr, "Only ReLU and Tanh activations are supported\n");
      std::abort();
    }
    if (jb(a, "gated", false)) {
      std::fprintf(stderr, "Gated activations are not supported\n");
      std::abort();
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
    std::fprintf(stderr,
                 "Weight count does not match the classic A1 layout "
                 "(expected %zu, got %zu)\n",
                 p + 1, m.weights.size());
    std::abort();
  }
  m.head_scale = m.weights[p];
  return m;
}
