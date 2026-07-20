#include "inference.h"
#include "model.h"
#include "wav.h"

#include <cargs.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

const cag_option options[] = {
    {'t', "t", "threads", "N", "number of CPU threads (default: 1)"},
    {'b', "b", "block", "N",
     "process with a reusable streaming graph of N samples per block; "
     "0 keeps whole-file offline inference (default: 0)"},
    {'h', "h", "help", nullptr, "show this help and exit"},
};

void print_usage(std::FILE *out) {
  std::fprintf(out,
               "Usage: nam [OPTIONS] MODEL INPUT OUTPUT\n"
               "\n"
               "Experimental offline NAM inference using ggml.\n"
               "\n"
               "Arguments:\n"
               "  MODEL   NAM model file\n"
               "  INPUT   mono float32 WAV input\n"
               "  OUTPUT  mono float32 WAV output\n"
               "\n"
               "Options:\n");
  cag_option_print(options, CAG_ARRAY_SIZE(options), out);
}

bool parse_int(const char *s, int min, int &value) {
  if (s == nullptr)
    return false;
  errno = 0;
  char *end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE || v < min || v > INT_MAX)
    return false;
  value = int(v);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int threads = 1;
  int block = 0;

  cag_option_context context;
  cag_option_init(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
  while (cag_option_fetch(&context)) {
    switch (cag_option_get_identifier(&context)) {
    case 't':
      if (!parse_int(cag_option_get_value(&context), 1, threads)) {
        std::fprintf(stderr, "nam: thread count must be a positive integer\n");
        return 1;
      }
      break;
    case 'b':
      if (!parse_int(cag_option_get_value(&context), 0, block)) {
        std::fprintf(stderr, "nam: block size must be a non-negative integer\n");
        return 1;
      }
      break;
    case 'h':
      print_usage(stdout);
      return 0;
    default:
      cag_option_print_error(&context, stderr);
      print_usage(stderr);
      return 1;
    }
  }

  int npos = argc - cag_option_get_index(&context);
  if (npos != 3) {
    std::fprintf(stderr, "nam: expected MODEL INPUT OUTPUT, got %d argument(s)\n",
                 npos);
    print_usage(stderr);
    return 1;
  }
  char **pos = argv + cag_option_get_index(&context);

  Model m = load_model(pos[0]);
  Wav w = read_wav(pos[1]);
  w.samples = block == 0 ? infer(m, w.samples, threads)
                         : infer_streaming(m, w.samples, threads, size_t(block));
  w.write(pos[2]);
  std::cout << "Done: " << w.samples.size() << " samples, receptive field "
            << m.receptive << "\n";
}
