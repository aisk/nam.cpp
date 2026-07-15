#include "inference.h"
#include "model.h"
#include "wav.h"

#include <argparse/argparse.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
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
    std::fprintf(stderr, "Thread count must be at least 1\n");
    std::abort();
  }
  int stream_block = program.get<int>("--stream-block");
  if (stream_block < 0) {
    std::fprintf(stderr, "Streaming block size must not be negative\n");
    std::abort();
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
}
