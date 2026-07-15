#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

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

Wav read_wav(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  char id[4];
  f.read(id, 4);
  (void)u32(f);
  char wave[4];
  f.read(wave, 4);
  if (!f || std::memcmp(id, "RIFF", 4) || std::memcmp(wave, "WAVE", 4)) {
    std::fprintf(stderr, "Input is not a RIFF/WAVE file\n");
    std::abort();
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
    std::fprintf(stderr, "Input must be a mono IEEE float32 WAV file\n");
    std::abort();
  }

  Wav w{rate, std::vector<float>(data.size() / 4)};
  std::memcpy(w.samples.data(), data.data(), data.size());
  return w;
}

void Wav::write(const std::string &path) const {
  std::ofstream f(path, std::ios::binary);
  uint32_t bytes = uint32_t(samples.size() * 4), riff = 36 + bytes;
  uint16_t one = 1, fmt = 3, bits = 32, align = 4;
  uint32_t br = rate * 4, fs = 16;
  f.write("RIFF", 4);
  f.write((char *)&riff, 4);
  f.write("WAVEfmt ", 8);
  f.write((char *)&fs, 4);
  f.write((char *)&fmt, 2);
  f.write((char *)&one, 2);
  f.write((char *)&rate, 4);
  f.write((char *)&br, 4);
  f.write((char *)&align, 2);
  f.write((char *)&bits, 2);
  f.write("data", 4);
  f.write((char *)&bytes, 4);
  f.write((char *)samples.data(), bytes);
}
