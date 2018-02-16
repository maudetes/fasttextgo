#include <iostream>
#include <istream>
#include "fasttext.h"
#include "real.h"
#include <streambuf>

typedef struct {
  float prob;
  char* label;
} tag;

extern "C" {

struct membuf : std::streambuf {
  membuf(char* begin, char* end) {
    this->setg(begin, begin, end);
  }
};

fasttext::FastText g_fasttext_model;

bool g_fasttext_initialized = false;

void load_model(char *path) {
  if (!g_fasttext_initialized) {
    g_fasttext_model.loadModel(std::string(path));
    g_fasttext_initialized = true;
  }
}

void predict(char *query, int k, float* prob, char** bufs, int buf_sz) {

  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  g_fasttext_model.predict(in, k, predictions);

  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
    *prob++ = (float)it->first;
    if (it->second.length() > buf_sz) {
      strncpy(*bufs, it->second.c_str(), buf_sz-1);
    } else {
      strncpy(*bufs, it->second.c_str(), it->second.length());
    }
    bufs++;
  }

}

}
