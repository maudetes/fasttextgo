#include <iostream>
#include <istream>
#include "fasttext.h"
#include "real.h"
#include <streambuf>
#include <string.h>
#include "args.h"
#include "fasttext_wrapper.h"

extern "C" {

struct membuf : std::streambuf
{
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

int fasttext_predict(void *ft_, char *query, float *prob, char **buf, int *size) {
  fasttext::FastText *ft = (fasttext::FastText*) ft_;
  if (ft == NULL) {
    ft = &g_fasttext_model;
  }

  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  ft->predict(in, 1, predictions);

  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
    *prob = (float)it->first;
    *size = it->second.size();
    *buf = (char*)malloc(*size);
    if (*buf == NULL) {
      return 1;
    }
    memcpy(*buf, it->second.c_str(), *size);
    return 0;
  }
  return 1;
}

int fasttext_predict_k(void *ft_, char *query, int k, float *prob, char **buf, int *sizes) {
  fasttext::FastText *ft = (fasttext::FastText*) ft_;
  if (ft == NULL) {
    ft = &g_fasttext_model;
  }

  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;

  ft->predict(in, k, predictions);

  std::vector<std::string> labels;
  size_t size = 0;
  int n = 0;
  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
    prob[n] = (float)it->first;
    sizes[n++] = it->second.size();
    size += it->second.size();
    labels.push_back(it->second);
  }
  *buf = (char *)malloc(size);
  if (*buf == NULL) {
    return -1;
  }
  char *p = *buf;
  for (int i = 0; i < n; i++) {
    memcpy(p, labels[i].c_str(), sizes[i]);
    p += sizes[i];
  }
  return n;
}

void *fasttext_new() {
  return new fasttext::FastText();
}

void fasttext_delete(void *p) {
  delete ((fasttext::FastText*) p);
}

void fasttext_load_model(void *p, char *path) {
  fasttext::FastText *ft = (fasttext::FastText*) p;
  ft->loadModel(std::string(path));
}

}