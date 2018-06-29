#ifdef __cplusplus
extern "C" {
#endif

void load_model(char *path);
void *fasttext_new();
void fasttext_delete(void *ft);
void fasttext_load_model(void *ft, char *path);
int fasttext_predict(void *ft, char *query, float *prob, char **buf, int *size);
int fasttext_predict_k(void *ft, char *query, int k, float *prob, char **buf, int *sizes);

#ifdef __cplusplus
}
#endif