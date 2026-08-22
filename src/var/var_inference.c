#include <pthread.h>

pthread_mutex_t inference_mutex = PTHREAD_MUTEX_INITIALIZER;
bool inference_mode = true;

void var_set_global_inference(bool infer) {
  pthread_mutex_lock(&inference_mutex);
  inference_mode = infer;
  pthread_mutex_unlock(&inference_mutex);
}

bool var_get_global_inference() {
  pthread_mutex_lock(&inference_mutex);
  bool infer = inference_mode;
  pthread_mutex_unlock(&inference_mutex);
  return infer;
}

bool thread_inference_mode = true;

bool *thread_inference_mode_location() { return &thread_inference_mode; }
