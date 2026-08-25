#if defined(__APPLE__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct PvBnnsGraph {
  bnns_graph_t g;
  int mmapped;
};

struct PvBnnsGraphCtx {
  bnns_graph_context_t ctx;
  int last_t;
  void *workspace;
  size_t workspace_cap;
};

static int compiled_path(const char *src, char *dst, size_t dst_n) {
  if (src == NULL || dst == NULL || dst_n < 16) {
    return -1;
  }
  int n = snprintf(dst, dst_n, "%s.bnnsgraph", src);
  return (n > 0 && (size_t)n < dst_n) ? 0 : -1;
}

static int try_mmap(const char *path, bnns_graph_t *g) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return -1;
  }
  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    return -1;
  }
  g->data = p;
  g->size = (size_t)st.st_size;
  return 0;
}

void *pv_bnns_graph_compile(const char *path) {
  if (path == NULL) {
    return NULL;
  }
  char out_path[4096];
  if (compiled_path(path, out_path, sizeof(out_path)) != 0) {
    return NULL;
  }
  bnns_graph_t g = {0};
  int mmapped = 0;
  if (try_mmap(out_path, &g) == 0) {
    mmapped = 1;
  } else {
    bnns_graph_compile_options_t opt = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetTargetSingleThread(opt, false);
    BNNSGraphCompileOptionsSetOptimizationPreference(
        opt, BNNSGraphOptimizationPreferencePerformance);
    BNNSGraphCompileOptionsSetOutputPath(opt, out_path);
    g = BNNSGraphCompileFromFile(path, NULL, opt);
    BNNSGraphCompileOptionsDestroy(opt);
    if (g.data == NULL) {
      return NULL;
    }
  }
  struct PvBnnsGraph *out = (struct PvBnnsGraph *)calloc(1, sizeof(*out));
  if (out == NULL) {
    return NULL;
  }
  out->g = g;
  out->mmapped = mmapped;
  return out;
}

void pv_bnns_graph_free(void *raw) {
  struct PvBnnsGraph *g = (struct PvBnnsGraph *)raw;
  if (g == NULL) {
    return;
  }
  if (g->mmapped && g->g.data != NULL && g->g.size > 0) {
    munmap(g->g.data, g->g.size);
  }
  free(g);
}

void *pv_bnns_graph_context(void *raw) {
  struct PvBnnsGraph *g = (struct PvBnnsGraph *)raw;
  if (g == NULL) {
    return NULL;
  }
  bnns_graph_context_t ctx = BNNSGraphContextMake(g->g);
  if (ctx.data == NULL) {
    return NULL;
  }
  BNNSGraphContextSetArgumentType(ctx, BNNSGraphArgumentTypePointer);
  struct PvBnnsGraphCtx *out = (struct PvBnnsGraphCtx *)calloc(1, sizeof(*out));
  if (out == NULL) {
    BNNSGraphContextDestroy(ctx);
    return NULL;
  }
  out->ctx = ctx;
  out->last_t = -1;
  return out;
}

void pv_bnns_graph_context_free(void *raw) {
  struct PvBnnsGraphCtx *c = (struct PvBnnsGraphCtx *)raw;
  if (c == NULL) {
    return;
  }
  BNNSGraphContextDestroy(c->ctx);
  free(c->workspace);
  free(c);
}

int pv_bnns_graph_exec(
    void *raw,
    int t,
    const float *in,
    float *out,
    int *out_w) {
  struct PvBnnsGraphCtx *c = (struct PvBnnsGraphCtx *)raw;
  if (c == NULL || t < 8 || in == NULL || out == NULL) {
    return -1;
  }
  uint64_t in_shape_v[4] = {1, 1, 80, (uint64_t)t};
  uint64_t out_shape_v[4] = {0, 0, 0, 0};
  bnns_graph_shape_t shapes[2];
  shapes[0].rank = 4;
  shapes[0].shape = out_shape_v;
  shapes[1].rank = 4;
  shapes[1].shape = in_shape_v;
  if (c->last_t != t) {
    int rc = BNNSGraphContextSetDynamicShapes(c->ctx, NULL, 2, shapes);
    if (rc < 0) {
      return rc;
    }
    c->last_t = t;
    size_t need = BNNSGraphContextGetWorkspaceSize(c->ctx, NULL);
    if (need > 0 && need != (size_t)-1 && need > c->workspace_cap) {
      /* page-aligned, as required by BNNSGraphContextExecute */
      free(c->workspace);
      c->workspace = NULL;
      c->workspace_cap = 0;
      void *p = NULL;
      if (posix_memalign(&p, 16384, need) == 0) {
        c->workspace = p;
        c->workspace_cap = need;
      }
    }
  }
  int oc = (int)out_shape_v[1];
  int oh = (int)out_shape_v[2];
  int ow = (int)out_shape_v[3];
  if (oc <= 0 || oh <= 0 || ow <= 0) {
    int w = t;
    for (int i = 0; i < 3; i++) {
      w = (w + 2 - 3) / 2 + 1;
    }
    oc = 256;
    oh = 10;
    ow = w;
  }
  if (out_w != NULL) {
    *out_w = ow;
  }
  bnns_graph_argument_t args[2];
  memset(args, 0, sizeof(args));
  args[0].data_ptr = (void *)(uintptr_t)out;
  args[0].data_ptr_size = (size_t)oc * (size_t)oh * (size_t)ow * sizeof(float);
  args[1].data_ptr = (void *)(uintptr_t)in;
  args[1].data_ptr_size = (size_t)80 * (size_t)t * sizeof(float);
  return BNNSGraphContextExecute(
      c->ctx, NULL, 2, args, c->workspace_cap, (char *)c->workspace);
}

#pragma clang diagnostic pop
#endif
