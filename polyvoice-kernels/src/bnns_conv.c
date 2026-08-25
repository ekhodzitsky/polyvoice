#if defined(__APPLE__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Accelerate/Accelerate.h>
#include <stddef.h>
#include <stdint.h>

void *pv_bnns_conv_create(
    int ic,
    int ih,
    int iw,
    int oc,
    int k,
    int stride,
    int pad,
    int relu,
    int n_threads,
    const float *weight,
    const float *bias) {
  int oh = (ih + 2 * pad - k) / stride + 1;
  int ow = (iw + 2 * pad - k) / stride + 1;
  if (oh <= 0 || ow <= 0 || ic <= 0 || oc <= 0 || k <= 0 || stride <= 0) {
    return NULL;
  }

  BNNSLayerParametersConvolution p = {0};
  p.i_desc.layout = BNNSDataLayoutImageCHW;
  p.i_desc.size[0] = (size_t)iw;
  p.i_desc.size[1] = (size_t)ih;
  p.i_desc.size[2] = (size_t)ic;
  p.i_desc.data_type = BNNSDataTypeFloat32;

  p.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
  p.w_desc.size[0] = (size_t)k;
  p.w_desc.size[1] = (size_t)k;
  p.w_desc.size[2] = (size_t)ic;
  p.w_desc.size[3] = (size_t)oc;
  p.w_desc.data = (void *)(uintptr_t)weight;
  p.w_desc.data_type = BNNSDataTypeFloat32;

  p.o_desc.layout = BNNSDataLayoutImageCHW;
  p.o_desc.size[0] = (size_t)ow;
  p.o_desc.size[1] = (size_t)oh;
  p.o_desc.size[2] = (size_t)oc;
  p.o_desc.data_type = BNNSDataTypeFloat32;

  p.bias.layout = BNNSDataLayoutVector;
  p.bias.size[0] = (size_t)oc;
  p.bias.data = (void *)(uintptr_t)bias;
  p.bias.data_type = BNNSDataTypeFloat32;

  p.activation.function = relu ? BNNSActivationFunctionRectifiedLinear
                               : BNNSActivationFunctionIdentity;
  p.x_stride = (size_t)stride;
  p.y_stride = (size_t)stride;
  p.x_padding = (size_t)pad;
  p.y_padding = (size_t)pad;

  /* Copy+pack weights (no UseClientPtr) so BNNS can retile for Winograd/AMX.
     n_threads==0 lets BNNS pick the machine default (typically 2–4 on Apple
     Silicon); 1 is 1.5–2× slower on every ResNet spatial size we measured. */
  BNNSFilterParameters fp = {0};
  fp.n_threads = n_threads < 0 ? 0 : (size_t)n_threads;
  return BNNSFilterCreateLayerConvolution(&p, &fp);
}

int pv_bnns_conv_apply(void *filter, const float *in, float *out) {
  if (filter == NULL) {
    return -1;
  }
  return BNNSFilterApply(filter, in, out);
}

int pv_bnns_conv_apply_n(
    void *filter,
    size_t n,
    const float *in,
    size_t in_stride,
    float *out,
    size_t out_stride) {
  if (filter == NULL) {
    return -1;
  }
  return BNNSFilterApplyBatch(filter, n, in, in_stride, out, out_stride);
}

void pv_bnns_conv_destroy(void *filter) {
  if (filter != NULL) {
    BNNSFilterDestroy(filter);
  }
}
#pragma clang diagnostic pop
#endif
