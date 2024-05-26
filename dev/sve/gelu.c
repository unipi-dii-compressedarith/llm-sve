#include <math.h>
#include <arm_sve.h>
#include "gpt2_sve.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *restrict out, float *restrict inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for(uint32_t i = 0; i < N; ++i)
    {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = GELU_SCALING_FACTOR * (x + cube);
    }

    tanh_array(out, N);

    for(uint32_t i = 0; i < N; ++i)
        out[i] = 0.5f * inp[i] * (1.0f + out[i]);
}


void gelu_backward(float *restrict dinp, float *restrict inp, float *restrict dout, int N) {
    // !!! STACK KILLER !!! (it works)
    float tanh_out[N];
    float coshf_out[N];
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        tanh_out[i] = GELU_SCALING_FACTOR * (x + cube);
        coshf_out[i] = tanh_out[i];
    }

    tanh_array(tanh_out, N);
    expf_array(coshf_out, N);

    for (int i = 0; i < N; i++)
        coshf_out[i] = (coshf_out[i] + 1.0f/coshf_out[i])*0.5f;

    for (int i = 0; i < N; i++)
    {
        float x = inp[i];
        float sech_out = 1.0f / (coshf_out[i] * coshf_out[i]);
        float local_grad = 0.5f * (1.0f + tanh_out[i]) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }

}
