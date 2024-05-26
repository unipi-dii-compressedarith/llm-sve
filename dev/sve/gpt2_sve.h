#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <arm_sve.h>
#ifdef OMP
#include <omp.h>
#endif

void expf_with_sw_LUT(float *data, uint32_t len);
void expf_with_hw_LUT(float *data, uint32_t len);
void expf_libc(float *data, uint32_t len);
#ifndef expf_array
#define expf_array expf_with_hw_LUT
#endif

svfloat32_t sv_exp_hw_lut (svfloat32_t x, const svbool_t pg);
svfloat32_t sv_exp_lut(svfloat32_t v, const svbool_t pg);

#ifndef sv_expf
#define sv_expf sv_exp_hw_lut
#endif


void log2_sw_LUT(float *values, uint32_t len, float factor);
void log2_w_trick(float *values, uint32_t len, float factor);
void log2_libc(float *data, uint32_t len, float factor);

#ifndef log2_array
#define log2_array log2_sw_LUT
#endif

void tanh_approx(float *values, uint32_t len);
void tanh_libc(float *data, uint32_t len);

#ifndef tanh_array
#define tanh_array tanh_approx
#endif

void attention_forward(float *restrict out, float *restrict preatt, float *restrict att,
                       float *restrict inp,
                       int B, int T, int C, int NH);

void attention_backward(float *restrict dinp, float *restrict dpreatt,
                        float *restrict datt, float *restrict dout,
                        float *restrict inp, float *restrict att,
                        int B, int T, int C, int NH);

void encoder_forward(float *restrict out,
                     int *restrict inp, float *restrict wte, float *restrict wpe,
                     int B, int T, int C);

void encoder_backward(float *restrict dwte, float *restrict dwpe,
                      const float *restrict dout, const int *restrict inp,
                      int B, int T, int C);

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C);

void layernorm_backward(
    float *restrict dinp, float *restrict dweight, float *restrict dbias,
    float *restrict dout, float *restrict inp, float *restrict weight,
    float *restrict mean, float *restrict rstd,
    int B, int T, int C);

void matmul_forward(float *restrict out,
                    float *restrict inp, float *restrict weight, const float *restrict bias,
                    int B, int T, int C, int OC);

void matmul_backward(float *restrict dinp, float *restrict dweight, float *restrict dbias,
                     float *restrict dout, float *restrict inp, float *restrict weight,
                     int B, int T, int C, int OC);

void residual_forward(float *restrict out, float *restrict inp1, float *restrict inp2, int N);

void residual_backward(float *restrict dinp1, float *restrict dinp2, float *restrict dout, int N);

void softmax_forward(float *restrict probs, float *restrict logits, int B, int T, int V, int Vp);

void crossentropy_forward(float *restrict losses,
                          float *restrict probs, int *restrict targets,
                          int B, int T, int Vp);

void crossentropy_softmax_backward(float *restrict dlogits,
                                   float *restrict dlosses, float *restrict probs,
                                   int *restrict targets,
                                   int B, int T, int V, int Vp);

void gelu_forward(float *restrict out, float *restrict inp, int N);
void gelu_backward(float *restrict dinp, float *restrict inp, float *restrict dout, int N);

#include "gpt2_model.h"
