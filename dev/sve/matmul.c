#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <memory.h>
#include "gpt2_sve.h"

void matmul_forward(float *restrict out,
                    float *restrict inp, float *restrict weight, const float *restrict bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;

            if(bias != NULL)
                memcpy(out_bt, bias, OC*sizeof(float));
            else
                memset(out_bt, 0x00, OC*sizeof(float));

            for (int o = 0; o < OC; o++) {
                float val = 0;
                float* wrow = weight + o*C;

                // dot product inp_bt * wrow_o
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                // out = inp_bt * wrow_o + BIAS
                out_bt[o] += val;
            }
        }
    }
}

void matmul_backward(float *restrict dinp, float *restrict dweight, float *restrict dbias,
                     float *restrict dout, float *restrict inp, float *restrict weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    // backward into weight/bias, parallelize over output channels OC
#pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}
