#include <arm_sve.h>
#include <math.h>
#include "gpt2_sve.h"

void encoder_forward(float *restrict out,
                     int *restrict inp, float *restrict wte, float *restrict wpe,
                     int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float *restrict dwte, float *restrict dwpe,
                      const float *restrict dout, const int *restrict inp,
                      int B, int T, int C) {
#pragma clang loop vectorize(assume_safety)
    for(int c = 0; c < C; ++c)
    {
        int BT = B * T;
        for (int i = 0; i < BT; i++) {
            int t = i % T;
            int ix = inp[i];
            float dout_btc = dout[i * C + c];
            dwte[ix * C + c] += dout_btc;
            dwpe[t * C + c] += dout_btc;
        }
    }
}
