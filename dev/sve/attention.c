#include <math.h>
#include <arm_sve.h>
#include <stddef.h>
#include <memory.h>
#include "gpt2_sve.h"

void attention_forward(float *restrict out, float *restrict preatt, float *restrict att,
                       float *restrict inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                    att_bth[t2] = preatt_bth[t2] - maxval;

                expf_array(att_bth, t + 1);

                // autovectorized
                for (int t2 = 0; t2 <= t; t2++)
                    expsum += att_bth[t2];

                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (uint32_t t2 = 0; t2 < T; t2 += svcntw()) {
                    svbool_t pg_loop = svwhilelt_b32(t2, (uint32_t)T);
                    svbool_t pg_condition = svwhilele_b32(t2, (uint32_t)t);
                    svbool_t pg = svand_z(svptrue_b32(), pg_loop, pg_condition);

                    svfloat32_t att_bthv = svld1(pg_loop, att_bth + t2);

                    // Notice we set to zero the elements outside the mask, ie t2: t2 > t
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bthv = svmul_z(pg, att_bthv, expsum_inv);

                    svst1(pg_loop, att_bth + t2, att_bthv);
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward(float *restrict dinp, float *restrict dpreatt,
                        float *restrict datt, float *restrict dout,
                        float *restrict inp, float *restrict att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}