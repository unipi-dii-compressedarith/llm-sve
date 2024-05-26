#include <arm_sve.h>
#include <math.h>
#include "gpt2_sve.h"

// 11 bit precision
#define ENTRIES 0x800

float log2_mantissa_table[ENTRIES] = {0};

const uint32_t MASK_EXPONENT = 0b01111111100000000000000000000000;
const uint32_t MASK_MANTISSA = 0b00000000011111111111111111111111;

__attribute__ ((constructor)) static void init_log2_LUT()
{
    // INIT TABLE
    for (int i = 0; i < ENTRIES; ++i)
    {
        float x = 1.0f + (float)i / ENTRIES;
        log2_mantissa_table[i] = log2f(x);
    }
}

void log2_sw_LUT(float *values, uint32_t len, float factor)
{
    // Cast
    uint32_t *values_c = (uint32_t *) values;

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svuint32_t v = svld1(pg, values_c + i);

        svuint32_t exponent_v = svand_x(pg, v, MASK_EXPONENT);
        exponent_v = svlsr_x(pg, exponent_v, 23);
        svint32_t log_int_v = svsub_x(pg, svreinterpret_s32(exponent_v), 127);

        svuint32_t mantissa_indices_v = svand_x(pg, v, MASK_MANTISSA);
        mantissa_indices_v = svlsr_x(pg, mantissa_indices_v, 23-11);

        svfloat32_t mantissa_v = svld1_gather_index(pg, log2_mantissa_table, mantissa_indices_v);
        svfloat32_t expontent_float_v = svcvt_f32_x(pg, log_int_v);

        svfloat32_t sum = svadd_x(pg, expontent_float_v, mantissa_v);
        sum = svmul_x(pg, sum, factor);
        svst1(pg, values + i, sum);
    }
}

void log2_w_trick(float *values, uint32_t len, float factor)
{
    // Cast
    int32_t *values_c = (int32_t *) values;

    svfloat32_t c1 = svdup_f32(-1.0f / 3);

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svint32_t vsrc = svld1(pg, values_c + i);

        // bitshift destro di 23 e maschera e sottrai 128
        svint32_t vlog2 = svasr_x(pg, vsrc, 23);      // log_2 = x >> 23
        vlog2 = svand_x(pg, vlog2, 255);     // vlog2 &= 255
        vlog2 = svsub_x(pg, vlog2, 128);     // vlog2 -= 128

        // convert to floating point
        svfloat32_t vlog2f = svcvt_f32_x(pg, vlog2);

        // seconda maschera e somma
        vsrc = svand_x(pg, vsrc, ~(255 << 23));    // x &= ~(255 << 23)
        vsrc = svadd_x(pg, vsrc, 127 << 23);       // x += 127 << 23

        // Now work with floats
        svfloat32_t vval = svreinterpret_f32(vsrc); //svld1(pg, values + i);

        svfloat32_t t1 = svmad_x(pg, vval, c1, 2.0f);      // t1 = val * (-1.0f/3) + 2
        vval = svmad_x(pg, vval, t1, -2.0f / 3);    // val = val*t1 - 2/3

        vval = svadd_x(pg, vval, vlog2f);   // val += log
        vval = svmul_x(pg, vval, factor);  // val *= factor

        // Store back
        svst1(pg, values + i, vval);
    }
}

void log2_libc(float *data, uint32_t len, float factor)
{
    for(uint32_t i = 0; i < len; ++i)
        data[i] = log2f(data[i]) * factor;
}
