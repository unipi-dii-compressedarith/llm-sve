#include <math.h>
#include <arm_sve.h>
#include "gpt2_sve.h"

#define ENABLE_ZERO_APROX

#define ENTRIES (16 + 4096)
float exp_table[ENTRIES] = {0};

__attribute__ ((constructor)) static void build_exp_LUT()
{
    // INIT LUT
    for (int i = 0; i < ENTRIES; ++i)
    {
        int n = i - 16;
        exp_table[i] = expf(n);
    }
}

inline svfloat32_t sv_exp_lut(svfloat32_t v, const svbool_t pg)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);

    // Get int and frac part
    //svfloat32_t int_v = svrintm_x(pg, v);
    svfloat32_t int_v = svrinta_x(pg, v);
    svfloat32_t mod_v = svsub_x(pg, v, int_v);

    // get indices
    svint32_t indices = svcvt_s32_x(pg, int_v);
    indices = svadd_x(pg, indices, 16);

    // Load integer approximations
    svfloat32_t exp_int_part_v = svld1_gather_index(pg, exp_table, indices);

    // Compute fractional approxiamtion
    // exp(eps) ~= 1 + eps*(1 + eps*(0.5 + eps*1/6))
    svfloat32_t part_taylor_v = svmla_x(pg, c1, mod_v, 1.0f / 6.0f);
    part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);
    part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);

    // compute exp(eps)*exp(n)
    return svmul_x(pg, part_taylor_v, exp_int_part_v);
}

void expf_with_sw_LUT(float *values, uint32_t len)
{
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);

#ifdef ENABLE_ZERO_APROX // @FIXME
        // exp(p) = 0 if p <= 1e-16
        // disabling this will cause a SEGFAULT for out-of-bound ns
        //svbool_t zero_mask = svcmpgt(pg, v, 16.0f);
        //pg = svnot_z(pg, zero_mask); // i.e. p st p > 1e-16
        svbool_t pg_tot = pg;
        pg = svcmple(pg, v, 16.0f);
#endif
        v = sv_exp_lut(v, pg);

#ifdef ENABLE_ZERO_APROX
        //v = svdup_f32_m(v, zero_mask, /*0.00000005f*/ 0.0f);
        v = svmax_x(pg_tot, v, 0.0f); // clamp negative values, faster that setting 0 manually

        // Store back
        svst1(pg_tot, values + i, v);
#else
        // Store back
        svst1(pg, values + i, v);
#endif
    }
}

struct sv_expf_data
{
    float poly[5];
    float inv_ln2, ln2_hi, ln2_lo, shift;
};

/* Coefficients copied from the polynomial in AdvSIMD variant, reversed for
   compatibility with polynomial helpers. Shift is 1.5*2^17 + 127.  */
#define SV_EXPF_DATA                                                          \
  {                                                                           \
    .poly = { 0x1.ffffecp-1f, 0x1.fffdb6p-2f, 0x1.555e66p-3f, 0x1.573e2ep-5f, \
	      0x1.0e4020p-7f },                                               \
                                                                              \
    .inv_ln2 = 0x1.715476p+0f, .ln2_hi = 0x1.62e4p-1f,                        \
    .ln2_lo = 0x1.7f7d1cp-20f, .shift = 0x1.803f8p17f,                        \
  }

#define C(i) svdup_n_f32(d.poly[i])

// Copyright (C) 2024 Free Software Foundation, Inc.
inline svfloat32_t
sv_exp_hw_lut (svfloat32_t x, const svbool_t pg /*, const struct sv_expf_data *d*/)
{
    static const struct sv_expf_data d = SV_EXPF_DATA;
    /* exp(x) = 2^n (1 + poly(r)), with 1 + poly(r) in [1/sqrt(2),sqrt(2)]
       x = ln2*n + r, with r in [-ln2/2, ln2/2].  */

    /* Load some constants in quad-word chunks to minimise memory access.  */
    svfloat32_t c4_invln2_and_ln2 = svld1rq (svptrue_b32 (), &d.poly[4]);

    /* n = round(x/(ln2/N)).  */
    svfloat32_t z = svmla_lane (svdup_n_f32 (d.shift), x, c4_invln2_and_ln2, 1);
    svfloat32_t n = svsub_x (pg, z, d.shift);

    /* r = x - n*ln2/N.  */
    svfloat32_t r = svmls_lane (x, n, c4_invln2_and_ln2, 2);
    r = svmls_lane (r, n, c4_invln2_and_ln2, 3);

    /* scale = 2^(n/N).  */
    svfloat32_t scale = svexpa (svreinterpret_u32_f32 (z));

    /* y = exp(r) - 1 ~= r + C0 r^2 + C1 r^3 + C2 r^4 + C3 r^5 + C4 r^6.  */
    svfloat32_t p12 = svmla_x (pg, C (1), C (2), r);
    svfloat32_t p34 = svmla_lane (C (3), r, c4_invln2_and_ln2, 0);
    svfloat32_t r2 = svmul_f32_x (pg, r, r);
    svfloat32_t p14 = svmla_x (pg, p12, p34, r2);
    svfloat32_t p0 = svmul_f32_x (pg, r, C (0));
    svfloat32_t poly = svmla_x (pg, p0, r2, p14);

    return svmla_x (pg, scale, scale, poly);
}

void expf_with_hw_LUT(float *data, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, (uint32_t) len);
        svfloat32_t v = svld1(pg, data + i);

        v = sv_exp_hw_lut(v, pg);

        // Store back
        svst1(pg, data + i, v);
    }
}

void expf_libc(float *data, uint32_t len)
{
    for(uint32_t i = 0; i < len; ++i)
        data[i] = expf(data[i]);
}
