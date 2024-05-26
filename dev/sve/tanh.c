#include <arm_sve.h>
#include <math.h>
#include "gpt2_sve.h"

#define ENABLE_ONE_APROX

void tanh_approx(float *values, uint32_t len)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);
        v = svscale_x(pg, v, 1); // 2x

#ifdef ENABLE_ONE_APROX // @FIXME @TODO
        // exp(p) = 0 if p <= 1e-16
        // disabling this will cause a SEGFAULT for out-of-bound ns
        //svbool_t zero_mask = svcmpgt(pg, v, 16.0f);
        //pg = svnot_z(pg, zero_mask); // i.e. p st p > 1e-16
        svbool_t pg_tot = pg;
        svbool_t pg_above_min = svcmple(pg, v, 10.0f);
        svbool_t pg_below_max = svcmpge(pg, v, -10.0f);

        pg = svand_z(pg, pg_above_min, pg_below_max);
#endif
        v = sv_expf(v, pg);

        // compute tanh
        svfloat32_t numerator_v = svadd_x(pg, v, -1.0f);
        svfloat32_t denominator_v = svadd_x(pg, v, +1.0f);
        v = svdiv_x(pg, numerator_v, denominator_v);

#ifdef ENABLE_ONE_APROX
        pg = pg_tot;

        v = svdup_f32_m(v, svnot_z(pg, pg_above_min), +1.0f);
        v = svdup_f32_m(v, svnot_z(pg, pg_below_max), -1.0f);
#endif
        // Store back
        svst1(pg, values + i, v);
    }
}

void tanh_libc(float *data, uint32_t len)
{
    for(uint32_t i = 0; i < len; ++i)
        data[i] = tanhf(data[i]);
}
