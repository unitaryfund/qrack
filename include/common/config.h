#define ENABLE_ALU 1
/* #undef ENABLE_BCD */
#define ENABLE_COMPLEX_X2 (!(defined(__GNUC__) || defined(__MINGW32__)) || ((FPPOW == 5) && __SSE__) || ((FPPOW == 6) && __SSE2__))
#define ENABLE_SSE3 (!(defined(__GNUC__) || defined(__MINGW32__)) || __SSE3__)
#define ENABLE_DEVRAND 1
#define ENABLE_ENV_VARS 1
#define ENABLE_OCL_MEM_GUARDS 1
#define ENABLE_OPENCL 1
#define ENABLE_PTHREAD 1
#define ENABLE_QBDT_CPU_PARALLEL 1
#define ENABLE_QBDT 1
/* #undef ENABLE_REG_GATES */
/* #undef ENABLE_ROT_API */
#define SEED_DEVRAND 1
/* #undef ENABLE_SNUCL */
#define ENABLE_QUNIT_CPU_PARALLEL 1
#define FPPOW 5
#define PSTRIDEPOW 11
#define QBCAPPOW 12
#define UINTPOW 6
/* #undef OPENCL_V3 */
#define CPP_STD 14
