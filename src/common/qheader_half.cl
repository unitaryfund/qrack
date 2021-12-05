//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma OPENCL EXTENSION cl_nv_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define cmplx half2
#define cmplx2 half4
#define cmplx4 half8
#define real1 half
#define real1_f float
#define ZERO_R1 ((real1)0.0f)
#define ONE_R1 ((real1)1.0f)
#define SineShift ((real1)M_PI_2_F)
#define PI_R1 ((real1)M_PI_F)
#define REAL1_EPSILON ((real1)2e-17h)

// Macro is defined, and functions are undefined, for NVIDIA pragma
#ifdef __OVERLOADABLE__
real1 __OVERLOADABLE__ dot(const half2 a, const half2 b) {
    float2 af = (float2)((float)a.x, (float)a.y);
    float2 bf = (float2)((float)b.x, (float)b.y);
    return dot(af, bf);
}

real1 __OVERLOADABLE__ dot(const half4 a, const half4 b) {
    float4 af = (float4)((float)a.x, (float)a.y, (float)a.z, (float)a.w);
    float4 bf = (float4)((float)b.x, (float)b.y, (float)b.z, (float)b.w);
    return dot(af, bf);
}

half2 __OVERLOADABLE__ sin(const half2 a) {
    float2 af = sin((float2)((float)a.x, (float)a.y));
    return (half2)((half)af.x, (half)af.y);
}

half __OVERLOADABLE__ sqrt(const half a) {
    return sqrt((float)a);
}

half2 __OVERLOADABLE__ sqrt(const half2 a) {
    float2 af = sqrt((float2)((float)a.x, (float)a.y));
    return (half2)((half)af.x, (half)af.y);
}
#endif
