//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

namespace Qrack {

__global__ void apply2x2(__global__ cmplx* stateVec, __constant__ real1* cmplxPtr, __constant__ bitCapInt* bitCapIntPtr,
    __constant__ bitCapInt* qPowersSorted);
__global__ void apply2x2single(
    __global__ cmplx* stateVec, __constant__ real1* cmplxPtr, __constant__ bitCapInt* bitCapIntPtr);
__global__ void apply2x2double(
    __global__ cmplx* stateVec, __constant__ real1* cmplxPtr, __constant__ bitCapInt* bitCapIntPtr);
__global__ void apply2x2wide(__global__ cmplx* stateVec, __constant__ real1* cmplxPtr,
    __constant__ bitCapInt* bitCapIntPtr, __constant__ bitCapInt* qPowersSorted);
__global__ void apply2x2singlewide(
    __global__ cmplx* stateVec, __constant__ real1* cmplxPtr, __constant__ bitCapInt* bitCapIntPtr);
__global__ void apply2x2doublewide(
    __global__ cmplx* stateVec, __constant__ real1* cmplxPtr, __constant__ bitCapInt* bitCapIntPtr);
__global__ void apply2x2normsingle(__global__ cmplx* stateVec, __constant__ real1* cmplxPtr,
    __constant__ bitCapInt* bitCapIntPtr, __global__ real1* nrmParts, __shared__ real1* lProbBuffer);
__global__ void apply2x2normsinglewide(__global__ cmplx* stateVec, __constant__ real1* cmplxPtr,
    __constant__ bitCapInt* bitCapIntPtr, __global__ real1* nrmParts, __shared__ real1* lProbBuffer);

__global__ void xsingle(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void xsinglewide(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void zsingle(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void zsinglewide(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);

__global__ void uniformlycontrolled(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __constant__ bitCapInt* qPowers, __constant__ cmplx4* mtrxs, __constant__ real1* nrmIn, __global__ real1* nrmParts,
    __shared__ real1* lProbBuffer);

__global__ void compose(__global__ cmplx* stateVec1, __global__ cmplx* stateVec2, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec);
__global__ void composewide(__global__ cmplx* stateVec1, __global__ cmplx* stateVec2,
    __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void composemid(__global__ cmplx* stateVec1, __global__ cmplx* stateVec2,
    __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void decomposeprob(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ real1* remainderStateProb, __global__ real1* remainderStateAngle, __global__ real1* partStateProb,
    __global__ real1* partStateAngle);
__global__ void decomposeamp(__global__ real1* stateProb, __global__ real1* stateAngle,
    __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void dispose(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);

__global__ void prob(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ real1* oneChanceBuffer, __shared__ real1* lProbBuffer);
__global__ void probreg(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ real1* oneChanceBuffer, __shared__ real1* lProbBuffer);
__global__ void probregall(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ real1* oneChanceBuffer);
__global__ void probmask(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ real1* oneChanceBuffer, __constant__ bitCapInt* qPowers, __shared__ real1* lProbBuffer);
__global__ void probmaskall(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ real1* oneChanceBuffer, __constant__ bitCapInt* qPowersMask, __constant__ bitCapInt* qPowersSkip);

__global__ void rol(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);

__global__ void inc(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void cinc(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec,
    __constant__ bitCapInt* controlPowers);
__global__ void incdecc(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void incs(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void incdecsc1(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void incdecsc2(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void incbcd(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void incdecbcdc(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void mul(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void div(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void mulmodnout(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void imulmodnout(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void powmodnout(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec);
__global__ void cmul(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec,
    __constant__ bitCapInt* controlPowers);
__global__ void cdiv(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec,
    __constant__ bitCapInt* controlPowers);
__global__ void cmulmodnout(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitCapInt* controlPowers);
__global__ void cimulmodnout(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitCapInt* controlPowers);
__global__ void cpowmodnout(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitCapInt* controlPowers);

__global__ void fulladd(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void ifulladd(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);

__global__ void indexedLda(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitLenInt* values);
__global__ void indexedAdc(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitLenInt* values);
__global__ void indexedSbc(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __global__ cmplx* nStateVec, __constant__ bitLenInt* values);
__global__ void hash(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __global__ cmplx* nStateVec,
    __constant__ bitLenInt* values);

__global__ void nrmlze(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __constant__ real1* args_ptr);
__global__ void nrmlzewide(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __constant__ real1* args_ptr);
__global__ void updatenorm(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr,
    __constant__ real1* args_ptr, __global__ real1* norm_ptr, __shared__ real1* lProbBuffer);

__global__ void approxcompare(__global__ cmplx* stateVec1, __global__ cmplx* stateVec2,
    __constant__ bitCapInt* bitCapIntPtr, __global__ real1* norm_ptr, __shared__ real1* lProbBuffer);

__global__ void applym(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __constant__ cmplx* cmplx_ptr);
__global__ void applymreg(
    __global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr, __constant__ cmplx* cmplx_ptr);

__global__ void phaseflip(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void zerophaseflip(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void cphaseflipifless(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);
__global__ void phaseflipifless(__global__ cmplx* stateVec, __constant__ bitCapInt* bitCapIntPtr);

} // namespace Qrack
