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

#include "common/qrack_types.hpp"

namespace Qrack {

__global__ void apply2x2(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted);
__global__ void apply2x2single(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2double(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2wide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted);
__global__ void apply2x2singlewide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2doublewide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2normsingle(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* nrmParts, qCudaReal1* lProbBuffer);
__global__ void apply2x2normsinglewide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* nrmParts, qCudaReal1* lProbBuffer);

__global__ void xsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void xsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void zsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void zsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);

__global__ void uniformlycontrolled(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    bitCapIntOcl* qPowers, qCudaCmplx* mtrxs, qCudaReal1* nrmIn,
    qCudaReal1* nrmParts, qCudaReal1* lProbBuffer);

__global__ void compose(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void composewide(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void composemid(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void decomposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* remainderStateProb, qCudaReal1* remainderStateAngle, qCudaReal1* partStateProb,
    qCudaReal1* partStateAngle);
__global__ void decomposeamp(qCudaReal1* stateProb, qCudaReal1* stateAngle,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void dispose(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);

__global__ void prob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, qCudaReal1* lProbBuffer);
__global__ void probreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, qCudaReal1* lProbBuffer);
__global__ void probregall(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* oneChanceBuffer);
__global__ void probmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, bitCapIntOcl* qPowers, qCudaReal1* lProbBuffer);
__global__ void probmaskall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, bitCapIntOcl* qPowersMask, bitCapIntOcl* qPowersSkip);

__global__ void rol(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);

__global__ void inc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void cinc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void incdecc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incs(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecsc1(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecsc2(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incbcd(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecbcdc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void mul(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void div(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void mulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void imulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void powmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void cmul(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cdiv(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cmulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cimulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cpowmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);

__global__ void fulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void ifulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);

__global__ void indexedLda(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitLenInt* values);
__global__ void indexedAdc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitLenInt* values);
__global__ void indexedSbc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitLenInt* values);
__global__ void hash(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitLenInt* values);

__global__ void nrmlze(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr);
__global__ void nrmlzewide(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr);
__global__ void updatenorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* args_ptr, qCudaReal1* norm_ptr, qCudaReal1* lProbBuffer);

__global__ void approxcompare(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* norm_ptr, qCudaReal1* lProbBuffer);

__global__ void applym(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr);
__global__ void applymreg(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr);

__global__ void phaseflip(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void zerophaseflip(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void cphaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void phaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);

} // namespace Qrack
