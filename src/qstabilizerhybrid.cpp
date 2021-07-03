//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <thread>

#include "qfactory.hpp"
#include "qstabilizerhybrid.hpp"

#define IS_NORM_0(c) (norm(c) <= amplitudeFloor)
#define IS_REAL_0(r) (abs(r) <= FP_NORM_EPSILON)
#define IS_CTRLED_CLIFFORD(top, bottom)                                                                                \
    ((IS_REAL_0(std::real(top)) || IS_REAL_0(std::imag(top))) && (IS_SAME(top, bottom) || IS_SAME(top, -bottom)))
#define IS_CLIFFORD(mtrx)                                                                                              \
    (IS_SAME(mtrx[0], mtrx[1]) || IS_SAME(mtrx[0], -mtrx[1]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) ||                 \
        IS_SAME(mtrx[0], -I_CMPLX * mtrx[1])) &&                                                                       \
        (IS_SAME(mtrx[0], mtrx[2]) || IS_SAME(mtrx[0], -mtrx[2]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) ||             \
            IS_SAME(mtrx[0], -I_CMPLX * mtrx[2])) &&                                                                   \
        (IS_SAME(mtrx[0], mtrx[3]) || IS_SAME(mtrx[0], -mtrx[3]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[3]) ||             \
            IS_SAME(mtrx[0], -I_CMPLX * mtrx[3]))

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount,
    bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem,
    int deviceId, bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , engineType(eng)
    , subEngineType(subEng)
    , engine(NULL)
    , shards(qubitCount)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , separabilityThreshold(sep_thresh)
    , thresholdQubits(qubitThreshold)
{
    if (subEngineType == QINTERFACE_STABILIZER_HYBRID) {
#if ENABLE_OPENCL
        subEngineType = OCLEngine::Instance()->GetDeviceCount() ? QINTERFACE_HYBRID : QINTERFACE_CPU;
#else
        subEngineType = QINTERFACE_CPU;
#endif
    }

    if (engineType == QINTERFACE_STABILIZER_HYBRID) {
#if ENABLE_OPENCL
        engineType = OCLEngine::Instance()->GetDeviceCount() ? QINTERFACE_HYBRID : QINTERFACE_CPU;
#else
        engineType = QINTERFACE_CPU;
#endif
    }

    if ((engineType == QINTERFACE_QPAGER) && (subEngineType == QINTERFACE_QPAGER)) {
#if ENABLE_OPENCL
        subEngineType = OCLEngine::Instance()->GetDeviceCount() ? QINTERFACE_HYBRID : QINTERFACE_CPU;
#else
        subEngineType = QINTERFACE_CPU;
#endif
    }

    concurrency = std::thread::hardware_concurrency();
    stabilizer = MakeStabilizer(initState);
    amplitudeFloor = REAL1_EPSILON;
}

QStabilizerPtr QStabilizerHybrid::MakeStabilizer(const bitCapInt& perm)
{
    return std::make_shared<QStabilizer>(qubitCount, perm, useRDRAND, rand_generator);
}

QInterfacePtr QStabilizerHybrid::MakeEngine(const bitCapInt& perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineType, subEngineType, qubitCount, perm, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int>{}, thresholdQubits, separabilityThreshold);
    toRet->SetConcurrency(concurrency);
    return toRet;
}

void QStabilizerHybrid::CacheEigenstate(const bitLenInt& target)
{
    if (engine) {
        return;
    }

    QStabilizerShardPtr toRet = NULL;
    // If in PauliX or PauliY basis, compose gate with conversion from/to PauliZ basis.
    if (stabilizer->IsSeparableX(target)) {
        // X eigenstate
        stabilizer->H(target);

        complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(-SQRT1_2_R1, ZERO_R1) };
        toRet = std::make_shared<QStabilizerShard>(mtrx);
    } else if (stabilizer->IsSeparableY(target)) {
        // Y eigenstate
        stabilizer->H(target);
        stabilizer->S(target);

        complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1) };
        toRet = std::make_shared<QStabilizerShard>(mtrx);
    }

    if (!toRet) {
        return;
    }

    if (shards[target]) {
        toRet->Compose(shards[target]->gate);
    }

    shards[target] = toRet;

    if (IS_CLIFFORD(shards[target]->gate)) {
        QStabilizerShardPtr shard = shards[target];
        shards[target] = NULL;
        ApplySingleBit(shard->gate, target);
    }
}

QInterfacePtr QStabilizerHybrid::Clone()
{
    QStabilizerHybridPtr c =
        std::dynamic_pointer_cast<QStabilizerHybrid>(CreateQuantumInterface(QINTERFACE_STABILIZER_HYBRID, engineType,
            subEngineType, qubitCount, 0, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID,
            useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}, thresholdQubits, separabilityThreshold));

    // TODO: Remove.
    SwitchToEngine();

    Finish();
    c->Finish();

    if (stabilizer) {
        c->engine = NULL;
        c->stabilizer = stabilizer->Clone();
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (shards[i]) {
                c->shards[i] = std::make_shared<QStabilizerShard>(shards[i]->gate);
            }
        }
    } else {
        // Clone and set engine directly.
        c->engine = engine->Clone();
        c->stabilizer = NULL;
    }

    return c;
}

void QStabilizerHybrid::SwitchToEngine()
{
    if (engine) {
        return;
    }

    engine = MakeEngine();
    stabilizer->GetQuantumState(engine);
    stabilizer.reset();
    FlushBuffers();
}

void QStabilizerHybrid::Decompose(bitLenInt start, QStabilizerHybridPtr dest)
{
    bitLenInt length = dest->qubitCount;

    if (length == qubitCount) {
        dest->stabilizer = stabilizer;
        stabilizer = NULL;
        dest->engine = engine;
        engine = NULL;

        dest->shards = shards;
        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        dest->SwitchToEngine();
        engine->Decompose(start, dest->engine);
        SetQubitCount(qubitCount - length);
        return;
    }

    if (dest->engine) {
        dest->engine.reset();
        dest->stabilizer = dest->MakeStabilizer(0);
    }

    stabilizer->Decompose(start, dest->stabilizer);
    std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length)
{
    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        engine->Dispose(start, length);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::SetQuantumState(const complex* inputState)
{
    DumpBuffers();

    if (qubitCount == 1U) {
        engine = NULL;

        if (stabilizer) {
            stabilizer->SetPermutation(0);
        } else {
            stabilizer = MakeStabilizer(0);
        }

        real1 prob = clampProb(norm(inputState[1]));
        real1 sqrtProb = sqrt(prob);
        real1 sqrt1MinProb = sqrt(clampProb(ONE_R1 - prob));
        complex phase0 = std::polar(ONE_R1, arg(inputState[0]));
        complex phase1 = std::polar(ONE_R1, arg(inputState[1]));
        complex mtrx[4] = { sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
        ApplySingleBit(mtrx, 0);

        return;
    }

    SwitchToEngine();
    engine->SetQuantumState(inputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    FlushBuffers();

    if (stabilizer) {
        stabilizer->GetProbs(outputProbs);
    } else {
        engine->GetProbs(outputProbs);
    }
}

void QStabilizerHybrid::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4];
    if (shards[target]) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }

    if (engine) {
        engine->ApplySingleBit(mtrx, target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        stabilizer->H(target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        // Equivalent to X before H
        stabilizer->ISqrtY(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        // Equivalent to H before X
        stabilizer->SqrtY(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        stabilizer->X(target);
        stabilizer->SqrtY(target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        stabilizer->H(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        stabilizer->ISqrtY(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        stabilizer->Y(target);
        stabilizer->H(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        stabilizer->Z(target);
        stabilizer->H(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        stabilizer->IS(target);
        stabilizer->H(target);
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        stabilizer->IS(target);
        stabilizer->SqrtY(target);
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        stabilizer->IS(target);
        stabilizer->H(target);
        stabilizer->Y(target);
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        stabilizer->IS(target);
        stabilizer->H(target);
        stabilizer->Z(target);
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        stabilizer->SqrtX(target);
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        stabilizer->ISqrtX(target);
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        stabilizer->SqrtX(target);
        stabilizer->Z(target);
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        stabilizer->Z(target);
        stabilizer->SqrtX(target);
        return;
    }

    shards[target] = std::make_shared<QStabilizerShard>(mtrx);
    CacheEigenstate(target);
}

void QStabilizerHybrid::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (shards[target]) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (engine) {
        engine->ApplySinglePhase(topLeft, bottomRight, target);
        return;
    }

    if (IS_SAME(topLeft, bottomRight)) {
        return;
    }

    if (IS_SAME(topLeft, -bottomRight)) {
        stabilizer->Z(target);
        return;
    }

    if (IS_SAME(topLeft, -I_CMPLX * bottomRight)) {
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(topLeft, I_CMPLX * bottomRight)) {
        stabilizer->IS(target);
        return;
    }

    if (stabilizer->IsSeparableZ(target)) {
        // This gate has no effect.
        return;
    }

    shards[target] = std::make_shared<QStabilizerShard>(mtrx);
    CacheEigenstate(target);
}

void QStabilizerHybrid::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (shards[target]) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (engine) {
        engine->ApplySingleInvert(topRight, bottomLeft, target);
        return;
    }

    if (IS_SAME(topRight, bottomLeft)) {
        stabilizer->X(target);
        return;
    }

    if (IS_SAME(topRight, -bottomLeft)) {
        stabilizer->Y(target);
        return;
    }

    if (IS_SAME(topRight, -I_CMPLX * bottomLeft)) {
        stabilizer->X(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(topRight, I_CMPLX * bottomLeft)) {
        stabilizer->S(target);
        stabilizer->X(target);
        return;
    }

    if (stabilizer->IsSeparableZ(target)) {
        // This gate has no meaningful effect on phase.
        stabilizer->X(target);
        return;
    }

    shards[target] = std::make_shared<QStabilizerShard>(mtrx);
    CacheEigenstate(target);
}

void QStabilizerHybrid::ApplyControlledSingleBit(
    const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        ApplyControlledSinglePhase(lControls, lControlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        ApplyControlledSingleInvert(lControls, lControlLen, target, mtrx[1], mtrx[2]);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleBit(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->ApplyControlledSingleBit(lControls, lControlLen, target, mtrx);
}

void QStabilizerHybrid::ApplyControlledSinglePhase(const bitLenInt* lControls, const bitLenInt& lControlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        ApplySinglePhase(topLeft, bottomRight, target);
        return;
    }

    if (controls.size() > 1U) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target, true);
    }

    if (engine) {
        engine->ApplyControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
        return;
    }

    bitLenInt control = controls[0];
    bool didDivert = false;

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            didDivert = true;
        } else if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->CZ(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            stabilizer->CNOT(control, target);
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topLeft, I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CY(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomRight, -I_CMPLX)) {
            stabilizer->CY(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topLeft, -I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            stabilizer->CNOT(control, target);
            stabilizer->CY(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomRight, -I_CMPLX)) {
            stabilizer->CY(control, target);
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        }
    }

    if (didDivert) {
        if (shards[control]) {
            CacheEigenstate(control);
        }
        if (shards[target]) {
            CacheEigenstate(target);
        }
        return;
    }

    SwitchToEngine();
    engine->ApplyControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
}

void QStabilizerHybrid::ApplyControlledSingleInvert(const bitLenInt* lControls, const bitLenInt& lControlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleInvert(topRight, bottomLeft, target);
        return;
    }

    if (controls.size() > 1U) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target);
    }

    if (engine) {
        engine->ApplyControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
        return;
    }

    bitLenInt control = controls[0];
    bool didDivert = false;

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->CNOT(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->CNOT(control, target);
            stabilizer->CZ(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CNOT(control, target);
            stabilizer->CZ(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topRight, I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CY(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomLeft, -I_CMPLX)) {
            stabilizer->CZ(control, target);
            stabilizer->CY(control, target);
            stabilizer->CZ(control, target);
            didDivert = true;
        }
    } else if (IS_SAME(topRight, -I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            stabilizer->CY(control, target);
            didDivert = true;
        } else if (IS_SAME(bottomLeft, -I_CMPLX)) {
            stabilizer->CY(control, target);
            stabilizer->CZ(control, target);
            didDivert = true;
        }
    }

    if (didDivert) {
        if (shards[control]) {
            CacheEigenstate(control);
        }
        if (shards[target]) {
            CacheEigenstate(target);
        }
        return;
    }

    SwitchToEngine();
    engine->ApplyControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
}

void QStabilizerHybrid::ApplyAntiControlledSingleBit(
    const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        ApplyAntiControlledSinglePhase(lControls, lControlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        ApplyAntiControlledSingleInvert(lControls, lControlLen, target, mtrx[1], mtrx[2]);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleBit(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->ApplyAntiControlledSingleBit(lControls, lControlLen, target, mtrx);
}

void QStabilizerHybrid::ApplyAntiControlledSinglePhase(const bitLenInt* lControls, const bitLenInt& lControlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        ApplySinglePhase(topLeft, bottomRight, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target, true);
    }

    if (engine) {
        engine->ApplyAntiControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
        return;
    }

    X(controls[0]);
    ApplyControlledSinglePhase(&(controls[0]), 1U, target, topLeft, bottomRight);
    X(controls[0]);
}

void QStabilizerHybrid::ApplyAntiControlledSingleInvert(const bitLenInt* lControls, const bitLenInt& lControlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleInvert(topRight, bottomLeft, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target);
    }

    if (engine) {
        engine->ApplyAntiControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
        return;
    }

    X(controls[0]);
    ApplyControlledSingleInvert(&(controls[0]), 1U, target, topRight, bottomLeft);
    X(controls[0]);
}

real1_f QStabilizerHybrid::Prob(bitLenInt qubit)
{
    if (engine) {
        return engine->Prob(qubit);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shards[qubit] && !shards[qubit]->IsPhase()) {
        // Bit was already rotated to Z basis, if separable.
        if (stabilizer->IsSeparableZ(qubit)) {
            if (stabilizer->M(qubit)) {
                return norm(shards[qubit]->gate[3]);
            }
            return norm(shards[qubit]->gate[2]);
        }

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        return ONE_R1 / 2;
    }

    if (stabilizer->IsSeparableZ(qubit)) {
        return stabilizer->M(qubit) ? ONE_R1 : ZERO_R1;
    }

    // Otherwise, state appears locally maximally mixed.
    return ONE_R1 / 2;
}

bool QStabilizerHybrid::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    // This check will first try to coax into decomposable form:
    if (stabilizer && !stabilizer->CanDecomposeDispose(qubit, 1)) {
        SwitchToEngine();
    }

    if (engine) {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shards[qubit]) {
        if (!shards[qubit]->IsPhase() && stabilizer->IsSeparableZ(qubit)) {
            if (doForce) {
                if (doApply) {
                    if (result != stabilizer->M(qubit)) {
                        stabilizer->X(qubit);
                    }
                    shards[qubit] = NULL;
                }

                return result;
            }
            // Bit was already rotated to Z basis, if separable.
            return CollapseSeparableShard(qubit);
        }

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        shards[qubit] = NULL;
    }

    return stabilizer->M(qubit, result, doForce, doApply);
}

bitCapInt QStabilizerHybrid::MAll()
{
    if (stabilizer) {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (shards[i] && shards[i]->IsInvert()) {
                InvertBuffer(i);
            }
            if (shards[i]) {
                if (!shards[i]->IsPhase() && stabilizer->IsSeparableZ(i)) {
                    // Bit was already rotated to Z basis, if separable.
                    CollapseSeparableShard(i);
                }

                // Otherwise, buffer will not change the fact that state appears maximally mixed.
                shards[i] = NULL;
            }
        }
    }

    bitCapIntOcl toRet = 0;
    if (stabilizer) {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (stabilizer->M(i)) {
                toRet |= pow2(i);
            }
        }
    } else {
        toRet = engine->MAll();
    }

    SetPermutation(toRet);

    return toRet;
}
} // namespace Qrack
