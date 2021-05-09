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

#define IS_NORM_ZERO(c) (norm(c) <= amplitudeFloor)

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
    , shardsEigen(qubitCount)
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
    QInterfacePtr toRet = CreateQuantumInterface(engineType, subEngineType, qubitCount, 0, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int>{}, thresholdQubits, separabilityThreshold);
    toRet->SetConcurrency(concurrency);
    return toRet;
}

QInterfacePtr QStabilizerHybrid::Clone()
{
    Finish();

    QStabilizerHybridPtr c =
        std::dynamic_pointer_cast<QStabilizerHybrid>(CreateQuantumInterface(QINTERFACE_STABILIZER_HYBRID, engineType,
            subEngineType, qubitCount, 0, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID,
            useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}, thresholdQubits, separabilityThreshold));

    if (stabilizer) {
        c->stabilizer = std::make_shared<QStabilizer>(*stabilizer);
        for (bitLenInt i = 0; i < shards.size(); i++) {
            if (shards[i]) {
                c->shards[i] = std::make_shared<QStabilizerShard>(shards[i]->gate);
            }
            c->shardsEigen[i] = shardsEigen[i];
        }
    } else {
        complex* stateVec = new complex[(bitCapIntOcl)maxQPower];
        engine->GetQuantumState(stateVec);
        c->SwitchToEngine();
        c->engine->SetQuantumState(stateVec);
        delete[] stateVec;
    }

    return c;
}

void QStabilizerHybrid::SwitchToEngine()
{
    if (engine) {
        return;
    }

    complex* stateVec = new complex[(bitCapIntOcl)maxQPower];
    stabilizer->GetQuantumState(stateVec);

    engine = MakeEngine();
    engine->SetQuantumState(stateVec);
    delete[] stateVec;

    if (engineType != QINTERFACE_QUNIT) {
        stabilizer.reset();
        FlushBuffers();
        return;
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (stabilizer->IsSeparableZ(i)) {
            engine->SetBit(i, stabilizer->M(i));
            continue;
        }

        stabilizer->H(i);
        if (stabilizer->IsSeparableZ(i)) {
            engine->SetBit(i, stabilizer->M(i));
            engine->H(i);
            continue;
        }

        stabilizer->S(i);
        if (stabilizer->IsSeparableZ(i)) {
            engine->SetBit(i, stabilizer->M(i));
            engine->H(i);
            engine->S(i);
        }
    }

    stabilizer.reset();
    FlushBuffers();
}

void QStabilizerHybrid::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if (shards[control1] || shards[control2] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control1);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CNOT(control2, target);
            return;
        }

        prob = Prob(control2);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CNOT(control1, target);
            return;
        }

        SwitchToEngine();
    }

    engine->CCNOT(control1, control2, target);
}

void QStabilizerHybrid::CH(bitLenInt control, bitLenInt target)
{
    if (shards[control] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->H(target);
            return;
        }

        SwitchToEngine();
    }

    engine->CH(control, target);
}

void QStabilizerHybrid::CS(bitLenInt control, bitLenInt target)
{
    if (shards[control] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->S(target);
            return;
        }

        SwitchToEngine();
    }

    engine->CS(control, target);
}

void QStabilizerHybrid::CIS(bitLenInt control, bitLenInt target)
{
    if (shards[control] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->IS(target);
            return;
        }

        SwitchToEngine();
    }

    engine->CIS(control, target);
}

void QStabilizerHybrid::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if (shards[control1] || shards[control2] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control1);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CZ(control2, target);
            return;
        }

        prob = Prob(control2);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CZ(control1, target);
            return;
        }

        SwitchToEngine();
    }

    engine->CCZ(control1, control2, target);
}

void QStabilizerHybrid::CCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if (shards[control1] || shards[control2] || shards[target]) {
        FlushBuffers();
    }

    if (stabilizer) {
        real1_f prob = Prob(control1);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CY(control2, target);
            return;
        }

        prob = Prob(control2);
        if (prob == ZERO_R1) {
            return;
        }
        if (prob == ONE_R1) {
            stabilizer->CY(control1, target);
            return;
        }

        SwitchToEngine();
    }

    engine->CCY(control1, control2, target);
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
        dest->shardsEigen = shardsEigen;
        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
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
    std::copy(shardsEigen.begin() + start, shardsEigen.begin() + start + length, dest->shardsEigen.begin());
    shardsEigen.erase(shardsEigen.begin() + start, shardsEigen.begin() + start + length);
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

    if (engine) {
        engine->Dispose(start, length);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    shardsEigen.erase(shardsEigen.begin() + start, shardsEigen.begin() + start + length);
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

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    shardsEigen.erase(shardsEigen.begin() + start, shardsEigen.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::SetQuantumState(const complex* inputState)
{
    DumpBuffers();

    if (qubitCount == 1U) {
        bool isClifford = false;
        bool isSet;
        bool isX = false;
        bool isY = false;
        if (norm(inputState[1]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = false;
        } else if (norm(inputState[0]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = true;
        } else if (norm(inputState[0] - inputState[1]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = false;
            isX = true;
        } else if (norm(inputState[0] + inputState[1]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = true;
            isX = true;
        } else if (norm((I_CMPLX * inputState[0]) - inputState[1]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = false;
            isY = true;
        } else if (norm((I_CMPLX * inputState[0]) + inputState[1]) <= REAL1_EPSILON) {
            isClifford = true;
            isSet = true;
            isY = true;
        }

        engine.reset();

        if (isClifford) {
            if (stabilizer) {
                stabilizer->SetPermutation(isSet ? 1 : 0);
            } else {
                stabilizer = MakeStabilizer(isSet ? 1 : 0);
            }
            if (isX || isY) {
                stabilizer->H(0);
            }
            if (isY) {
                stabilizer->S(0);
            }
            return;
        }

        if (stabilizer) {
            stabilizer->SetPermutation(0);
        } else {
            stabilizer = MakeStabilizer(0);
        }

        real1 sqrtProb = sqrt(norm(inputState[1]));
        real1 sqrt1MinProb = sqrt(norm(inputState[0]));
        complex probMatrix[4] = { sqrt1MinProb, sqrtProb, sqrtProb, -sqrt1MinProb };
        ApplySingleBit(probMatrix, 0);
        ApplySinglePhase(inputState[0] / sqrt1MinProb, inputState[1] / sqrtProb, 0);

        return;
    }

    SwitchToEngine();
    engine->SetQuantumState(inputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    FlushBuffers();

    if (stabilizer) {
        complex* stateVec = new complex[(bitCapIntOcl)maxQPower];
        stabilizer->GetQuantumState(stateVec);
        for (bitCapIntOcl i = 0; i < maxQPower; i++) {
            outputProbs[i] = norm(stateVec[i]);
        }
        delete[] stateVec;
    } else {
        engine->GetProbs(outputProbs);
    }
}

void QStabilizerHybrid::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    bool wasCached;
    complex mtrx[4];
    if (shards[target]) {
        QStabilizerShardPtr shard = shards[target];
        shard->Compose(lMtrx);
        std::copy(shard->gate, shard->gate + 4, mtrx);
        shards[target] = NULL;
        wasCached = true;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
        wasCached = false;
    }

    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (!wasCached && IS_NORM_ZERO(mtrx[1]) && IS_NORM_ZERO(mtrx[2])) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }
    if (!wasCached && IS_NORM_ZERO(mtrx[0]) && IS_NORM_ZERO(mtrx[3])) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }
    if (IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) &&
        IS_SAME(mtrx[2], -mtrx[3])) {
        H(target);
        return;
    }

    if (stabilizer) {

        if (IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[3], complex(ZERO_R1, -SQRT1_2_R1))) {
            if (IS_SAME(mtrx[1], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[2], complex(ZERO_R1, SQRT1_2_R1))) {
                H(target);
                S(target);
                return;
            }

            if (IS_SAME(mtrx[1], complex(ZERO_R1, SQRT1_2_R1)) && IS_SAME(mtrx[2], complex(SQRT1_2_R1, ZERO_R1))) {
                S(target);
                H(target);
                return;
            }
        }

        if (IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[3], complex(ZERO_R1, SQRT1_2_R1))) {
            if (IS_SAME(mtrx[1], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[2], complex(ZERO_R1, -SQRT1_2_R1))) {
                H(target);
                IS(target);
                return;
            }

            if (IS_SAME(mtrx[1], complex(ZERO_R1, -SQRT1_2_R1)) && IS_SAME(mtrx[2], complex(SQRT1_2_R1, ZERO_R1))) {
                IS(target);
                H(target);
                return;
            }
        }

        if (IS_SAME(mtrx[0], complex(ONE_R1, -ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[1], complex(ONE_R1, ONE_R1) / (real1)2.0f) && IS_SAME(mtrx[0], mtrx[3]) &&
            IS_SAME(mtrx[1], mtrx[2])) {
            ISqrtX(target);
            return;
        }

        if (IS_SAME(mtrx[0], complex(ONE_R1, ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[1], complex(ONE_R1, -ONE_R1) / (real1)2.0f) && IS_SAME(mtrx[0], mtrx[3]) &&
            IS_SAME(mtrx[1], mtrx[2])) {
            SqrtX(target);
            return;
        }

        if (IS_SAME(mtrx[0], complex(ONE_R1, -ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[1], complex(ONE_R1, -ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[2], complex(-ONE_R1, ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[3], complex(ONE_R1, -ONE_R1) / (real1)2.0f)) {
            ISqrtY(target);
            return;
        }

        if (IS_SAME(mtrx[0], complex(ONE_R1, ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[1], complex(-ONE_R1, -ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[2], complex(ONE_R1, ONE_R1) / (real1)2.0f) &&
            IS_SAME(mtrx[3], complex(ONE_R1, ONE_R1) / (real1)2.0f)) {
            SqrtY(target);
            return;
        }

        if (!wasCached) {
            if (stabilizer->IsSeparableZ(target)) {
                shardsEigen[target] = PauliZ;
            } else if (stabilizer->IsSeparableX(target)) {
                shardsEigen[target] = PauliX;
            } else if (stabilizer->IsSeparableY(target)) {
                shardsEigen[target] = PauliY;
            } else {
                shardsEigen[target] = PauliI;
            }
        }

        QStabilizerShardPtr shard = std::make_shared<QStabilizerShard>(mtrx);
        if (shardsEigen[target]) {
            QStabilizerShardPtr nShard;
            // If in PauliX or PauliY basis, compose gate with conversion from/to PauliZ basis.
            if (shardsEigen[target] == PauliX) {
                complex nMtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
                    complex(SQRT1_2_R1, ZERO_R1), complex(-SQRT1_2_R1, ZERO_R1) };
                nShard = std::make_shared<QStabilizerShard>(nMtrx);
                nShard->Compose(shard->gate);
                nShard->Compose(nMtrx);
            } else if (shardsEigen[target] == PauliY) {
                complex nMtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1),
                    complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1) };
                complex aMtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
                    complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };
                nShard = std::make_shared<QStabilizerShard>(nMtrx);
                nShard->Compose(shard->gate);
                nShard->Compose(aMtrx);
            } else {
                nShard = std::make_shared<QStabilizerShard>(shard->gate);
            }

            if ((norm(nShard->gate[1]) < REAL1_EPSILON) && (norm(nShard->gate[2]) < REAL1_EPSILON)) {
                return;
            }
            if ((norm(nShard->gate[0]) < REAL1_EPSILON) && (norm(nShard->gate[3]) < REAL1_EPSILON)) {
                if (shardsEigen[target] == PauliZ) {
                    X(target);
                } else if (shardsEigen[target] == PauliX) {
                    H(target);
                    X(target);
                    H(target);
                } else {
                    IS(target);
                    H(target);
                    X(target);
                    H(target);
                    S(target);
                }
                return;
            }
        }

        shards[target] = shard;
        return;
    }

    SwitchToEngine();
    engine->ApplySingleBit(mtrx, target);
}

void QStabilizerHybrid::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    if (shards[target]) {
        complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
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

    complex sTest = bottomRight / topLeft;

    if (IS_SAME(sTest, I_CMPLX)) {
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(sTest, -I_CMPLX)) {
        stabilizer->IS(target);
        return;
    }

    if (stabilizer->IsSeparableZ(target)) {
        // This gate has no effect.
        return;
    }

    SwitchToEngine();
    engine->ApplySinglePhase(topLeft, bottomRight, target);
}

void QStabilizerHybrid::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    if (shards[target]) {
        complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
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
        stabilizer->X(target);
        stabilizer->Z(target);
        return;
    }

    complex sTest = topRight / bottomLeft;

    if (IS_SAME(sTest, I_CMPLX)) {
        stabilizer->X(target);
        stabilizer->S(target);
        return;
    }

    if (IS_SAME(sTest, -I_CMPLX)) {
        stabilizer->X(target);
        stabilizer->IS(target);
        return;
    }

    SwitchToEngine();
    engine->ApplySingleInvert(topRight, bottomLeft, target);
}

void QStabilizerHybrid::ApplyControlledSingleBit(
    const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& target, const complex* mtrx)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (controls.size() > 1U) {
        SwitchToEngine();
        engine->ApplyControlledSingleBit(lControls, lControlLen, target, mtrx);
        return;
    }

    if (IS_NORM_ZERO(mtrx[1]) && IS_NORM_ZERO(mtrx[2])) {
        ApplyControlledSinglePhase(&(controls[0]), 1U, target, mtrx[0], mtrx[3]);
        return;
    }

    if (IS_NORM_ZERO(mtrx[0]) && IS_NORM_ZERO(mtrx[3])) {
        ApplyControlledSingleInvert(&(controls[0]), 1U, target, mtrx[1], mtrx[2]);
        return;
    }

    if ((controls.size() == 1U) && IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1)) && IS_SAME(mtrx[0], mtrx[1]) &&
        IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[2], -mtrx[3])) {
        CH(controls[0], target);
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
    }

    FlushIfBlocked(controls, target);

    if (engine) {
        engine->ApplyControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
        return;
    }

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            return;
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->CZ(controls[0], target);
            return;
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            return;
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            return;
        }
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
    }

    FlushIfBlocked(controls, target);

    if (engine) {
        engine->ApplyControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
        return;
    }

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->CNOT(controls[0], target);
            return;
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            return;
        }
    }

    if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            return;
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            return;
        }
    }

    if (IS_SAME(topRight, -I_CMPLX) && IS_SAME(bottomLeft, I_CMPLX)) {
        stabilizer->CY(controls[0], target);
        return;
    }

    SwitchToEngine();
    engine->ApplyControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
}

void QStabilizerHybrid::ApplyAntiControlledSingleBit(
    const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& target, const complex* mtrx)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (controls.size() > 1U) {
        SwitchToEngine();
        engine->ApplyAntiControlledSingleBit(lControls, lControlLen, target, mtrx);
        return;
    }

    if (IS_NORM_ZERO(mtrx[1]) && IS_NORM_ZERO(mtrx[2])) {
        ApplyAntiControlledSinglePhase(&(controls[0]), 1U, target, mtrx[0], mtrx[3]);
        return;
    }

    if (IS_NORM_ZERO(mtrx[0]) && IS_NORM_ZERO(mtrx[3])) {
        ApplyAntiControlledSingleInvert(&(controls[0]), 1U, target, mtrx[1], mtrx[2]);
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

    if (controls.size() > 1U) {
        SwitchToEngine();
    }

    FlushIfBlocked(controls, target);

    if (engine) {
        engine->ApplyAntiControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
        return;
    }

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            return;
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }
    }

    SwitchToEngine();
    engine->ApplyAntiControlledSinglePhase(lControls, lControlLen, target, topLeft, bottomRight);
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

    FlushIfBlocked(controls, target);

    if (controls.size() > 1U) {
        SwitchToEngine();
    }

    if (engine) {
        engine->ApplyAntiControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
        return;
    }

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CNOT(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }
    }

    if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        if (IS_SAME(topRight, -ONE_CMPLX) && IS_SAME(bottomLeft, -ONE_CMPLX)) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }
    }

    if (IS_SAME(topRight, -I_CMPLX) && IS_SAME(bottomLeft, I_CMPLX)) {
        stabilizer->X(controls[0]);
        stabilizer->CY(controls[0], target);
        stabilizer->X(controls[0]);
        return;
    }

    SwitchToEngine();
    engine->ApplyAntiControlledSingleInvert(lControls, lControlLen, target, topRight, bottomLeft);
}

bitCapInt QStabilizerHybrid::MAll()
{
    FlushBuffers();

    if (stabilizer) {
        bitCapIntOcl toRet = 0;
        for (bitLenInt i = 0; i < qubitCount; i++) {
            toRet |= ((stabilizer->M(i) ? 1 : 0) << i);
        }
        return (bitCapInt)toRet;
    }

    SwitchToEngine();
    return engine->MAll();
}
} // namespace Qrack
