//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License VMIN_FUSION_BITS.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <future>
#include <ctime>
#include <initializer_list>
#include <map>

#include "qfactory.hpp"
#include "qfusion.hpp"

namespace Qrack {

QFusion::QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
    , engineType(eng)
    , bitBuffers(qBitCount)
{
    if (rgp == nullptr) {
        /* Used to control the random seed for all allocated interfaces. */
        rand_generator = std::make_shared<std::default_random_engine>();
        rand_generator->seed(std::time(0));
    } else {
        rand_generator = rgp;
    }

    qReg = CreateQuantumInterface(engineType, qBitCount, initState, rand_generator);
}

void QFusion::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex)
{
    if (qubitCount < MIN_FUSION_BITS) {
        FlushBit(qubitIndex);
        qReg->ApplySingleBit(mtrx, doCalcNorm, qubitIndex);
        return;
    }

    std::shared_ptr<complex[4]> outBuffer(new complex[4]);
    if (bitBuffers[qubitIndex]) {
        std::shared_ptr<complex[4]> inBuffer = bitBuffers[qubitIndex];
        std::vector<std::future<void>> futures(4);

        futures[0] = std::async(std::launch::async, [&]() {
            outBuffer[0] = (mtrx[0] * inBuffer[0]) + (mtrx[1] * inBuffer[2]);
            if (norm(outBuffer[0]) < min_norm) {
                outBuffer[0] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[1] = std::async(std::launch::async, [&]() {
            outBuffer[1] = (mtrx[0] * inBuffer[1]) + (mtrx[1] * inBuffer[3]);
            if (norm(outBuffer[1]) < min_norm) {
                outBuffer[1] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[2] = std::async(std::launch::async, [&]() {
            outBuffer[2] = (mtrx[2] * inBuffer[0]) + (mtrx[3] * inBuffer[2]);
            if (norm(outBuffer[2]) < min_norm) {
                outBuffer[2] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[3] = std::async(std::launch::async, [&]() {
            outBuffer[3] = (mtrx[2] * inBuffer[1]) + (mtrx[3] * inBuffer[3]);
            if (norm(outBuffer[3]) < min_norm) {
                outBuffer[3] = complex(ZERO_R1, ZERO_R1);
            }
        });

        for (int i = 0; i < 4; i++) {
            futures[i].get();
        }
    } else {
        std::copy(mtrx, mtrx + 4, outBuffer.get());
    }

    bitBuffers[qubitIndex] = outBuffer;
}

void QFusion::SetQuantumState(complex* inputState)
{
    DiscardAll();
    qReg->SetQuantumState(inputState);
}

void QFusion::GetQuantumState(complex* outputState)
{
    FlushAll();
    qReg->GetQuantumState(outputState);
}

complex QFusion::GetAmplitude(bitCapInt perm)
{
    FlushAll();
    return qReg->GetAmplitude(perm);
}

void QFusion::SetPermutation(bitCapInt perm)
{
    DiscardAll();
    qReg->SetPermutation(perm);
}

bitLenInt QFusion::Cohere(QFusionPtr toCopy)
{
    FlushAll();
    toCopy->FlushAll();
    bitLenInt toRet = qReg->Cohere(toCopy->qReg);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

std::map<QInterfacePtr, bitLenInt> QFusion::Cohere(std::vector<QFusionPtr> toCopy)
{
    std::vector<QInterfacePtr> tCI(toCopy.size());
    FlushAll();
    for (bitLenInt i = 0; i < toCopy.size(); i++) {
        toCopy[i]->FlushAll();
        tCI[i] = toCopy[i]->qReg;
    }
    std::map<QInterfacePtr, bitLenInt> toRet = qReg->Cohere(tCI);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

bitLenInt QFusion::Cohere(QInterfacePtr toCopy)
{
    FlushAll();
    bitLenInt toRet = qReg->Cohere(toCopy);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

std::map<QInterfacePtr, bitLenInt> QFusion::Cohere(std::vector<QInterfacePtr> toCopy)
{
    FlushAll();
    std::map<QInterfacePtr, bitLenInt> toRet = qReg->Cohere(toCopy);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

void QFusion::Decohere(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    FlushReg(start, length);
    qReg->Decohere(start, length, dest->qReg);
    dest->SetQubitCount(length);
    for (bitLenInt i = 0; i < length; i++) {
        dest->bitBuffers[i] = bitBuffers[start + i];
    }
    bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    SetQubitCount(qReg->GetQubitCount());
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
}

void QFusion::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    FlushReg(start, length);
    qReg->Decohere(start, length, dest);
    bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    SetQubitCount(qReg->GetQubitCount());
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
}

void QFusion::Dispose(bitLenInt start, bitLenInt length)
{
    DiscardReg(start, length);
    qReg->Dispose(start, length);
    bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    SetQubitCount(qReg->GetQubitCount());
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
}

void QFusion::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushList(controls, controlLen);
    FlushBit(target);
    qReg->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
}

void QFusion::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushList(controls, controlLen);
    FlushBit(target);
    qReg->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
}

void QFusion::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CSqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CISqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
}

bool QFusion::ForceM(bitLenInt qubit, bool result, bool doForce, real1 nrmlzr)
{
    FlushBit(qubit);
    return qReg->ForceM(qubit, result, doForce, nrmlzr);
}

void QFusion::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->INC(toAdd, start, length);
}

void QFusion::CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inOutStart, length);
    qReg->CINC(toAdd, inOutStart, length, controls, controlLen);
}

void QFusion::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->INCC(toAdd, start, length, carryIndex);
}

void QFusion::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    FlushReg(start, length);
    FlushBit(overflowIndex);
    qReg->INCS(toAdd, start, length, overflowIndex);
}

void QFusion::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(overflowIndex);
    FlushBit(carryIndex);
    qReg->INCSC(toAdd, start, length, overflowIndex, carryIndex);
}

void QFusion::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->INCSC(toAdd, start, length, carryIndex);
}

void QFusion::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->INCBCD(toAdd, start, length);
}

void QFusion::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->INCBCDC(toAdd, start, length, carryIndex);
}

void QFusion::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->DEC(toSub, start, length);
}

void QFusion::CDEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inOutStart, length);
    qReg->CDEC(toSub, inOutStart, length, controls, controlLen);
}

void QFusion::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->DECC(toSub, start, length, carryIndex);
}

void QFusion::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    FlushReg(start, length);
    FlushBit(overflowIndex);
    qReg->DECS(toSub, start, length, overflowIndex);
}

void QFusion::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(overflowIndex);
    FlushBit(carryIndex);
    qReg->DECSC(toSub, start, length, overflowIndex, carryIndex);
}

void QFusion::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->DECSC(toSub, start, length, carryIndex);
}

void QFusion::DECBCD(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->DECBCD(toSub, start, length);
}

void QFusion::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->DECBCDC(toSub, start, length, carryIndex);
}

void QFusion::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    FlushReg(inOutStart, length);
    FlushReg(carryStart, length);
    qReg->MUL(toMul, inOutStart, carryStart, length);
}

void QFusion::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    FlushReg(inOutStart, length);
    FlushReg(carryStart, length);
    qReg->DIV(toDiv, inOutStart, carryStart, length);
}

void QFusion::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inOutStart, length);
    FlushReg(carryStart, length);
    qReg->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
}

void QFusion::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inOutStart, length);
    FlushReg(carryStart, length);
    qReg->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
}

void QFusion::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->ZeroPhaseFlip(start, length);
}

void QFusion::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    FlushReg(start, length);
    FlushBit(flagIndex);
    qReg->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
}

void QFusion::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    qReg->PhaseFlipIfLess(greaterPerm, start, length);
}

void QFusion::PhaseFlip() { qReg->PhaseFlip(); }

bitCapInt QFusion::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    FlushReg(indexStart, indexLength);
    FlushReg(valueStart, valueLength);
    return qReg->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values);
}

bitCapInt QFusion::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    FlushReg(indexStart, indexLength);
    FlushReg(valueStart, valueLength);
    FlushBit(carryIndex);
    return qReg->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

bitCapInt QFusion::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    FlushReg(indexStart, indexLength);
    FlushReg(valueStart, valueLength);
    FlushBit(carryIndex);
    return qReg->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

void QFusion::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    std::swap(bitBuffers[qubitIndex1], bitBuffers[qubitIndex2]);
    qReg->Swap(qubitIndex1, qubitIndex2);
}

void QFusion::SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    FlushBit(qubitIndex1);
    FlushBit(qubitIndex2);
    qReg->SqrtSwap(qubitIndex1, qubitIndex2);
}

void QFusion::ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    FlushBit(qubitIndex1);
    FlushBit(qubitIndex2);
    qReg->ISqrtSwap(qubitIndex1, qubitIndex2);
}

void QFusion::CopyState(QFusionPtr orig)
{
    FlushAll();
    orig->FlushAll();
    qReg->CopyState(orig->qReg);
}

void QFusion::CopyState(QInterfacePtr orig)
{
    FlushAll();
    qReg->CopyState(orig);
}

bool QFusion::IsPhaseSeparable(bool forceCheck)
{
    FlushAll();
    return IsPhaseSeparable(forceCheck);
}

real1 QFusion::Prob(bitLenInt qubitIndex)
{
    FlushBit(qubitIndex);
    return qReg->Prob(qubitIndex);
}

real1 QFusion::ProbAll(bitCapInt fullRegister)
{
    FlushAll();
    return qReg->ProbAll(fullRegister);
}
} // namespace Qrack
