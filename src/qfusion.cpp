//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License VMIN_FUSION_BITS.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <ctime>
#include <initializer_list>
#include <map>

#include "qfactory.hpp"
#include "qfusion.hpp"

namespace Qrack {

QFusion::QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG,
    bool useSparseStateVec)
    : QInterface(qBitCount, rgp, deviceID, useHardwareRNG)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , randGlobalPhase(randomGlobalPhase)
    , bitBuffers(qBitCount)
    , bitControls(qBitCount)
{
    qReg = CreateQuantumInterface(eng, qBitCount, initState, rgp, phaseFactor, doNormalize, randGlobalPhase, useHostMem,
        deviceID, useHardwareRNG, useSparseStateVec);
}

QFusion::QFusion(QInterfacePtr target)
    : QInterface(target->GetQubitCount())
    , phaseFactor(complex(-999.0, -999.0))
    , doNormalize(true)
    , randGlobalPhase(true)
    , bitBuffers(target->GetQubitCount())
    , bitControls(target->GetQubitCount())
{
    qReg = target;
}

/**
 * All buffering operations happen in ApplySingleBit, which underlies the application of all 2x2 complex element matrix
 * operators, such as H, X, Y, Z, RX, etc..
 *
 * Without a QFusion layer, ApplySingleBit applies the Kroenecker product of a 2x2 complex element matrix operator to a
 * state vector of 2^N elements for N bits. With tensor slicing, this implies a complexity that scales as approximately
 * 2^(N+1) complex multiplications for each single bit gate application. However, the succesive application of single
 * bit gate "A" followed by single bit gate "B" is equal to a single application of their matrix product, "B*A". (Note
 * that B and A do not necessarily "commute," that "B*A" is not generally equal to "A*B," without additional
 * constraints.)
 *
 * Composing two single bit gates into one gate requires a constant 8 complex multiplications. Adding any additional
 * number of gates requires an additional 8 complex multiplications per gate, independent of the number of qubits in the
 * composed state vector. Ultimately applying the buffered, composed gate with tensor slicing requires 2^(N+1) complex
 * multiplications, once. Hence, if a QEngine has at least 3 qubits, the successive application of at least 2 gates on
 * the same bit is cheaper with "gate fusion," (M-1)*8+2^(N+1) multiplications for M gates instead of M*(2^(N+1))
 * multiplications.
 *
 * QFusion must flush these buffers, applying them to the state vector, when an operation is applied that can't be
 * buffered (and doesn't "commute") and before output from qubits. The rest of the engine simply wraps the other public
 * methods of QInterface to flush or discard the buffers as necessary.
 */
void QFusion::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex)
{
    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates.
    if (qubitCount < MIN_FUSION_BITS) {
        // Directly apply the gate and return.
        FlushBit(qubitIndex);
        qReg->ApplySingleBit(mtrx, doCalcNorm, qubitIndex);
        return;
    }

    // If we pass the threshold number of qubits for buffering, we just do 2x2 complex matrix multiplication.
    GateBufferPtr bfr = std::make_shared<GateBuffer>(false, (const bitLenInt*)NULL, 0, mtrx);
    if (!(bfr->Combinable(bitBuffers[qubitIndex]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(qubitIndex);
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[qubitIndex] = bfr->LeftRightCompose(bitBuffers[qubitIndex]);
}

// Almost all additional methods, besides controlled variants of this one, just wrap operations with buffer flushes, or
// discard the buffers.

void QFusion::FlushBit(const bitLenInt& qubitIndex)
{
    // If we ended up with a buffer that's (approximately or exactly) equal to identity operator, we can discard it
    // instead of applying it.
    if (bitBuffers[qubitIndex] && bitBuffers[qubitIndex]->IsIdentity()) {
        DiscardBit(qubitIndex);
        return;
    }

    bitLenInt i;

    // Before any bit is buffered as a control, it's flushed.
    // If the bit needs to be flushed again, before buffering as a target bit, everything that depends on it as a
    // control needs to be flushed.
    for (i = 0; i < bitControls[qubitIndex].size(); i++) {
        if (bitControls[qubitIndex][i] != qubitIndex) {
            FlushBit(bitControls[qubitIndex][i]);
        }
    }
    bitControls[qubitIndex].resize(0);

    BitBufferPtr bfr = bitBuffers[qubitIndex];
    if (bfr) {
        // First, we flush this bit.
        bfr->Apply(qReg, qubitIndex, &bitBuffers);

        if (bfr->controls.size() > 0) {
            // Finally, nothing controls this bit any longer, so we remove all bitControls entries indicating that it is
            // controlled by another bit.
            std::vector<bitLenInt>::iterator found;
            bitLenInt control;
            for (i = 0; i < bfr->controls.size(); i++) {
                control = bfr->controls[i];
                found = std::find(bitControls[control].begin(), bitControls[control].end(), qubitIndex);
                if (found != bitControls[control].end()) {
                    bitControls[control].erase(found);
                }
            }
        }
    }
}

void QFusion::DiscardBit(const bitLenInt& qubitIndex)
{
    BitBufferPtr bfr = bitBuffers[qubitIndex];
    if (bfr) {
        // If this is an arithmetic buffer, it has side-effects for other bits.
        if (bfr->isArithmetic) {
            // In this branch, we definitely have an ArithmeticBuffer, so it's safe to cast.
            if (bfr->IsIdentity()) {
                // If the buffer is adding 0, we can throw it away.
                ArithmeticBuffer* aBfr = dynamic_cast<ArithmeticBuffer*>(bfr.get());
                for (bitLenInt i = 0; i < (aBfr->length); i++) {
                    bitBuffers[aBfr->start + i] = NULL;
                }
            } else {
                // If the buffer is adding or subtracting a nonzero value, it has side-effects for other bits.
                FlushBit(qubitIndex);
                return;
            }
        }
        // If we are discarding this bit, it is no longer controlled by any other bit.
        std::vector<bitLenInt>::iterator found;
        bitLenInt control;
        for (bitLenInt i = 0; i < bfr->controls.size(); i++) {
            control = bfr->controls[i];
            found = std::find(bitControls[control].begin(), bitControls[control].end(), qubitIndex);
            if (found != bitControls[control].end()) {
                bitControls[control].erase(found);
            }
        }
    }
    bitBuffers[qubitIndex] = NULL;
}

void QFusion::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushList(controls, controlLen);

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen)) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    GateBufferPtr bfr = std::make_shared<GateBuffer>(false, controls, controlLen, mtrx);
    if (!(bfr->Combinable(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    // We record that this bit is controlled by the bits in its control list.
    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].push_back(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushList(controls, controlLen);

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen)) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    GateBufferPtr bfr = std::make_shared<GateBuffer>(true, controls, controlLen, mtrx);
    if (!(bfr->Combinable(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    // We record that this bit is controlled by the bits in its control list.
    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].push_back(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    FlushAll();
    qReg->UniformlyControlledSingleBit(
        controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
}

// "Compose" will increase the cost of application of every currently buffered gate by a factor of 2 per "composed"
// qubit, so it's most likely cheaper just to FlushAll() immediately.
bitLenInt QFusion::Compose(QFusionPtr toCopy)
{
    FlushAll();
    toCopy->FlushAll();
    bitLenInt toRet = qReg->Compose(toCopy->qReg);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

bitLenInt QFusion::Compose(QFusionPtr toCopy, bitLenInt start)
{
    FlushAll();
    toCopy->FlushAll();
    bitLenInt toRet = qReg->Compose(toCopy->qReg, start);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

// "Decompose" will reduce the cost of application of every currently buffered gate a by a factor of 2 per "decompose"
// qubit, so it's definitely cheaper to maintain our buffers until after the Decompose.
void QFusion::Decompose(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    FlushReg(start, length);
    dest->FlushReg(0, length);

    qReg->Decompose(start, length, dest->qReg);

    if (length < qubitCount) {
        bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    }
    SetQubitCount(qReg->GetQubitCount());
    dest->SetQubitCount(length);

    // If the Decompose caused us to fall below the MIN_FUSION_BITS threshold, this is the cheapest buffer application
    // gets:
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
    if (dest->GetQubitCount() < MIN_FUSION_BITS) {
        dest->FlushAll();
    }
}

// "Dispose" will reduce the cost of application of every currently buffered gate a by a factor of 2 per "disposed"
// qubit, so it's definitely cheaper to maintain our buffers until after the Dispose.
void QFusion::Dispose(bitLenInt start, bitLenInt length)
{
    DiscardReg(start, length);
    qReg->Dispose(start, length);

    // Since we're disposing bits, (and since we assume that the programmer knows that they're separable before calling
    // "Dispose,") we can just throw the corresponding buffers away:
    if (length < qubitCount) {
        bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    }

    // If the Dispose caused us to fall below the MIN_FUSION_BITS threshold, this is the cheapest buffer application
    // gets:
    SetQubitCount(qReg->GetQubitCount());
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
}

// "TryDecompose" will reduce the cost of application of every currently buffered gate a by a factor of 2 per
// "decomposed" qubit, so it's definitely cheaper to maintain our buffers until after the Decomposed.
bool QFusion::TryDecompose(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    FlushReg(start, length);

    bool result = qReg->TryDecompose(start, length, dest->qReg);

    if (result == false) {
        return false;
    }

    if (length < qubitCount) {
        bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    }
    SetQubitCount(qReg->GetQubitCount());
    dest->SetQubitCount(length);

    // If the Decompose caused us to fall below the MIN_FUSION_BITS threshold, this is the cheapest buffer application
    // gets:
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
    }
    if (dest->GetQubitCount() < MIN_FUSION_BITS) {
        dest->FlushAll();
    }

    return true;
}

// "PhaseFlip" can be buffered as a single bit operation to make it cheaper, (equivalent to the application of the gates
// Z X Z X to any given bit, for example).
void QFusion::PhaseFlip()
{
    // If we're below the buffering threshold, direct application is cheaper.
    if (qubitCount < MIN_FUSION_BITS) {
        FlushAll();
        qReg->PhaseFlip();
        return;
    }

    // We buffer the phase flip as a single bit operation in bit 0.
    complex pfm[4] = { complex(-ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(-ONE_R1, ZERO_R1) };
    ApplySingleBit(pfm, false, 0);
}

// Every other operation just wraps the QEngine with the appropriate buffer flushes.
void QFusion::SetQuantumState(const complex* inputState)
{
    DiscardAll();
    qReg->SetQuantumState(inputState);
}

void QFusion::GetQuantumState(complex* outputState)
{
    FlushAll();
    qReg->GetQuantumState(outputState);
}

void QFusion::GetProbs(real1* outputProbs)
{
    FlushAll();
    qReg->GetProbs(outputProbs);
}

complex QFusion::GetAmplitude(bitCapInt perm)
{
    FlushAll();
    return qReg->GetAmplitude(perm);
}

void QFusion::SetPermutation(bitCapInt perm, complex phaseFac)
{
    DiscardAll();
    qReg->SetPermutation(perm, phaseFac);
}

void QFusion::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    DiscardReg(start, length);
    qReg->SetReg(start, length, value);
}

void QFusion::SetBit(bitLenInt qubitIndex, bool value)
{
    DiscardBit(qubitIndex);
    qReg->SetBit(qubitIndex, value);
}

void QFusion::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CSqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->CISqrtSwap(controls, controlLen, qubit1, qubit2);
}

void QFusion::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushList(controls, controlLen);
    FlushBit(qubit1);
    FlushBit(qubit2);
    qReg->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
}

bool QFusion::ForceM(bitLenInt qubit, bool result, bool doForce)
{
    FlushAll();
    return qReg->ForceM(qubit, result, doForce);
}

bitCapInt QFusion::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values)
{
    FlushAll();
    return qReg->ForceM(bits, length, values);
}

bitCapInt QFusion::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce)
{
    FlushAll();
    return qReg->ForceMReg(start, length, result, doForce);
}

void QFusion::BufferArithmetic(
    bitLenInt* controls, bitLenInt controlLen, int toAdd, bitLenInt inOutStart, bitLenInt length)
{
    // We can fuse arithmetic, but this does not necessarily commute with nonarithmetic gates.
    // We must flush the bit buffers, if they aren't arithmetic buffers.

    bitLenInt i;

    FlushList(controls, controlLen);

    BitBufferPtr toCheck;
    BitBufferPtr bfr = std::make_shared<ArithmeticBuffer>(false, controls, controlLen, inOutStart, length, toAdd);

    for (i = 0; i < length; i++) {
        toCheck = bitBuffers[inOutStart + i];
        // "Combinable" checks whether two buffers can be combined, including gate vs. arithmetic types.
        if (!(bfr->Combinable(toCheck))) {
            FlushReg(inOutStart, length);
            break;
        }
    }

    toCheck = bitBuffers[inOutStart];

    // After the buffers have been compared with "Combinable," it's safe to assume the old buffer is an
    // ArithmeticBuffer.
    BitBufferPtr nBfr = bfr->LeftRightCompose(toCheck);
    for (i = 0; i < length; i++) {
        bitBuffers[inOutStart + i] = nBfr;
    }

    if (toCheck == NULL) {
        for (i = 0; i < controlLen; i++) {
            bitControls[controls[i]].push_back(inOutStart);
        }
    }
}

void QFusion::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    if (MODLEN(toAdd, length)) {
        BufferArithmetic(NULL, 0, toAdd, start, length);
    }
}

void QFusion::CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (MODLEN(toAdd, length)) {
        BufferArithmetic(controls, controlLen, toAdd, inOutStart, length);
    }
}

void QFusion::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->INCC(toAdd, start, length, carryIndex);
}

void QFusion::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    if (MODLEN(toAdd, length)) {
        FlushReg(start, length);
        FlushBit(overflowIndex);
        qReg->INCS(toAdd, start, length, overflowIndex);
    }
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
    if (MODLEN(toAdd, length)) {
        FlushReg(start, length);
        qReg->INCBCD(toAdd, start, length);
    }
}

void QFusion::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->INCBCDC(toAdd, start, length, carryIndex);
}

void QFusion::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    if (MODLEN(toSub, length)) {
        BufferArithmetic(NULL, 0, -toSub, start, length);
    }
}

void QFusion::CDEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (MODLEN(toSub, length)) {
        BufferArithmetic(controls, controlLen, -toSub, inOutStart, length);
    }
}

void QFusion::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->DECC(toSub, start, length, carryIndex);
}

void QFusion::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    if (MODLEN(toSub, length)) {
        FlushReg(start, length);
        FlushBit(overflowIndex);
        qReg->DECS(toSub, start, length, overflowIndex);
    }
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
    if (MODLEN(toSub, length)) {
        FlushReg(start, length);
        qReg->DECBCD(toSub, start, length);
    }
}

void QFusion::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    FlushReg(start, length);
    FlushBit(carryIndex);
    qReg->DECBCDC(toSub, start, length, carryIndex);
}

void QFusion::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toMul == 0U) {
        SetReg(inOutStart, length, 0U);
        SetReg(carryStart, length, 0U);
    } else if (toMul > 1U) {
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->MUL(toMul, inOutStart, carryStart, length);
    }
}

void QFusion::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv != 1U) {
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->DIV(toDiv, inOutStart, carryStart, length);
    }
}

void QFusion::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->MULModNOut(toMul, modN, inStart, outStart, length);
}

void QFusion::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->POWModNOut(base, modN, inStart, outStart, length);
}

void QFusion::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    if (toMul != 1U) {
        FlushList(controls, controlLen);
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
}

void QFusion::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    if (toDiv != 1U) {
        FlushList(controls, controlLen);
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
}

void QFusion::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
}

void QFusion::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    FlushList(controls, controlLen);
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
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

void QFusion::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    std::rotate(bitBuffers.begin() + start, bitBuffers.begin() + start + shift, bitBuffers.begin() + start + length);
    qReg->ROL(shift, start, length);
}

void QFusion::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    if (qubitIndex1 != qubitIndex2) {
        std::swap(bitBuffers[qubitIndex1], bitBuffers[qubitIndex2]);
        qReg->Swap(qubitIndex1, qubitIndex2);
    }
}

void QFusion::SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    if (qubitIndex1 != qubitIndex2) {
        FlushBit(qubitIndex1);
        FlushBit(qubitIndex2);
        qReg->SqrtSwap(qubitIndex1, qubitIndex2);
    }
}

void QFusion::ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    if (qubitIndex1 != qubitIndex2) {
        FlushBit(qubitIndex1);
        FlushBit(qubitIndex2);
        qReg->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
}

void QFusion::CopyState(QFusionPtr orig)
{
    FlushAll();
    orig->FlushAll();
    qReg->CopyState(orig->qReg);
}

real1 QFusion::Prob(bitLenInt qubitIndex)
{
    FlushBit(qubitIndex);
    return qReg->Prob(qubitIndex);
}

real1 QFusion::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    FlushReg(start, length);
    return qReg->ProbReg(start, length, permutation);
}

real1 QFusion::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    FlushMask(mask);
    return qReg->ProbMask(mask, permutation);
}

real1 QFusion::ProbAll(bitCapInt fullRegister)
{
    FlushAll();
    return qReg->ProbAll(fullRegister);
}

bool QFusion::ApproxCompare(QFusionPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    // Make sure all buffers are flushed before comparison
    FlushAll();
    toCompare->FlushAll();

    // Compare the wrapped objects
    return qReg->ApproxCompare(toCompare->qReg);
}

// Avoid calling this, when a QFusion layer is being used:
void QFusion::UpdateRunningNorm() { qReg->UpdateRunningNorm(); }

// Avoid calling this, when a QFusion layer is being used:
void QFusion::NormalizeState(real1 nrm) { qReg->NormalizeState(nrm); }

bool QFusion::TrySeparate(bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    return qReg->TrySeparate(start, length);
}

} // namespace Qrack
