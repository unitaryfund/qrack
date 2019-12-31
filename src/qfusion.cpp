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
    : QInterface(qBitCount, rgp, deviceID, useHardwareRNG, randomGlobalPhase)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , bitBuffers(qBitCount)
    , bitControls(qBitCount)
{
    qReg = CreateQuantumInterface(eng, qBitCount, initState, rgp, phaseFactor, doNormalize, randomGlobalPhase,
        useHostMem, deviceID, useHardwareRNG, useSparseStateVec);
}

QFusion::QFusion(QInterfacePtr target)
    : QInterface(target->GetQubitCount())
    , phaseFactor(complex(-999.0, -999.0))
    , doNormalize(true)
    , bitBuffers(target->GetQubitCount())
    , bitControls(target->GetQubitCount())
{
    qReg = target;
}

/**
 * All buffering operations happen in ApplySingleBit and its variants, which underly the application of all 2x2 complex
 * element matrix operators, such as H, X, Y, Z, RX, etc..
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

void QFusion::EraseControls(std::vector<bitLenInt> controls, bitLenInt qubitIndex)
{
    for (bitLenInt i = 0; i < controls.size(); i++) {
        bitControls[controls[i]].erase(qubitIndex);
    }
}

void QFusion::FlushBit(const bitLenInt& qubitIndex)
{
    BitBufferPtr bfr = bitBuffers[qubitIndex];

    if (!bfr) {
        return;
    }

    // If we ended up with a buffer that's (approximately or exactly) equal to identity operator, we can discard it
    // instead of applying it.
    if (bfr->IsIdentity()) {
        DiscardBit(qubitIndex);
        return;
    }

    std::vector<bitLenInt> controls = bfr->controls;

    // First, we flush this bit.
    bfr->Apply(qReg, qubitIndex, &bitBuffers);
    // Finally, nothing controls this bit any longer, so we remove all bitControls entries indicating that it is
    // controlled by another bit.
    EraseControls(controls, qubitIndex);
}

void QFusion::DiscardBit(const bitLenInt& qubitIndex)
{
    BitBufferPtr bfr = bitBuffers[qubitIndex];

    if (!bfr) {
        return;
    }

    // Only discard if this operator doesn't control anything or is the identity operator
    if ((bitControls[qubitIndex].size() > 0) && !(bfr->IsIdentity())) {
        FlushBit(qubitIndex);
        return;
    }

    // If this is an arithmetic buffer, it has side-effects for other bits.
    if (bfr->isArithmetic) {
        // In this branch, we definitely have an ArithmeticBuffer, so it's safe to cast.
        if (bfr->IsIdentity()) {
            // If the buffer is adding 0, we can throw it away.
            ArithmeticBuffer* aBfr = dynamic_cast<ArithmeticBuffer*>(bfr.get());
            std::vector<bitLenInt> controls = aBfr->controls;
            for (bitLenInt i = 0; i < (aBfr->length); i++) {
                bitBuffers[aBfr->start + i] = NULL;
                EraseControls(bfr->controls, qubitIndex);
            }
        } else {
            // If the buffer is adding or subtracting a nonzero value, it has side-effects for other bits.
            FlushBit(qubitIndex);
        }
        return;
    }

    // If we are discarding this bit, it is no longer controlled by any other bit.
    EraseControls(bfr->controls, qubitIndex);
    bitBuffers[qubitIndex] = NULL;
}

// Almost all additional methods, besides controlled variants of this one, just wrap operations with buffer flushes, or
// discard the buffers.
void QFusion::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex)
{
    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates.
    if (qubitCount < (MIN_FUSION_BITS + ((bitBuffers[qubitIndex] == NULL) ? 0 : 1))) {
        // Directly apply the gate and return.
        FlushBit(qubitIndex);
        qReg->ApplySingleBit(mtrx, doCalcNorm, qubitIndex);
        return;
    }

    FlushSet(bitControls[qubitIndex]);

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
void QFusion::ApplySinglePhase(const complex topLeft, const complex bottomRight, bool doCalcNorm, bitLenInt qubitIndex)
{
    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates.
    if (qubitCount < (MIN_FUSION_BITS + ((bitBuffers[qubitIndex] == NULL) ? 0 : 1))) {
        // Directly apply the gate and return.
        FlushBit(qubitIndex);
        qReg->ApplySinglePhase(topLeft, bottomRight, doCalcNorm, qubitIndex);
        return;
    }

    // Unlike the general single bit variant, phase gates definitely commute with control bits, so there's no need to
    // flush this bit as a control.

    // If we pass the threshold number of qubits for buffering, we just do 2x2 complex matrix multiplication.
    complex mtrx[4] = { topLeft, 0, 0, bottomRight };
    GateBufferPtr bfr = std::make_shared<GateBuffer>(false, (const bitLenInt*)NULL, 0, mtrx);
    if (!(bfr->Combinable(bitBuffers[qubitIndex]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(qubitIndex);
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[qubitIndex] = bfr->LeftRightCompose(bitBuffers[qubitIndex]);
}

void QFusion::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushArray(controls, controlLen);
    FlushSet(bitControls[target]);

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen + ((bitBuffers[target] == NULL) ? 0 : 1))) {
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
            bitControls[controls[i]].insert(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    FlushArray(controls, controlLen);
    // Unlike the general single bit variant, phase gates definitely commute with control bits, so there's no need to
    // flush this bit as a control.

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen + ((bitBuffers[target] == NULL) ? 0 : 1))) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
        return;
    }

    complex mtrx[4] = { topLeft, 0, 0, bottomRight };
    GateBufferPtr bfr = std::make_shared<GateBuffer>(false, controls, controlLen, mtrx);
    if (!(bfr->Combinable(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    // We record that this bit is controlled by the bits in its control list.
    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].insert(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    FlushArray(controls, controlLen);

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen + ((bitBuffers[target] == NULL) ? 0 : 1))) {
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
            bitControls[controls[i]].insert(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    FlushArray(controls, controlLen);
    // Unlike the general single bit variant, phase gates definitely commute with control bits, so there's no need to
    // flush this bit as a control.

    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen + ((bitBuffers[target] == NULL) ? 0 : 1))) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyAntiControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
        return;
    }

    complex mtrx[4] = { topLeft, 0, 0, bottomRight };
    GateBufferPtr bfr = std::make_shared<GateBuffer>(true, controls, controlLen, mtrx);
    if (!(bfr->Combinable(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    // We record that this bit is controlled by the bits in its control list.
    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].insert(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    bitBuffers[target] = bfr->LeftRightCompose(bitBuffers[target]);
}

void QFusion::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt target,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    FlushArray(controls, controlLen);
    FlushBit(target);
    qReg->UniformlyControlledSingleBit(
        controls, controlLen, target, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
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
// qubit, so it's probably cheaper to maintain our buffers until after the Decompose. However, this requires re-indexing
// cached gates, to account for the changes in bit indices at the tail past the decomposed segment.
void QFusion::Decompose(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    if (length == 0) {
        return;
    }

    FlushAll();
    dest->FlushAll();
    qReg->Decompose(start, length, dest->qReg);
    SetQubitCount(qReg->GetQubitCount());
    dest->SetQubitCount(dest->GetQubitCount());
}

// "Dispose" will reduce the cost of application of every currently buffered gate a by a factor of 2 per "disposed"
// qubit, so it's probably cheaper to maintain our buffers until after the Dispose. However, this requires re-indexing
// cached gates, to account for the changes in bit indices at the tail past the disposed segment.
void QFusion::Dispose(bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    FlushAll();
    qReg->Dispose(start, length);
    SetQubitCount(qReg->GetQubitCount());
}

bool QFusion::TryDecompose(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    FlushAll();
    return qReg->TryDecompose(start, length, dest->qReg);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);
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

    FlushArray(controls, controlLen);

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
            bitControls[controls[i]].insert(inOutStart);
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
        DiscardReg(inOutStart, length);
        DiscardReg(carryStart, length);
        SetReg(inOutStart, length, 0U);
        SetReg(carryStart, length, 0U);
    } else if (toMul > ONE_BCI) {
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->MUL(toMul, inOutStart, carryStart, length);
    }
}

void QFusion::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv != ONE_BCI) {
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

void QFusion::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->IMULModNOut(toMul, modN, inStart, outStart, length);
}

void QFusion::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->POWModNOut(base, modN, inStart, outStart, length);
}

void QFusion::FullAdd(bitLenInt input1, bitLenInt input2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FlushBit(input1);
    FlushBit(input2);
    FlushBit(carryInSumOut);
    FlushBit(carryOut);
    qReg->FullAdd(input1, input2, carryInSumOut, carryOut);
}

void QFusion::IFullAdd(bitLenInt input1, bitLenInt input2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FlushBit(input1);
    FlushBit(input2);
    FlushBit(carryInSumOut);
    FlushBit(carryOut);
    qReg->IFullAdd(input1, input2, carryInSumOut, carryOut);
}

void QFusion::CFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FlushArray(controls, controlLen);
    FlushBit(input1);
    FlushBit(input2);
    FlushBit(carryInSumOut);
    FlushBit(carryOut);
    qReg->CFullAdd(controls, controlLen, input1, input2, carryInSumOut, carryOut);
}

void QFusion::CIFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FlushArray(controls, controlLen);
    FlushBit(input1);
    FlushBit(input2);
    FlushBit(carryInSumOut);
    FlushBit(carryOut);
    qReg->CIFullAdd(controls, controlLen, input1, input2, carryInSumOut, carryOut);
}

void QFusion::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    if (toMul != ONE_BCI) {
        FlushArray(controls, controlLen);
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
}

void QFusion::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    if (toDiv != ONE_BCI) {
        FlushArray(controls, controlLen);
        FlushReg(inOutStart, length);
        FlushReg(carryStart, length);
        qReg->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
}

void QFusion::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    FlushArray(controls, controlLen);
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
}

void QFusion::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    FlushArray(controls, controlLen);
    FlushReg(inStart, length);
    FlushReg(outStart, length);
    qReg->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
}

void QFusion::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    FlushArray(controls, controlLen);
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

void QFusion::ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    if (qubitIndex1 != qubitIndex2) {
        FlushBit(qubitIndex1);
        FlushBit(qubitIndex2);
        qReg->ISwap(qubitIndex1, qubitIndex2);
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
void QFusion::NormalizeState(real1 nrm, real1 norm_thresh) { qReg->NormalizeState(nrm, norm_thresh); }

bool QFusion::TrySeparate(bitLenInt start, bitLenInt length)
{
    FlushReg(start, length);
    return qReg->TrySeparate(start, length);
}

} // namespace Qrack
