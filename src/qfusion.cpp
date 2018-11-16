//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License VMIN_FUSION_BITS.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <ctime>
#include <future>
#include <initializer_list>
#include <map>

#include "qfactory.hpp"
#include "qfusion.hpp"

namespace Qrack {

QFusion::QFusion(
    QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
    , engineType(eng)
    , bitBuffers(qBitCount)
    , bitControls(qBitCount)
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
 * the same bit is cheaper with "gate fusion," (M-1)*8+2^(N+1) multiplications for M gates instead of 2^(M+N+1)
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

    BitBufferPtr bfr = std::make_shared<BitBuffer>(false, (const bitLenInt*)NULL, 0, mtrx);
    if (!(bfr->CompareControls(bitBuffers[qubitIndex]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(qubitIndex);
    }

    // Now, we're going to chain our buffered gates;
    BitOp inBuffer(new complex[4]);
    std::copy(mtrx, mtrx + 4, inBuffer.get());
    bfr->matrix = Mul2x2(inBuffer, bitBuffers[qubitIndex] == NULL ? NULL : bitBuffers[qubitIndex]->matrix);
    bitBuffers[qubitIndex] = bfr;

    // Almost all additional methods just wrap operations with buffer flushes, or discard the buffers.
}

BitOp QFusion::Mul2x2(BitOp left, BitOp right)
{
    // If we pass the threshold number of qubits for buffering, we just do 2x2 complex matrix multiplication.
    // We parallelize this, since we can.
    // If a matrix component is very close to zero, we assume it's floating-point-error on a composition that has an
    // exactly 0 component, number theoretically. (If it's not exactly 0 by number theory, it's numerically negligible,
    // and we're safe.)

    BitOp outBuffer(new complex[4]);

    if (right) {
        std::vector<std::future<void>> futures(4);

        futures[0] = std::async(std::launch::async, [&]() {
            outBuffer[0] = (left[0] * right[0]) + (left[1] * right[2]);
            if (norm(outBuffer[0]) < min_norm) {
                outBuffer[0] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[1] = std::async(std::launch::async, [&]() {
            outBuffer[1] = (left[0] * right[1]) + (left[1] * right[3]);
            if (norm(outBuffer[1]) < min_norm) {
                outBuffer[1] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[2] = std::async(std::launch::async, [&]() {
            outBuffer[2] = (left[2] * right[0]) + (left[3] * right[2]);
            if (norm(outBuffer[2]) < min_norm) {
                outBuffer[2] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[3] = std::async(std::launch::async, [&]() {
            outBuffer[3] = (left[2] * right[1]) + (left[3] * right[3]);
            if (norm(outBuffer[3]) < min_norm) {
                outBuffer[3] = complex(ZERO_R1, ZERO_R1);
            }
        });

        for (int i = 0; i < 4; i++) {
            futures[i].get();
        }
    } else {
        std::copy(left.get(), left.get() + 4, outBuffer.get());
    }

    return outBuffer;
}

void QFusion::FlushBit(const bitLenInt& qubitIndex)
{
    bitLenInt i;

    for (i = 0; i < bitControls[qubitIndex].size(); i++) {
        FlushBit(bitControls[qubitIndex][i]);
    }
    bitControls[qubitIndex].resize(0);

    BitBufferPtr bfr = bitBuffers[qubitIndex];
    if (bfr) {
        if (bfr->controls.size() == 0) {
            qReg->ApplySingleBit(bfr->matrix.get(), true, qubitIndex);
        } else {
            bitLenInt* ctrls = new bitLenInt[bfr->controls.size()];
            std::copy(bfr->controls.begin(), bfr->controls.end(), ctrls);

            if (bfr->anti) {
                qReg->ApplyAntiControlledSingleBit(ctrls, bfr->controls.size(), qubitIndex, bfr->matrix.get());
            } else {
                qReg->ApplyControlledSingleBit(ctrls, bfr->controls.size(), qubitIndex, bfr->matrix.get());
            }

            delete[] ctrls;

            // Finally, nothing controls this bit any longer:
            std::vector<bitLenInt>::iterator found;
            bitLenInt control;
            for (i = 0; i < bfr->controls.size(); i++) {
                control = bfr->controls[i];
                found = std::find(bitControls[control].begin(), bitControls[control].end(), control);
                if (found != bitControls[control].end()) {
                    bitControls[control].erase(found);
                }
            }
        }
        bitBuffers[qubitIndex] = NULL;
    }
}

void QFusion::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen)) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    // If we pass the threshold number of qubits for buffering, we track the buffered control bits, and we do 2x2
    // complex matrix multiplication.

    for (bitLenInt i = 0; i < controlLen; i++) {
        FlushBit(controls[i]);
        bitControls[controls[i]].push_back(target);
    }

    BitBufferPtr bfr = std::make_shared<BitBuffer>(false, controls, controlLen, mtrx);
    if (!(bfr->CompareControls(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].push_back(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    BitOp outMatrix = Mul2x2(bfr->matrix, bitBuffers[target] == NULL ? NULL : bitBuffers[target]->matrix);
    bfr->matrix = outMatrix;
    bitBuffers[target] = bfr;
}

void QFusion::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    // MIN_FUSION_BITS might be 3 qubits, or more. If there are only 1 or 2 qubits in a QEngine, buffering is definitely
    // more expensive than directly applying the gates. Each control bit reduces the complexity by a factor of two, and
    // buffering is only efficient if we have one additional total bit for each additional control bit to buffer.
    if (qubitCount < (MIN_FUSION_BITS + controlLen)) {
        // Directly apply the gate and return.
        FlushBit(target);
        qReg->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    // If we pass the threshold number of qubits for buffering, we track the buffered control bits, and we do 2x2
    // complex matrix multiplication.

    for (bitLenInt i = 0; i < controlLen; i++) {
        FlushBit(controls[i]);
        bitControls[controls[i]].push_back(target);
    }

    BitBufferPtr bfr = std::make_shared<BitBuffer>(true, controls, controlLen, mtrx);
    if (!(bfr->CompareControls(bitBuffers[target]))) {
        // Flush the old buffer, if the buffered control bits don't match.
        FlushBit(target);
    }

    if (bitBuffers[target] == NULL) {
        for (bitLenInt i = 0; i < controlLen; i++) {
            bitControls[controls[i]].push_back(target);
        }
    }

    // Now, we're going to chain our buffered gates;
    BitOp outMatrix = Mul2x2(bfr->matrix, bitBuffers[target] == NULL ? NULL : bitBuffers[target]->matrix);
    bfr->matrix = outMatrix;
    bitBuffers[target] = bfr;
}

// "Cohere" will increase the cost of application of every currently buffered gate by a factor of 2 per "cohered" qubit,
// so it's most likely cheaper just to FlushAll() immediately.
bitLenInt QFusion::Cohere(QFusionPtr toCopy)
{
    FlushAll();
    toCopy->FlushAll();
    bitLenInt toRet = qReg->Cohere(toCopy->qReg);
    SetQubitCount(qReg->GetQubitCount());
    return toRet;
}

// "Decohere" will reduce the cost of application of every currently buffered gate a by a factor of 2 per "decohered"
// qubit, so it's definitely cheaper to maintain our buffers until after the Decohere.
void QFusion::Decohere(bitLenInt start, bitLenInt length, QFusionPtr dest)
{
    FlushReg(start, length);

    qReg->Decohere(start, length, dest->qReg);

    if (length < qubitCount) {
        bitBuffers.erase(bitBuffers.begin() + start, bitBuffers.begin() + start + length);
    }
    SetQubitCount(qReg->GetQubitCount());
    dest->SetQubitCount(length);

    // If the Decohere caused us to fall below the MIN_FUSION_BITS threshold, this is the cheapest buffer application
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

// "PhaseFlip" can be buffered as a single bit operation to make it cheaper, (equivalent to the application of the gates
// Z X Z X to any given bit, for example).
void QFusion::PhaseFlip()
{
    FlushAll();

    // If we're below the buffering threshold, direct application is cheaper.
    if (qubitCount < MIN_FUSION_BITS) {
        qReg->PhaseFlip();
        return;
    }

    complex pfm[4] = { complex(-ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(-ONE_R1, ZERO_R1) };
    ApplySingleBit(pfm, false, 0);
}

// Every other operation just wraps the QEngine with the appropriate buffer flushes.
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

bool QFusion::ForceM(bitLenInt qubit, bool result, bool doForce, real1 nrmlzr)
{
    FlushAll();
    return qReg->ForceM(qubit, result, doForce, nrmlzr);
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

bool QFusion::IsPhaseSeparable(bool forceCheck)
{
    FlushAll();
    return qReg->IsPhaseSeparable(forceCheck);
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
} // namespace Qrack
