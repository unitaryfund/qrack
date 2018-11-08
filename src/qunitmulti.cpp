//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qunitmulti.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QUnit(QINTERFACE_OPENCL, qBitCount, initState, rgp)
{
    deviceCount = OCLEngine::Instance()->GetDeviceCount();
    defaultDeviceID = OCLEngine::Instance()->GetDefaultDeviceID();

    deviceIDs.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        deviceIDs[i] = i;
    }
    if (defaultDeviceID > 0) {
        std::swap(deviceIDs[0], deviceIDs[defaultDeviceID]);
    }

    RedistributeQEngines();
}

void QUnitMulti::RedistributeQEngines()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    bitCapInt totSize = 0;
    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            totSize += 1U << ((shard.unit)->GetQubitCount());
            qips.push_back(shard.unit);
        }
    }

    bitCapInt partSize = 0;
    int devicesLeft = deviceCount;
    for (bitLenInt i = 0; i < qips.size(); i++) {
        partSize += 1U << (qips[i]->GetQubitCount());
        if (partSize >= (totSize / devicesLeft)) {
            (dynamic_cast<QEngineOCL*>(qips[i].get()))->SetDevice(deviceIDs[deviceCount - devicesLeft]);
            partSize = 0;
            if (devicesLeft > 1) {
                devicesLeft--;
            }
        }
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

template <class It> QInterfacePtr QUnitMulti::EntangleIterator(It first, It last)
{
    QInterfacePtr toRet = QUnit::EntangleIterator(first, last);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(bitLenInt start, bitLenInt length)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start, length);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start1, length1, start2, length2);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(
    bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start1, length1, start2, length2, start3, length3);
    RedistributeQEngines();
    return toRet;
}

bool QUnitMulti::TrySeparate(std::vector<bitLenInt> bits)
{
    bool didSeparate = QUnit::TrySeparate(bits);
    if (didSeparate) {
        RedistributeQEngines();
    }
    return didSeparate;
}

/// Set register bits to given permutation
void QUnitMulti::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        shards[bit + start].unit->SetPermutation((value & (1 << bit)) > 0 ? 1 : 0);
    });
}

// Bit-wise apply measurement gate to a register
bitCapInt QUnitMulti::MReg(bitLenInt start, bitLenInt length)
{
    // Measurement introduces an overall phase shift. Since it is applied to every state, this will not change the
    // status of our cached knowledge of phase separability. However, measurement could set some amplitudes to zero,
    // meaning the relative amplitude phases might only become separable in the process if they are not already.
    if (knowIsPhaseSeparable && (!isPhaseSeparable)) {
        knowIsPhaseSeparable = false;
    }

    int numCores = GetConcurrencyLevel();

    bitCapInt* partResults = new bitCapInt[numCores]();

    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { partResults[cpu] |= M(start + bit) ? (1 << bit) : 0; });

    bitCapInt result = 0;
    for (int i = 0; i < numCores; i++) {
        result |= partResults[i];
    }

    return result;
}

// Bit-wise apply swap to two registers
void QUnitMulti::Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Swap(qubit1 + bit, qubit2 + bit); });
}

// Bit-wise apply square root of swap to two registers
void QUnitMulti::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { SqrtSwap(qubit1 + bit, qubit2 + bit); });
}

// Bit-wise apply inverse of square root of swap to two registers
void QUnitMulti::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ISqrtSwap(qubit1 + bit, qubit2 + bit); });
}

// Bit-wise apply "anti-"controlled-not to three registers
void QUnitMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { AntiCCNOT(control1 + bit, control2 + bit, target + bit); });
}

void QUnitMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CCNOT(control1 + bit, control2 + bit, target + bit); });
}

void QUnitMulti::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { AntiCNOT(control + bit, target + bit); });
}

void QUnitMulti::CNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CNOT(control + bit, target + bit); });
}

// Apply S gate (1/8 phase rotation) to each bit in "length," starting from bit index "start"
void QUnitMulti::S(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { S(start + bit); });
}

// Apply inverse S gate (1/8 phase rotation) to each bit in "length," starting from bit index "start"
void QUnitMulti::IS(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { IS(start + bit); });
}

// Apply T gate (1/8 phase rotation) to each bit in "length," starting from bit index "start"
void QUnitMulti::T(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { T(start + bit); });
}

// Apply T gate (1/8 phase rotation) to each bit in "length," starting from bit index "start"
void QUnitMulti::IT(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { IT(start + bit); });
}

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QUnitMulti::X(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { X(start + bit); });
}

// Single register instructions:

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
void QUnitMulti::H(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { H(start + bit); });
}

/// Apply Pauli Y matrix to each bit
void QUnitMulti::Y(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Y(start + bit); });
}

/// Apply Pauli Z matrix to each bit
void QUnitMulti::Z(bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Z(start + bit); });
}

/// Apply controlled Pauli Y matrix to each bit
void QUnitMulti::CY(bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CY(control + bit, target + bit); });
}

/// Apply controlled Pauli Z matrix to each bit
void QUnitMulti::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CZ(control + bit, target + bit); });
}

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLAND(qInputStart + bit, cBit, outputStart + bit);
    });
}

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLOR(qInputStart + bit, cBit, outputStart + bit);
    });
}

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLXOR(qInputStart + bit, cBit, outputStart + bit);
    });
}

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
void QUnitMulti::RT(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RT(radians, start + bit); });
}

/**
 * Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QUnitMulti::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RTDyad(numerator, denominator, start + bit); });
}

/**
 * Bitwise (identity) exponentiation gate - Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
 */
void QUnitMulti::Exp(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Exp(radians, start + bit); });
}

/**
 * Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
 * exponentiation of the identity operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QUnitMulti::ExpDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpDyad(numerator, denominator, start + bit); });
}

/**
 * Bitwise Pauli X exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
 */
void QUnitMulti::ExpX(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpX(radians, start + bit); });
}

/**
 * Dyadic fraction Pauli X exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_x / 2^denomPower} \f$,
 * exponentiation of the Pauli X operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QUnitMulti::ExpXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpXDyad(numerator, denominator, start + bit); });
}

/**
 * Bitwise Pauli Y exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
 */
void QUnitMulti::ExpY(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpY(radians, start + bit); });
}

/**
 * Dyadic fraction Pauli Y exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_y / 2^denomPower} \f$,
 * exponentiation of the Pauli Y operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QUnitMulti::ExpYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpYDyad(numerator, denominator, start + bit); });
}

/**
 * Bitwise Pauli Z exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
 */
void QUnitMulti::ExpZ(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpZ(radians, start + bit); });
}

/**
 * Dyadic fraction Pauli Z exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_z / 2^denomPower} \f$,
 * exponentiation of the Pauli Z operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QUnitMulti::ExpZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpZDyad(numerator, denominator, start + bit); });
}

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
void QUnitMulti::RX(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RX(radians, start + bit); });
}

/**
 * Dyadic fraction x axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli x
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QUnitMulti::RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RXDyad(numerator, denominator, start + bit); });
}

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
void QUnitMulti::RY(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RY(radians, start + bit); });
}

/**
 * Dyadic fraction y axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QUnitMulti::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RYDyad(numerator, denominator, start + bit); });
}

/// z axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli z axis
void QUnitMulti::RZ(real1 radians, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RZ(radians, start + bit); });
}

/**
 * Dyadic fraction z axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QUnitMulti::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RZDyad(numerator, denominator, start + bit); });
}

/// Controlled "phase shift gate"
void QUnitMulti::CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRT(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction "phase shift gate"
void QUnitMulti::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(
        0, length, [&](bitLenInt bit, bitLenInt cpu) { CRTDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled x axis rotation
void QUnitMulti::CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRX(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void QUnitMulti::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(
        0, length, [&](bitLenInt bit, bitLenInt cpu) { CRXDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled y axis rotation
void QUnitMulti::CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRY(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void QUnitMulti::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(
        0, length, [&](bitLenInt bit, bitLenInt cpu) { CRYDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled z axis rotation
void QUnitMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRZ(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void QUnitMulti::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    par_for(
        0, length, [&](bitLenInt bit, bitLenInt cpu) { CRZDyad(numerator, denominator, control + bit, target + bit); });
}

} // namespace Qrack
