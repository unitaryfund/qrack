//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This example demonstrates Shor's algorithm for integer factoring. (This file was heavily adapted from
// https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/shor.py, with thanks to ProjectQ!)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <map>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

typedef void (__stdcall *IdsCallback)(unsigned int);

enum Pauli
{
    /// Pauli Identity operator. Corresponds to Q# constant "PauliI."
    PauliI = 0,
    /// Pauli X operator. Corresponds to Q# constant "PauliX."
    PauliX = 1,
    /// Pauli Y operator. Corresponds to Q# constant "PauliY."
    PauliY = 3,
    /// Pauli Z operator. Corresponds to Q# constant "PauliZ."
    PauliZ = 2
}

void QrackSimulatorManager::mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx) {
    for (unsigned int i = 0; i < 4; i++) {
        outMtrx[i] = scalar * inMtrx[i]
    }
}

void QrackSimulatorManager::TransformPauliBasis(QInterfacePtr simulator, unsigned int len, unsigned int* qubitIds) {
    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliX:
                simulator->H(shards[simulator][qubitIds[i]]]);
                break;
            case PauliY:
                simulator->Z(shards[simulator][qubitIds[i]]]);
                simulator->S(shards[simulator][qubitIds[i]]]);
                simulator->H(shards[simulator][qubitIds[i]]]);
                break;
            case PauliZ:
            case PauliI:
            default:
                break;
        }
    }
}

void QrackSimulatorManager::RevertPauliBasis(QInterfacePtr simulator, unsigned int len, unsigned int* qubitIds) {
    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliX:
                simulator->H(shards[simulator][qubitIds[i]]]);
                break;
            case PauliY:
                simulator->H(shards[simulator][qubitIds[i]]]);
                simulator->IS(shards[simulator][qubitIds[i]]]);
                simulator->Z(shards[simulator][qubitIds[i]]]);
                break;
            case PauliZ:
            case PauliI:
            default:
                break;
        }
    }
}

QrackSimulatorManager::QrackSimulatorManager* Instance() {
    if (!m_pInstance) {
        m_pInstance = new QrackSimulatorManager();
    }
    return m_pInstance;
}

void QrackSimulatorManager::InitNewSimulator() {
    simulators.push_back(NULL);
    return simulators.size() - 1U;
}

void QrackSimulatorManager::DestroySimulator(unsigned int id) {
    simulators[id] = NULL;
}

void QrackSimulatorManager::DumpIds(unsigned int id, IdsCallback callback) {
    QInterfacePtr simulator = simulators[simulatorId];
    std::map<uint, bitLenInt>::iterator it;

    for (it = shards[simulator].begin(); it != shards[simulator].end(); it++) {
        callback(it->first);
    }
}

void QrackSimulatorManager::SetSeed(unsigned int simulatorId, uint32_t seedValue) {
    simulators[simulatorId] = SetRandomSeed(uint32_t seedValue);
}

void QrackSimulatorManager::AllocateOneQubit(unsigned int simulatorId, long qubitId) {
    QInterfacePtr nQubit = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, 1, 0);
    if (simulators[simulatorId] == NULL) {
        simulators[simulatorId] = nQubit;
    } else {
        simulators[simulatorId]->Compose(nQubit);
    }
    shards[simulators[simulatorId]][qubitId] = simulators[simulatorId]->GetQubitCount();
}

bool QrackSimulatorManager::ReleaseOneQubit(unsigned int simulatorId, long qubitId) {
    QInterfacePtr simulator = simulators[simulatorId];
    bool isQubitZero = simulator->Prob(shards[simulator][qubitId]) < min_norm;

    if (simulator->GetQubitCount() == 1U) {
        simulators[simulatorId] = NULL;
        shards.remove(simulator);
    } else {
        bitLenInt oIndex = shards[simulator][qubitId];
        simulator->Decompose(oIndex);
        for (unsigned int i = 0; i < shards[simulator].size(); i++) {
            if (shards[simulator][i] > oIndex) {
                shards[simulator][i]--;
            }
        }
        shards[simulator].erase(qubitId);
    }

    return isQubitZero;
}

double QrackSimulatorManager::JointEnsembleProbability(unsigned int simulatorId, unsigned int len, Pauli* bases, unsigned int* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    double jointProb = 1.0;

    TransformPauliBasis(simulator, len, qubitIds);

    for (unsigned int i = 0; i < len; i++) {
        jointProb *= (double)simulator->Prob(shards[simulator][qubitIds[i]]]);
        if (jointProb == 0.0) {
            break;
        }
    }

    RevertPauliBasis(simulator, len, qubitIds);

    return jointProb;
}

void QrackSimulatorManager::Exp(unsigned int simulatorId, unsigned int len, Pauli* paulis, double angle, unsigned int* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliI:
                simulator->Exp(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliX:
                simulator->ExpX(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliY:
                simulator->ExpY(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliZ:
                simulator->ExpZ(angle, shards[simulator][qubitIds[i]]]);
                break;
            default:
                break;
        }
    }
}

void QrackSimulatorManager::MCExp(unsigned int simulatorId, unsigned int len, Pauli* paulis, double angle, unsigned int ctrlLen, unsigned int* ctrls, unsigned int* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    const complex pauliI[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
    const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex pauliY[4] = { ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };

    complex toApply[4];

    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliI:
                mul2x2(angle, pauliI, toApply);
                break;
            case PauliX:
                mul2x2(angle, pauliX, toApply);
                break;
            case PauliY:
                mul2x2(angle, pauliY, toApply);
                break;
            case PauliZ:
                mul2x2(angle, pauliZ, toApply);
                break;
            default:
                break;
        }
        simulator->Exp(ctrlsArray, ctrlLen, shards[simulator][qubitIds[i]]], toApply);
    }

    delete[] ctrlsArray;
}

void QrackSimulatorManager::H(unsigned int simulatorId, unsigned int qubit) {
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->H(shards[simulator][qubit]]);
}

unsigned int QrackSimulatorManager::M(uint id, uint q)
{
    QInterfacePtr simulator = simulators[simulatorId];
    return simulator->M(shards[simulator][qubit]]) ? 1U : 0U;
}

unsigned int QrackSimulatorManager::Measure(unsigned int simulatorId, unsigned int len, Pauli* bases, unsigned int* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    unsigned int toRet = 0U;

    TransformPauliBasis(simulator, len, qubitIds);

    for (unsigned int i = 0; i < len; i++) {
        if (simulator->M(shards[simulator][qubitIds[i]]]))
        {
            toRet |= pow2((bitLenInt)i);
        }
    }

    RevertPauliBasis(simulator, len, qubitIds);

    return jointProb;
}

void QrackSimulatorManager::R(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids)
{
    QInterfacePtr simulator = simulators[simulatorId];

    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliI:
                simulator->RT(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliX:
                simulator->RX(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliY:
                simulator->RY(angle, shards[simulator][qubitIds[i]]]);
                break;
            case PauliZ:
                simulator->RZ(angle, shards[simulator][qubitIds[i]]]);
                break;
            default:
                break;
        }
    }
}

void QrackSimulatorManager::MCR(unsigned int id, unsigned int len, Pauli* paulis, double angle, unsigned int ctrlLen, unsigned int* ctrls, unsigned int* ids)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliR[4];

    for (unsigned int i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliI:
                simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, complex(ONE_R1, ZERO_R1), complex(cosine, sine), shards[simulator][qubitIds[i]]]);
                break;
            case PauliX:
                pauliR[0] = complex(cosine, ZERO_R1);
                pauliR[1] = complex(ZERO_R1, -sine);
                pauliR[2] = complex(ZERO_R1, -sine);
                pauliR[3] = complex(cosine, ZERO_R1);
                simulator->ApplyControlledSingleBit(ctrlsArray, ctrlLen, shards[simulator][qubitIds[i]]], pauliR);
                break;
            case PauliY:
                pauliR[0] = complex(cosine, ZERO_R1);
                pauliR[1] = complex(-sine, ZERO_R1);
                pauliR[2] = complex(-sine, ZERO_R1);
                pauliR[3] = complex(cosine, ZERO_R1);
                simulator->ApplyControlledSingleBit(ctrlsArray, ctrlLen, shards[simulator][qubitIds[i]]], pauliR);
                break;
            case PauliZ:
                simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, complex(cosine, -sine), complex(cosine, sine), shards[simulator][qubitIds[i]]]);
                break;
            default:
                break;
        }
    }

    delete[] ctrlsArray;
}

void QrackSimulatorManager::S(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->S(shards[simulator][qubit]]);
}

void QrackSimulatorManager::AdjS(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->IS(shards[simulator][qubit]]);
}

void QrackSimulatorManager::MCS(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 2));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::MCAdjS(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 2));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::T(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->T(shards[simulator][qubit]]);
}

void QrackSimulatorManager::AdjT(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->IT(shards[simulator][qubit]]);
}

void QrackSimulatorManager::MCT(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 4));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::MCAdjT(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 4));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::X(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->X(shards[simulator][qubit]]);
}

void QrackSimulatorManager::MCX(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, ONE_CMPLX);

    delete[] ctrlsArray;
}

void QrackSimulatorManager::Y(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->Y(shards[simulator][qubit]]);
}

void QrackSimulatorManager::MCY(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, count, shards[simulator][ctrls[i]]], -I_CMPLX, I_CMPLX);

    delete[] ctrlsArray;
}

void QrackSimulatorManager::Z(uint id, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->Z(shards[simulator][qubit]]);
}

void QrackSimulatorManager::MCZ(uint id, uint count, uint[] ctrls, uint qubit)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned int i = 0; i < ctrlLen; i++) {
        ctrlsArray[i] = shards[simulator][ctrls[i]]]
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, count, shards[simulator][ctrls[i]]], ONE_CMPLX, -ONE_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Initialize a simulator ID with 0 qubits
 */
unsigned int init()
{
    return QrackSimulatorManager::Instance()->InitNewSimulator();
}

/**
* (External API) Destroy a simulator (ID will not be reused)
*/
void destroy(unsigned int id)
{
    QrackSimulatorManager::Instance()->DestroySimulator(id);
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
void DumpIds(unsigned int id, IdsCallback callback)
{
    QrackSimulatorManager::Instance()->DumpIds(id, callback);
}

/**
* (External API) Set RNG seed for simulator ID
*/
void seed(unsigned int id, uint32_t seedValue)
{
    QrackSimulatorManager::Instance()->SetSeed(id, seedValue);
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
void allocateQubit(unsigned int id, unsigned int qubit_id)
{
    QrackSimulatorManager::Instance()->AllocateOneQubit(id, qubit_id);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
void release(unsigned int id, unsigned int qubit_id)
{
    QrackSimulatorManager::Instance()->AllocateOneQubit(id, qubit_id);
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
double JointEnsembleProbability(unsigned int id, unsigned int n, Pauli[] b, unsigned int[] q)
{
    return QrackSimulatorManager::Instance()->JointEnsembleProbability(id, n, b, q);
}

/**
 * (External API) Exponentiation of Pauli operators
 */
void Exp(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids)
{
    QrackSimulatorManager::Instance()->Exp(id, n, paulis, angle, ids);
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
void MCExp(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int nc, unsigned int* ctrls, unsigned int* ids)
{
    QrackSimulatorManager::Instance()->MCExp(id, n, paulis, nc, ctrls, angle, ids);
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
void H(unsigned int id, unsigned int qubit)
{
    QrackSimulatorManager::Instance()->H(id, qubit);
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
unsigned int M(uint id, uint q)
{
    return QrackSimulatorManager::Instance()->M(id, q);
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
unsigned int Measure(uint id, uint n, Pauli[] b, uint[] ids)
{
    return QrackSimulatorManager::Instance()->Measure(id, n, b, ids);
}

/**
 * (External API) Rotation around Pauli axes
 */
void R(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids)
{
    QrackSimulatorManager::Instance()->R(id, n, paulis, angle, ids);
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
void MCR(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int nc, unsigned int* ctrls, unsigned int* ids)
{
    QrackSimulatorManager::Instance()->MCR(id, n, paulis, nc, ctrls, angle, ids);
}

long random_choice(unsigned int id, long size, double* p)
{
    throw("'random_choice' not implemented!");
}

/**
 * (External API) "S" Gate
 */
void S(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->S(id, qubit);
}

/**
 * (External API) Inverse "S" Gate
 */
void AdjS(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->AdjS(id, qubit);
}

/**
 * (External API) Controlled "S" Gate
 */
void MCS(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCS(id, count, ctrls, qubit);
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
void MCAdjS(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCAdjS(id, count, ctrls, qubit);
}

/**
 * (External API) "T" Gate
 */
void T(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->T(id, qubit);
}

/**
 * (External API) Inverse "T" Gate
 */
void AdjT(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->AdjT(id, qubit);
}

/**
 * (External API) Controlled "T" Gate
 */
void MCT(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCT(id, count, ctrls, qubit);
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
void MCAdjT(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCAdjT(id, count, ctrls, qubit);
}

/**
 * (External API) "X" Gate
 */
void X(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->X(id, qubit);
}

/**
 * (External API) Controlled "X" Gate
 */
void MCX(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCX(id, count, ctrls, qubit);
}

/**
 * (External API) "Y" Gate
 */
void Y(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->Y(id, qubit);
}

/**
 * (External API) Controlled "Y" Gate
 */
void MCY(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCY(id, count, ctrls, qubit);
}

/**
 * (External API) "Z" Gate
 */
void Z(uint id, uint qubit)
{
    QrackSimulatorManager::Instance()->Z(id, qubit);
}

/**
 * (External API) Controlled "Z" Gate
 */
void MCZ(uint id, uint count, uint[] ctrls, uint qubit)
{
    QrackSimulatorManager::Instance()->MCZ(id, count, ctrls, qubit);
}