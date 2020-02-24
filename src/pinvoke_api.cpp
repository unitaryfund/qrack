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

#include "pinvoke_api.hpp"

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

#pragma once

#include <map>
#include <vector>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

enum Pauli
{
	/// Pauli Identity operator. Corresponds to Q# constant "PauliI."
	PauliI = 0U,
	/// Pauli X operator. Corresponds to Q# constant "PauliX."
	PauliX = 1U,
	/// Pauli Y operator. Corresponds to Q# constant "PauliY."
	PauliY = 3U,
	/// Pauli Z operator. Corresponds to Q# constant "PauliZ."
	PauliZ = 2U
};

class QrackSimulatorManager {
protected:
    static QrackSimulatorManager* m_pInstance;
    qrack_rand_gen_ptr rng;
    std::vector<QInterfacePtr> simulators;
    std::map<QInterfacePtr, std::map<unsigned, bitLenInt>> shards;

    QrackSimulatorManager()
    {
        rng = std::make_shared<qrack_rand_gen>(std::time(0));
    }

    void mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx);

    void TransformPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds);

    void RevertPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds);

public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static QrackSimulatorManager* Instance();

   /**
    * Initialize a simulator ID with 0 qubits
    */
    unsigned InitNewSimulator();

    /**
    * Destroy a simulator (ID will not be reused)
    */
    void DestroySimulator(unsigned id);

    /**
    * "Dump" all IDs from the selected simulator ID into the callback
    */
    void DumpIds(unsigned id, void(*callback)(unsigned));

	/**
     * Select from a distribution of "n" elements according the discrete probabilities in "d."
     */
	std::size_t random_choice(unsigned simulatorId, std::size_t n, double* d);

    /**
    * Set RNG seed for simulator ID
    */
    void SetSeed(unsigned simulatorId, unsigned seedValue);

    /**
    * Allocate 1 new qubit with the given qubit ID, under the simulator ID
    */
    void AllocateOneQubit(unsigned simulatorId, long qubitId);

    /**
    * Release 1 qubit with the given qubit ID, under the simulator ID
    */
    bool ReleaseOneQubit(unsigned simulatorId, long qubitId);

    /**
    * Get currently allocated number of qubits, under the simulator ID
    */
	unsigned NumQubits(unsigned simulatorId);

    /**
     * Find the joint probability for all specified qubits under the respective Pauli basis transformations.
     */
    double JointEnsembleProbability(unsigned simulatorId, unsigned len, unsigned* bases, unsigned* qubitIds);

    /**
     * Exponentiation of Pauli operators
    */
    void Exp(unsigned simulatorId, unsigned len, unsigned* paulis, double angle, unsigned* qubitIds);

    /**
     * Exponentiation of Pauli operators
    */
    void MCExp(unsigned simulatorId, unsigned len, unsigned* paulis, double angle, unsigned ctrlLen, unsigned* ctrls, unsigned* qubitIds);

    /**
    * Walsh-Hadamard transform applied for simulator ID and qubit ID
    */
    void H(unsigned simulatorId, unsigned qubit);

    /**
     * (External API) Measure bit in |0>/|1> basis
     */
    unsigned M(unsigned id, unsigned q);

    /**
     * Measure bits in specified Pauli bases
     */
    unsigned Measure(unsigned simulatorId, unsigned len, unsigned* bases, unsigned* qubitIds);

    /**
    * (External API) Rotation around Pauli axes
    */
    void R(unsigned simulatorId, unsigned basis, double phi, unsigned qubitId);

    /**
    * (External API) Controlled rotation around Pauli axes
    */
    void MCR(unsigned simulatorId, unsigned basis, double phi, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId);

    /**
     * "S" Gate
     */
    void S(unsigned id, unsigned qubit);

    /**
     * Inverse "S" Gate
     */
    void AdjS(unsigned id, unsigned qubit);

    /**
     * Controlled "S" Gate
     */
    void MCS(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * Controlled inverse "S" Gate
     */
    void MCAdjS(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * "T" Gate
     */
    void T(unsigned id, unsigned qubit);

    /**
     * Inverse "T" Gate
     */
    void AdjT(unsigned id, unsigned qubit);

    /**
     * Controlled "T" Gate
     */
    void MCT(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * Controlled inverse "T" Gate
     */
    void MCAdjT(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * "X" Gate
     */
    void X(unsigned id, unsigned qubit);

    /**
     * Controlled "X" Gate
     */
    void MCX(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * "Y" Gate
     */
    void Y(unsigned id, unsigned qubit);

    /**
     * Controlled "Y" Gate
     */
    void MCY(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * "Z" Gate
     */
    void Z(unsigned id, unsigned qubit);

    /**
     * Controlled "Z" Gate
     */
    void MCZ(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);

    /**
     * Controlled "H" Gate
     */
    void MCH(unsigned id, unsigned count, unsigned* ctrls, unsigned qubit);
};

void QrackSimulatorManager::mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx) {
    for (unsigned i = 0; i < 4; i++) {
		outMtrx[i] = scalar * inMtrx[i];
    }
}

void QrackSimulatorManager::TransformPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds) {
    for (unsigned i = 0; i < len; i++) {
        switch (bases[i]) {
            case PauliX:
                simulator->H(shards[simulator][qubitIds[i]]);
                break;
            case PauliY:
                simulator->Z(shards[simulator][qubitIds[i]]);
                simulator->S(shards[simulator][qubitIds[i]]);
                simulator->H(shards[simulator][qubitIds[i]]);
                break;
            case PauliZ:
            case PauliI:
            default:
                break;
        }
    }
}

void QrackSimulatorManager::RevertPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds) {
    for (unsigned i = 0; i < len; i++) {
        switch (bases[i]) {
            case PauliX:
                simulator->H(shards[simulator][qubitIds[i]]);
                break;
            case PauliY:
                simulator->H(shards[simulator][qubitIds[i]]);
                simulator->IS(shards[simulator][qubitIds[i]]);
                simulator->Z(shards[simulator][qubitIds[i]]);
                break;
            case PauliZ:
            case PauliI:
            default:
                break;
        }
    }
}

QrackSimulatorManager* QrackSimulatorManager::m_pInstance = NULL;

QrackSimulatorManager* QrackSimulatorManager::Instance() {
    if (!m_pInstance) {
        m_pInstance = new QrackSimulatorManager();
    }
    return m_pInstance;
}

unsigned QrackSimulatorManager::InitNewSimulator() {
    simulators.push_back(NULL);
    return simulators.size() - 1U;
}

void QrackSimulatorManager::DestroySimulator(unsigned id) {
    simulators[id] = NULL;
}

void QrackSimulatorManager::DumpIds(unsigned simulatorId, void(*callback)(unsigned)) {
    QInterfacePtr simulator = simulators[simulatorId];
    std::map<unsigned, bitLenInt>::iterator it;

    for (it = shards[simulator].begin(); it != shards[simulator].end(); it++) {
        callback(it->first);
    }
}

std::size_t QrackSimulatorManager::random_choice(unsigned simulatorId, std::size_t n, double* d)
{
    std::discrete_distribution<std::size_t> dist(d, d+n);
    return dist(*rng.get());
}

void QrackSimulatorManager::SetSeed(unsigned simulatorId, unsigned seedValue) {
    simulators[simulatorId]->SetRandomSeed(seedValue);
}

void QrackSimulatorManager::AllocateOneQubit(unsigned simulatorId, long qubitId) {
    QInterfacePtr nQubit = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, 1, 0);
    if (simulators[simulatorId] == NULL) {
        simulators[simulatorId] = nQubit;
    } else {
        simulators[simulatorId]->Compose(nQubit);
    }
    shards[simulators[simulatorId]][qubitId] = simulators[simulatorId]->GetQubitCount();
}

bool QrackSimulatorManager::ReleaseOneQubit(unsigned simulatorId, long qubitId) {
    QInterfacePtr simulator = simulators[simulatorId];
    bool isQubitZero = simulator->Prob(shards[simulator][qubitId]) < min_norm;

    if (simulator->GetQubitCount() == 1U) {
        simulators[simulatorId] = NULL;
        shards.erase(simulator);
    } else {
        bitLenInt oIndex = shards[simulator][qubitId];
        simulator->Dispose(oIndex, 1U);
        for (unsigned i = 0; i < shards[simulator].size(); i++) {
            if (shards[simulator][i] > oIndex) {
                shards[simulator][i]--;
            }
        }
        shards[simulator].erase(qubitId);
    }

    return isQubitZero;
}

unsigned QrackSimulatorManager::NumQubits(unsigned simulatorId) {
    return (unsigned)simulators[simulatorId]->GetQubitCount();
}

double QrackSimulatorManager::JointEnsembleProbability(unsigned simulatorId, unsigned len, unsigned* bases, unsigned* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    double jointProb = 1.0;

    TransformPauliBasis(simulator, len, bases, qubitIds);

    for (unsigned i = 0; i < len; i++) {
        jointProb *= (double)simulator->Prob(shards[simulator][qubitIds[i]]);
        if (jointProb == 0.0) {
            break;
        }
    }

    RevertPauliBasis(simulator, len, bases, qubitIds);

    return jointProb;
}

void QrackSimulatorManager::Exp(unsigned simulatorId, unsigned len, unsigned* paulis, double angle, unsigned* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    for (unsigned i = 0; i < len; i++) {
        switch (paulis[i]) {
            case PauliI:
                simulator->Exp(angle, shards[simulator][qubitIds[i]]);
                break;
            case PauliX:
                simulator->ExpX(angle, shards[simulator][qubitIds[i]]);
                break;
            case PauliY:
                simulator->ExpY(angle, shards[simulator][qubitIds[i]]);
                break;
            case PauliZ:
                simulator->ExpZ(angle, shards[simulator][qubitIds[i]]);
                break;
            default:
                break;
        }
    }
}

void QrackSimulatorManager::MCExp(unsigned simulatorId, unsigned len, unsigned* paulis, double angle, unsigned ctrlLen, unsigned* ctrls, unsigned* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    const complex pauliI[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
    const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex pauliY[4] = { ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };

    complex toApply[4];

    for (unsigned i = 0; i < len; i++) {
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
        simulator->Exp(ctrlsArray, ctrlLen, shards[simulator][qubitIds[i]], toApply);
    }

    delete[] ctrlsArray;
}

void QrackSimulatorManager::H(unsigned simulatorId, unsigned qubitId) {
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->H(shards[simulator][qubitId]);
}

unsigned QrackSimulatorManager::M(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    return simulator->M(shards[simulator][qubitId]) ? 1U : 0U;
}

unsigned QrackSimulatorManager::Measure(unsigned simulatorId, unsigned len, unsigned* bases, unsigned* qubitIds)
{
    QInterfacePtr simulator = simulators[simulatorId];
    unsigned toRet = 0U;

    TransformPauliBasis(simulator, len, bases, qubitIds);

    for (unsigned i = 0; i < len; i++) {
        if (simulator->M(shards[simulator][qubitIds[i]]))
        {
            toRet |= pow2((bitLenInt)i);
        }
    }

    RevertPauliBasis(simulator, len, bases, qubitIds);

    return toRet;
}

void QrackSimulatorManager::R(unsigned simulatorId, unsigned basis, double phi, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];

    switch (basis) {
        case PauliI:
            simulator->RT(phi, shards[simulator][qubitId]);
            break;
        case PauliX:
            simulator->RX(phi, shards[simulator][qubitId]);
            break;
        case PauliY:
            simulator->RY(phi, shards[simulator][qubitId]);
            break;
        case PauliZ:
            simulator->RZ(phi, shards[simulator][qubitId]);
            break;
        default:
            break;
    }
}

void QrackSimulatorManager::MCR(unsigned simulatorId, unsigned basis, double phi, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    real1 cosine = cos(phi / 2.0);
    real1 sine = sin(phi / 2.0);
    complex pauliR[4];

    switch (basis) {
        case PauliI:
            simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], complex(ONE_R1, ZERO_R1), complex(cosine, sine));
            break;
        case PauliX:
            pauliR[0] = complex(cosine, ZERO_R1);
            pauliR[1] = complex(ZERO_R1, -sine);
            pauliR[2] = complex(ZERO_R1, -sine);
            pauliR[3] = complex(cosine, ZERO_R1);
            simulator->ApplyControlledSingleBit(ctrlsArray, ctrlLen, shards[simulator][qubitId], pauliR);
            break;
        case PauliY:
            pauliR[0] = complex(cosine, ZERO_R1);
            pauliR[1] = complex(-sine, ZERO_R1);
            pauliR[2] = complex(-sine, ZERO_R1);
            pauliR[3] = complex(cosine, ZERO_R1);
            simulator->ApplyControlledSingleBit(ctrlsArray, ctrlLen, shards[simulator][qubitId], pauliR);
            break;
        case PauliZ:
            simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], complex(cosine, -sine), complex(cosine, sine));
            break;
        default:
            break;
    }

    delete[] ctrlsArray;
}

void QrackSimulatorManager::S(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->S(shards[simulator][qubitId]);
}

void QrackSimulatorManager::AdjS(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->IS(shards[simulator][qubitId]);
}

void QrackSimulatorManager::MCS(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 2));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::MCAdjS(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 2));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::T(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->T(shards[simulator][qubitId]);
}

void QrackSimulatorManager::AdjT(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->IT(shards[simulator][qubitId]);
}

void QrackSimulatorManager::MCT(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 4));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::MCAdjT(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 4));

    delete[] ctrlsArray;
}

void QrackSimulatorManager::X(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->X(shards[simulator][qubitId]);
}

void QrackSimulatorManager::MCX(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, ONE_CMPLX);

    delete[] ctrlsArray;
}

void QrackSimulatorManager::Y(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->Y(shards[simulator][qubitId]);
}

void QrackSimulatorManager::MCY(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, ctrlLen, shards[simulator][qubitId], -I_CMPLX, I_CMPLX);

    delete[] ctrlsArray;
}

void QrackSimulatorManager::Z(unsigned simulatorId, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    simulator->Z(shards[simulator][qubitId]);
}

void QrackSimulatorManager::MCZ(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, ctrlLen, shards[simulator][qubitId], ONE_CMPLX, -ONE_CMPLX);

    delete[] ctrlsArray;
}

void QrackSimulatorManager::MCH(unsigned simulatorId, unsigned ctrlLen, unsigned* ctrls, unsigned qubitId)
{
    QInterfacePtr simulator = simulators[simulatorId];
    bitLenInt* ctrlsArray = new bitLenInt[ctrlLen];
    for (unsigned i = 0; i < ctrlLen; i++) {
		ctrlsArray[i] = shards[simulator][ctrls[i]];
    }

	const complex hGate[4] = {
		complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
		complex(M_SQRT1_2, ZERO_R1), complex(-M_SQRT1_2, ZERO_R1)
	};

    simulator->ApplyControlledSingleBit(ctrlsArray, ctrlLen, shards[simulator][qubitId], hGate);

    delete[] ctrlsArray;
}

extern "C" {

/**
	* (External API) Initialize a simulator ID with 0 qubits
	*/
MICROSOFT_QUANTUM_DECL unsigned init()
{
	return QrackSimulatorManager::Instance()->InitNewSimulator();
}

/**
* (External API) Destroy a simulator (ID will not be reused)
*/
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid)
{
	QrackSimulatorManager::Instance()->DestroySimulator(sid);
}

/**
* (External API) Set RNG seed for simulator ID
*/
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s)
{
	QrackSimulatorManager::Instance()->SetSeed(sid, s);
}


/**
	* (External API) "Dump" all IDs from the selected simulator ID into the callback
	*/
MICROSOFT_QUANTUM_DECL void DumpIds(_In_ unsigned sid, _In_ void(*callback)(unsigned))
{
	QrackSimulatorManager::Instance()->DumpIds(sid, callback);
}

/**
	* (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
	*/
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
	return QrackSimulatorManager::Instance()->random_choice(sid, n, p);
}

/**
	* (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
	*/
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
	return QrackSimulatorManager::Instance()->JointEnsembleProbability(sid, n, b, q);
}

/**
	* (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
	*/
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
	QrackSimulatorManager::Instance()->AllocateOneQubit(sid, qid);
}

/**
	* (External API) Release 1 qubit with the given qubit ID, under the simulator ID
	*/
MICROSOFT_QUANTUM_DECL void release(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->ReleaseOneQubit(sid, q);
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
	return QrackSimulatorManager::Instance()->NumQubits(sid);
}

/**
	* (External API) "X" Gate
	*/
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->X(sid, q);
}

/**
	* (External API) "Y" Gate
	*/
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->Y(sid, q);
}

/**
	* (External API) "Z" Gate
	*/
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->Z(sid, q);
}

/**
	* (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
	*/
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->H(sid, q);
}

/**
	* (External API) "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->S(sid, q);
}

/**
	* (External API) "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->T(sid, q);
}

/**
	* (External API) Inverse "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->AdjS(sid, q);
}

/**
	* (External API) Inverse "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->AdjT(sid, q);
}

/**
	* (External API) Controlled "X" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCX(sid, n, c, q);
}

/**
	* (External API) Controlled "Y" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCY(sid, n, c, q);
}

/**
	* (External API) Controlled "Z" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCZ(sid, n, c, q);
}

/**
	* (External API) Controlled "H" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCH(sid, n, c, q);
}

/**
	* (External API) Controlled "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCS(sid, n, c, q);;
}

/**
	* (External API) Controlled "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCT(sid, n, c, q);;
}

/**
	* (External API) Controlled Inverse "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCAdjS(sid, n, c, q);;
}

/**
	* (External API) Controlled Inverse "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCAdjT(sid, n, c, q);;
}

/**
	* (External API) Rotation around Pauli axes
	*/
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->R(sid, b, phi, q);
}

/**
	* (External API) Controlled rotation around Pauli axes
	*/
MICROSOFT_QUANTUM_DECL void MCR(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QrackSimulatorManager::Instance()->MCR(sid, b, phi, n, c, q);
}

/**
	* (External API) Exponentiation of Pauli operators
	*/
MICROSOFT_QUANTUM_DECL void Exp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi, _In_reads_(n) unsigned* q)
{
	QrackSimulatorManager::Instance()->Exp(sid, n, b, phi, q);
}

/**
	* (External API) Controlled exponentiation of Pauli operators
	*/
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi, _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q)
{
	QrackSimulatorManager::Instance()->MCExp(sid, n, b, phi, nc, cs, q);
}

/**
	* (External API) Measure bit in |0>/|1> basis
	*/
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q)
{
	return QrackSimulatorManager::Instance()->M(sid, q);
}

/**
	* (External API) Measure bits in specified Pauli bases
	*/
MICROSOFT_QUANTUM_DECL unsigned Measure(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
	return QrackSimulatorManager::Instance()->Measure(sid, n, b, q);
}

}
