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

#if defined DLL_EXPORTS
    #if defined WIN32
        #define LIB_API(RetType) extern "C" __declspec(dllexport) RetType
    #else
        #define LIB_API(RetType) extern "C" RetType __attribute__((visibility("default")))
    #endif
#else
    #if defined WIN32
        #define LIB_API(RetType) extern "C" __declspec(dllimport) RetType
    #else
        #define LIB_API(RetType) extern "C" RetType
    #endif
#endif

#include <map>
#include <vector>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

#if defined(_WIN32)
typedef void (__stdcall *IdsCallback)(unsigned int);
#else
typedef void (*IdsCallback)(unsigned int);
#endif

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
};

class QrackSimulatorManager {
protected:
    static QrackSimulatorManager* m_pInstance;
    std::vector<QInterfacePtr> simulators;
    std::map<QInterfacePtr, std::map<unsigned int, bitLenInt>> shards;

    QrackSimulatorManager()
    {
        // Intentionally left blank;
    }

    void mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx);

    void TransformPauliBasis(QInterfacePtr simulator, unsigned int len, Pauli* bases, unsigned int* qubitIds);

    void RevertPauliBasis(QInterfacePtr simulator, unsigned int len, Pauli* bases, unsigned int* qubitIds);

public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static QrackSimulatorManager* Instance();

   /**
    * Initialize a simulator ID with 0 qubits
    */
    unsigned int InitNewSimulator();

    /**
    * Destroy a simulator (ID will not be reused)
    */
    void DestroySimulator(unsigned int id);

    /**
    * "Dump" all IDs from the selected simulator ID into the callback
    */
    void DumpIds(unsigned int id, IdsCallback callback);

    /**
    * Set RNG seed for simulator ID
    */
    void SetSeed(unsigned int simulatorId, uint32_t seedValue);

    /**
    * Allocate 1 new qubit with the given qubit ID, under the simulator ID
    */
    void AllocateOneQubit(unsigned int simulatorId, long qubitId);

    /**
    * Release 1 qubit with the given qubit ID, under the simulator ID
    */
    bool ReleaseOneQubit(unsigned int simulatorId, long qubitId);

    /**
     * Find the joint probability for all specified qubits under the respective Pauli basis transformations.
     */
    double JointEnsembleProbability(unsigned int simulatorId, unsigned int len, Pauli* bases, unsigned int* qubitIds);

    /**
     * Exponentiation of Pauli operators
    */
    void Exp(unsigned int simulatorId, unsigned int len, Pauli* paulis, double angle, unsigned int* qubitIds);

    /**
     * Exponentiation of Pauli operators
    */
    void MCExp(unsigned int simulatorId, unsigned int len, Pauli* paulis, double angle, unsigned int ctrlLen, unsigned int* ctrls, unsigned int* qubitIds);

    /**
    * Walsh-Hadamard transform applied for simulator ID and qubit ID
    */
    void H(unsigned int simulatorId, unsigned int qubit);

    /**
     * (External API) Measure bit in |0>/|1> basis
     */
    unsigned int M(unsigned int id, unsigned int q);

    /**
     * Measure bits in specified Pauli bases
     */
    unsigned int Measure(unsigned int simulatorId, unsigned int len, Pauli* bases, unsigned int* qubitIds);

    /**
    * (External API) Rotation around Pauli axes
    */
    void R(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids);

    /**
    * (External API) Controlled rotation around Pauli axes
    */
    void MCR(unsigned int id, unsigned int len, Pauli* paulis, double angle, unsigned int ctrlLen, unsigned int* ctrls, unsigned int* ids);

    /**
     * "S" Gate
     */
    void S(unsigned int id, unsigned int qubit);

    /**
     * Inverse "S" Gate
     */
    void AdjS(unsigned int id, unsigned int qubit);

    /**
     * Controlled "S" Gate
     */
    void MCS(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * Controlled inverse "S" Gate
     */
    void MCAdjS(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * "T" Gate
     */
    void T(unsigned int id, unsigned int qubit);

    /**
     * Inverse "T" Gate
     */
    void AdjT(unsigned int id, unsigned int qubit);

    /**
     * Controlled "T" Gate
     */
    void MCT(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * Controlled inverse "T" Gate
     */
    void MCAdjT(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * "X" Gate
     */
    void X(unsigned int id, unsigned int qubit);

    /**
     * Controlled "X" Gate
     */
    void MCX(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * "Y" Gate
     */
    void Y(unsigned int id, unsigned int qubit);

    /**
     * Controlled "Y" Gate
     */
    void MCY(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

    /**
     * "Z" Gate
     */
    void Z(unsigned int id, unsigned int qubit);

    /**
     * Controlled "Z" Gate
     */
    void MCZ(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);
};

/**
 * (External API) Initialize a simulator ID with 0 qubits
 */
LIB_API(unsigned int) init();

/**
* (External API) Destroy a simulator (ID will not be reused)
*/
LIB_API(void) destroy(unsigned int id);

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
LIB_API(void) DumpIds(unsigned int id, IdsCallback callback);

/**
* (External API) Set RNG seed for simulator ID
*/
LIB_API(void) seed(unsigned int id, uint32_t seedValue);

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
LIB_API(void) allocateQubit(unsigned int id, unsigned int qubit_id);

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
LIB_API(void) release(unsigned int id, unsigned int qubit_id);

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
LIB_API(double) JointEnsembleProbability(unsigned int id, unsigned int n, Pauli* b, unsigned int* q);

/**
 * (External API) Exponentiation of Pauli operators
 */
LIB_API(void) Exp(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids);

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
LIB_API(void) MCExp(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int nc, unsigned int* ctrls, unsigned int* ids);

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
LIB_API(void) H(unsigned int id, unsigned int qubit);

/**
 * (External API) Measure bit in |0>/|1> basis
 */
LIB_API(unsigned int) M(unsigned int id, unsigned int q);

/**
 * (External API) Measure bits in specified Pauli bases
 */
LIB_API(unsigned int) Measure(unsigned int id, unsigned int n, Pauli* b, unsigned int* ids);

/**
 * (External API) Rotation around Pauli axes
 */
LIB_API(void) R(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int* ids);

/**
 * (External API) Controlled rotation around Pauli axes
 */
LIB_API(void) MCR(unsigned int id, unsigned int n, Pauli* paulis, double angle, unsigned int nc, unsigned int* ctrls, unsigned int* ids);

LIB_API(long) random_choice(unsigned int id, long size, double* p);

/**
 * (External API) "S" Gate
 */
LIB_API(void) S(unsigned int id, unsigned int qubit);

/**
 * (External API) Inverse "S" Gate
 */
LIB_API(void) AdjS(unsigned int id, unsigned int qubit);

/**
 * (External API) Controlled "S" Gate
 */
LIB_API(void) MCS(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) Controlled Inverse "S" Gate
 */
LIB_API(void) MCAdjS(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) "T" Gate
 */
LIB_API(void) T(unsigned int id, unsigned int qubit);

/**
 * (External API) Inverse "T" Gate
 */
LIB_API(void) AdjT(unsigned int id, unsigned int qubit);

/**
 * (External API) Controlled "T" Gate
 */
LIB_API(void) MCT(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) Controlled Inverse "T" Gate
 */
LIB_API(void) MCAdjT(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) "X" Gate
 */
LIB_API(void) X(unsigned int id, unsigned int qubit);

/**
 * (External API) Controlled "X" Gate
 */
LIB_API(void) MCX(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) "Y" Gate
 */
LIB_API(void) Y(unsigned int id, unsigned int qubit);

/**
 * (External API) Controlled "Y" Gate
 */
LIB_API(void) MCY(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);

/**
 * (External API) "Z" Gate
 */
LIB_API(void) Z(unsigned int id, unsigned int qubit);

/**
 * (External API) Controlled "Z" Gate
 */
LIB_API(void) MCZ(unsigned int id, unsigned int count, unsigned int* ctrls, unsigned int qubit);