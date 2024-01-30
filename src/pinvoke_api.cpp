//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "pinvoke_api.hpp"
#include "qcircuit.hpp"
#include "qneuron.hpp"

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#if !(FPPOW < 6 && !defined(ENABLE_COMPLEX_X2))
#include "hamiltonian.hpp"
#endif

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>

#define META_LOCK_GUARD() const std::lock_guard<std::mutex> metaLock(metaOperationMutex);

// SIMULATOR_LOCK_GUARD variants will lock simulatorMutexes[NULL], if the requested simulator doesn't exist.
// This is CORRECT behavior. This will effectively emplace a mutex for NULL key.
#if CPP_STD > 13
#define SIMULATOR_LOCK_GUARD(simulator)                                                                                \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator]);                                                    \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock =                                                                                                \
            std::make_unique<const std::lock_guard<std::mutex>>(simulatorMutexes[simulator], std::adopt_lock);         \
    }
#else
#define SIMULATOR_LOCK_GUARD(simulator)                                                                                \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator]);                                                    \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[simulator], std::adopt_lock));                      \
    }
#endif

#define SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                 \
    if (sid > simulators.size()) {                                                                                     \
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;                                         \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    SIMULATOR_LOCK_GUARD(simulator.get())                                                                              \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }

#define SIMULATOR_LOCK_GUARD_TYPED(sid, def)                                                                           \
    if (sid > simulators.size()) {                                                                                     \
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;                                         \
        metaError = 2;                                                                                                 \
        return def;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    SIMULATOR_LOCK_GUARD(simulator.get())                                                                              \
    if (!simulator) {                                                                                                  \
        return def;                                                                                                    \
    }

#define SIMULATOR_LOCK_GUARD_BOOL(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, false)

#define SIMULATOR_LOCK_GUARD_DOUBLE(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, 0.0)

#define SIMULATOR_LOCK_GUARD_INT(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, 0U)

#if CPP_STD > 13
#define NEURON_LOCK_GUARD(neuron)                                                                                      \
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock;                                                     \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[neuronSimulators[neuron]], neuronMutexes[neuron.get()]);        \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        neuronLock =                                                                                                   \
            std::make_unique<const std::lock_guard<std::mutex>>(neuronMutexes[neuron.get()], std::adopt_lock);         \
        simulatorLock = std::make_unique<const std::lock_guard<std::mutex>>(                                           \
            simulatorMutexes[neuronSimulators[neuron]], std::adopt_lock);                                              \
    }
#else
#define NEURON_LOCK_GUARD(neuron)                                                                                      \
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock;                                                     \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[neuronSimulators[neuron]], neuronMutexes[neuron.get()]);        \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        neuronLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                               \
            new const std::lock_guard<std::mutex>(neuronMutexes[neuron.get()], std::adopt_lock));                      \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[neuronSimulators[neuron]], std::adopt_lock));       \
    }
#endif

#define NEURON_LOCK_GUARD_VOID(nid)                                                                                    \
    if (nid > neurons.size()) {                                                                                        \
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;                                            \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    QNeuronPtr neuron = neurons[nid];                                                                                  \
    NEURON_LOCK_GUARD(neuron)                                                                                          \
    if (!neuron) {                                                                                                     \
        return;                                                                                                        \
    }

#define NEURON_LOCK_GUARD_TYPED(nid, def)                                                                              \
    if (nid > neurons.size()) {                                                                                        \
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;                                            \
        metaError = 2;                                                                                                 \
        return def;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    QNeuronPtr neuron = neurons[nid];                                                                                  \
    NEURON_LOCK_GUARD(neuron)                                                                                          \
    if (!neuron) {                                                                                                     \
        return def;                                                                                                    \
    }

#define NEURON_LOCK_GUARD_DOUBLE(nid) NEURON_LOCK_GUARD_TYPED(nid, 0.0)

#define NEURON_LOCK_GUARD_INT(nid) NEURON_LOCK_GUARD_TYPED(nid, 0U)

#if CPP_STD > 13
#define CIRCUIT_LOCK_GUARD(circuit)                                                                                    \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, circuitMutexes[circuit.get()]);                                                  \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        circuitLock =                                                                                                  \
            std::make_unique<const std::lock_guard<std::mutex>>(circuitMutexes[circuit.get()], std::adopt_lock);       \
    }
#else
#define CIRCUIT_LOCK_GUARD(circuit)                                                                                    \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, circuitMutexes[circuit.get()]);                                                  \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        circuitLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                              \
            new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()], std::adopt_lock));                    \
    }
#endif

#define CIRCUIT_LOCK_GUARD_TYPED(cid, def)                                                                             \
    if (cid > circuits.size()) {                                                                                       \
        std::cout << "Invalid argument: circuit ID not found!" << std::endl;                                           \
        metaError = 2;                                                                                                 \
        return def;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    CIRCUIT_LOCK_GUARD(circuit)                                                                                        \
    if (!circuit) {                                                                                                    \
        return def;                                                                                                    \
    }

#define CIRCUIT_LOCK_GUARD_VOID(cid)                                                                                   \
    if (cid > circuits.size()) {                                                                                       \
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;                                            \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    CIRCUIT_LOCK_GUARD(circuit)                                                                                        \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }

#define CIRCUIT_LOCK_GUARD_INT(cid) CIRCUIT_LOCK_GUARD_TYPED(cid, 0U)

#if CPP_STD > 13
#define CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)                                                                \
    if (sid > simulators.size()) {                                                                                     \
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;                                         \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    if (cid > circuits.size()) {                                                                                       \
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;                                            \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator.get()], circuitMutexes[circuit.get()]);               \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock =                                                                                                \
            std::make_unique<const std::lock_guard<std::mutex>>(simulatorMutexes[simulator.get()], std::adopt_lock);   \
        circuitLock =                                                                                                  \
            std::make_unique<const std::lock_guard<std::mutex>>(circuitMutexes[circuit.get()], std::adopt_lock);       \
    }                                                                                                                  \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }
#else
#define CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)                                                                \
    if (sid > simulators.size()) {                                                                                     \
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;                                         \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    if (cid > circuits.size()) {                                                                                       \
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;                                            \
        metaError = 2;                                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator.get()], circuitMutexes[circuit.get()]);               \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()], std::adopt_lock));                \
        circuitLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                              \
            new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()], std::adopt_lock));                    \
    }                                                                                                                  \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }
#endif

#define QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

using namespace Qrack;

qrack_rand_gen_ptr randNumGen = std::make_shared<qrack_rand_gen>(time(0));
std::mutex metaOperationMutex;
int metaError = 0;
std::vector<int> simulatorErrors;
std::vector<QInterfacePtr> simulators;
std::vector<std::vector<QInterfaceEngine>> simulatorTypes;
std::vector<bool> simulatorHostPointer;
std::map<QInterface*, std::mutex> simulatorMutexes;
std::vector<bool> simulatorReservations;
std::map<QInterface*, std::map<uintq, bitLenInt>> shards;
std::vector<int> neuronErrors;
std::vector<QNeuronPtr> neurons;
std::map<QNeuronPtr, QInterface*> neuronSimulators;
std::map<QNeuron*, std::mutex> neuronMutexes;
std::vector<bool> neuronReservations;
std::vector<QCircuitPtr> circuits;
std::map<QCircuit*, std::mutex> circuitMutexes;
std::vector<bool> circuitReservations;
bitLenInt _maxShardQubits = 0U;
bitLenInt MaxShardQubits()
{
    if (!_maxShardQubits) {
#if ENABLE_ENV_VARS
        _maxShardQubits =
            (bitLenInt)(getenv("QRACK_MAX_PAGING_QB") ? std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB"))) : -1);
#else
        _maxShardQubits = -1;
#endif
    }

    return _maxShardQubits;
}

void TransformPauliBasis(QInterfacePtr simulator, uintq len, int* bases, uintq* qubitIds)
{
    for (uintq i = 0U; i < len; ++i) {
        switch (bases[i]) {
        case PauliX:
            simulator->H(shards[simulator.get()][qubitIds[i]]);
            break;
        case PauliY:
            simulator->IS(shards[simulator.get()][qubitIds[i]]);
            simulator->H(shards[simulator.get()][qubitIds[i]]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void RevertPauliBasis(QInterfacePtr simulator, uintq len, int* bases, uintq* qubitIds)
{
    for (uintq i = 0U; i < len; ++i) {
        switch (bases[i]) {
        case PauliX:
            simulator->H(shards[simulator.get()][qubitIds[i]]);
            break;
        case PauliY:
            simulator->H(shards[simulator.get()][qubitIds[i]]);
            simulator->S(shards[simulator.get()][qubitIds[i]]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void removeIdentities(std::vector<int>* b, std::vector<bitLenInt>* qs)
{
    uintq i = 0U;
    while (i != b->size()) {
        if ((*b)[i] == PauliI) {
            b->erase(b->begin() + i);
            qs->erase(qs->begin() + i);
        } else {
            ++i;
        }
    }
}

void RHelper(uintq sid, uintq b, double phi, uintq q)
{
    QInterfacePtr simulator = simulators[sid];

    switch (b) {
    case PauliI: {
        // This is a global phase factor, with no measurable physical effect.
        // However, the underlying QInterface will not execute the gate
        // UNLESS it is specifically "keeping book" for non-measurable phase effects.
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->Phase(phaseFac, phaseFac, shards[simulator.get()][q]);
        break;
    }
    case PauliX:
        simulator->RX((real1_f)phi, shards[simulator.get()][q]);
        break;
    case PauliY:
        simulator->RY((real1_f)phi, shards[simulator.get()][q]);
        break;
    case PauliZ:
        simulator->RZ((real1_f)phi, shards[simulator.get()][q]);
        break;
    default:
        break;
    }
}

void MCRHelper(uintq sid, uintq b, double phi, uintq n, uintq* c, uintq q)
{
    QInterfacePtr simulator = simulators[sid];
    std::vector<bitLenInt> ctrlsArray(n);
    for (uintq i = 0U; i < n; ++i) {
        ctrlsArray[i] = shards[simulator.get()][c[i]];
    }

    if (b == PauliI) {
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->MCPhase(ctrlsArray, phaseFac, phaseFac, shards[simulator.get()][q]);
        return;
    }

    real1 cosine = (real1)cos(phi / 2);
    real1 sine = (real1)sin(phi / 2);
    complex pauliR[4U];

    switch (b) {
    case PauliX:
        pauliR[0U] = complex(cosine, ZERO_R1);
        pauliR[1U] = complex(ZERO_R1, -sine);
        pauliR[2U] = complex(ZERO_R1, -sine);
        pauliR[3U] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(ctrlsArray, pauliR, shards[simulator.get()][q]);
        break;
    case PauliY:
        pauliR[0U] = complex(cosine, ZERO_R1);
        pauliR[1U] = complex(-sine, ZERO_R1);
        pauliR[2U] = complex(sine, ZERO_R1);
        pauliR[3U] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(ctrlsArray, pauliR, shards[simulator.get()][q]);
        break;
    case PauliZ:
        simulator->MCPhase(ctrlsArray, complex(cosine, -sine), complex(cosine, sine), shards[simulator.get()][q]);
        break;
    case PauliI:
    default:
        break;
    }
}

inline std::size_t make_mask(std::vector<bitLenInt> const& qs)
{
    std::size_t mask = 0U;
    for (const std::size_t q : qs)
        mask = mask | pow2Ocl(q);
    return mask;
}

std::map<uintq, bitLenInt>::iterator FindShardValue(bitLenInt v, std::map<uintq, bitLenInt>& simMap)
{
    for (auto it = simMap.begin(); it != simMap.end(); ++it) {
        if (it->second == v) {
            // We have the matching it1, if we break.
            return it;
        }
    }

    return simMap.end();
}

void SwapShardValues(bitLenInt v1, bitLenInt v2, std::map<uintq, bitLenInt>& simMap)
{
    auto it1 = FindShardValue(v1, simMap);
    auto it2 = FindShardValue(v2, simMap);
    std::swap(it1->second, it2->second);
}

uintq MapArithmetic(QInterfacePtr simulator, uintq n, uintq* q)
{
    uintq start = shards[simulator.get()][q[0U]];
    std::unique_ptr<bitLenInt[]> bitArray(new bitLenInt[n]);
    for (uintq i = 0U; i < n; ++i) {
        bitArray[i] = shards[simulator.get()][q[i]];
        if (start > bitArray[i]) {
            start = bitArray[i];
        }
    }
    for (uintq i = 0U; i < n; ++i) {
        simulator->Swap(start + i, bitArray[i]);
        SwapShardValues(start + i, bitArray[i], shards[simulator.get()]);
    }

    return start;
}

struct MapArithmeticResult2 {
    uintq start1;
    uintq start2;

    MapArithmeticResult2(uintq s1, uintq s2)
        : start1(s1)
        , start2(s2)
    {
    }
};

MapArithmeticResult2 MapArithmetic2(QInterfacePtr simulator, uintq n, uintq* q1, uintq* q2)
{
    uintq start1 = shards[simulator.get()][q1[0U]];
    uintq start2 = shards[simulator.get()][q2[0U]];
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[n]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[n]);
    for (uintq i = 0U; i < n; ++i) {
        bitArray1[i] = shards[simulator.get()][q1[i]];
        if (start1 > bitArray1[i]) {
            start1 = bitArray1[i];
        }

        bitArray2[i] = shards[simulator.get()][q2[i]];
        if (start2 > bitArray2[i]) {
            start2 = bitArray2[i];
        }
    }

    bool isReversed = (start2 < start1);

    if (isReversed) {
        std::swap(start1, start2);
        bitArray1.swap(bitArray2);
    }

    for (uintq i = 0U; i < n; ++i) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }

    if ((start1 + n) > start2) {
        start2 = start1 + n;
    }

    for (uintq i = 0U; i < n; ++i) {
        simulator->Swap(start2 + i, bitArray2[i]);
        SwapShardValues(start2 + i, bitArray2[i], shards[simulator.get()]);
    }

    if (isReversed) {
        std::swap(start1, start2);
    }

    return MapArithmeticResult2(start1, start2);
}

MapArithmeticResult2 MapArithmetic3(QInterfacePtr simulator, uintq n1, uintq* q1, uintq n2, uintq* q2)
{
    uintq start1 = shards[simulator.get()][q1[0U]];
    uintq start2 = shards[simulator.get()][q2[0U]];
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[n1]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[n2]);
    for (uintq i = 0U; i < n1; ++i) {
        bitArray1[i] = shards[simulator.get()][q1[i]];
        if (start1 > bitArray1[i]) {
            start1 = bitArray1[i];
        }
    }

    for (uintq i = 0U; i < n2; ++i) {
        bitArray2[i] = shards[simulator.get()][q2[i]];
        if (start2 > bitArray2[i]) {
            start2 = bitArray2[i];
        }
    }

    bool isReversed = (start2 < start1);

    if (isReversed) {
        std::swap(start1, start2);
        std::swap(n1, n2);
        bitArray1.swap(bitArray2);
    }

    for (uintq i = 0U; i < n1; ++i) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }

    if ((start1 + n1) > start2) {
        start2 = start1 + n1;
    }

    for (uintq i = 0U; i < n2; ++i) {
        simulator->Swap(start2 + i, bitArray2[i]);
        SwapShardValues(start2 + i, bitArray2[i], shards[simulator.get()]);
    }

    if (isReversed) {
        std::swap(start1, start2);
    }

    return MapArithmeticResult2(start1, start2);
}

void _darray_to_creal1_array(double* params, bitCapIntOcl componentCount, complex* amps)
{
    for (bitCapIntOcl j = 0U; j < componentCount; ++j) {
        amps[j] = complex(real1(params[2U * j]), real1(params[2U * j + 1U]));
    }
}

bitCapInt _combineA(uintq na, const uintq* a)
{
    if (na > (bitsInCap / (8U * sizeof(uintq)))) {
        metaError = 2;
        std::cout << "Big integer is too large for bitCapInt!" << std::endl;
        return ZERO_BCI;
    }

#if QBCAPPOW > 6
    bitCapInt aTot = ZERO_BCI;
    for (uintq i = 0U; i < na; ++i) {
        bi_lshift_ip(&aTot, bitsInCap);
        bi_or_ip(&aTot, a[na - (i + 1U)]);
    }
    return aTot;
#else
    return a[0U];
#endif
}

extern "C" {

/**
 * (External API) Poll after each operation to check whether error occurred.
 */
MICROSOFT_QUANTUM_DECL int get_error(_In_ uintq sid)
{
    if (metaError) {
        metaError = 0;
        return 2;
    }
    return simulatorErrors[sid];
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and explicit layer options on/off
 */
MICROSOFT_QUANTUM_DECL uintq init_count_type(_In_ uintq q, _In_ bool tn, _In_ bool md, _In_ bool sd, _In_ bool sh,
    _In_ bool bdt, _In_ bool pg, _In_ bool zxf, _In_ bool hy, _In_ bool oc, _In_ bool hp)
{
    META_LOCK_GUARD()

    uintq sid = (uintq)simulators.size();

    for (uintq i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

#if ENABLE_OPENCL
    bool isOcl = oc && (OCLEngine::Instance().GetDeviceCount() > 0);
    bool isOclMulti = oc && md && (OCLEngine::Instance().GetDeviceCount() > 1);
#elif ENABLE_CUDA
    bool isOcl = oc && (CUDAEngine::Instance().GetDeviceCount() > 0);
    bool isOclMulti = oc && md && (CUDAEngine::Instance().GetDeviceCount() > 1);
#else
    bool isOclMulti = false;
#endif

    // Construct backwards, then reverse:
    std::vector<QInterfaceEngine> simulatorType;

#if ENABLE_OPENCL
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
    }
#elif ENABLE_CUDA
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_CUDA : QINTERFACE_CPU);
    }
#endif

    if (pg && !sh && simulatorType.size()) {
        simulatorType.push_back(QINTERFACE_QPAGER);
    }

    if (bdt) {
        // To recover the original QBdt stack behavior,
        // set env. var. QRACK_QBDT_HYBRID_THRESHOLD=1
        simulatorType.push_back(QINTERFACE_BDT_HYBRID);
    }

    if (sh && (!sd || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

    if (sd) {
        simulatorType.push_back(isOclMulti ? QINTERFACE_QUNIT_MULTI : QINTERFACE_QUNIT);
    }

    if (tn) {
        simulatorType.push_back(QINTERFACE_TENSOR_NETWORK);
    }

    // (...then reverse:)
    std::reverse(simulatorType.begin(), simulatorType.end());

    if (!simulatorType.size()) {
#if ENABLE_OPENCL
        if (hy && isOcl) {
            simulatorType.push_back(QINTERFACE_HYBRID);
        } else {
            simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
        }
#elif ENABLE_CUDA
        if (hy && isOcl) {
            simulatorType.push_back(QINTERFACE_HYBRID);
        } else {
            simulatorType.push_back(isOcl ? QINTERFACE_CUDA : QINTERFACE_CPU);
        }
#else
        simulatorType.push_back(QINTERFACE_CPU);
#endif
    }

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator =
                CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
        } catch (const std::exception& ex) {
            std::cout << ex.what() << std::endl;
            isSuccess = false;
        }
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(hp);
        simulatorErrors.push_back(isSuccess ? 0 : 1);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = hp;
        simulatorErrors[sid] = isSuccess ? 0 : 1;
    }

    if (!q) {
        return sid;
    }

    shards[simulator.get()] = {};
    for (uintq i = 0U; i < q; ++i) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
MICROSOFT_QUANTUM_DECL uintq init_count(_In_ uintq q, _In_ bool hp)
{
    META_LOCK_GUARD()

    uintq sid = (uintq)simulators.size();

    for (uintq i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    const std::vector<QInterfaceEngine> simulatorType{ QINTERFACE_TENSOR_NETWORK };

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator =
                CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
        } catch (const std::exception& ex) {
            std::cout << ex.what() << std::endl;
            isSuccess = false;
        }
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(hp);
        simulatorErrors.push_back(isSuccess ? 0 : 1);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = hp;
        simulatorErrors[sid] = isSuccess ? 0 : 1;
    }

    if (!q) {
        return sid;
    }

    shards[simulator.get()] = {};
    for (uintq i = 0U; i < q; ++i) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
MICROSOFT_QUANTUM_DECL uintq init_count_pager(_In_ uintq q, _In_ bool hp)
{
    META_LOCK_GUARD()

    uintq sid = (uintq)simulators.size();

    for (uintq i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    const std::vector<QInterfaceEngine> simulatorType{ QINTERFACE_TENSOR_NETWORK, QINTERFACE_OPTIMAL };

    std::vector<int64_t> deviceList;
#if ENABLE_OPENCL
    std::vector<DeviceContextPtr> deviceContext = OCLEngine::Instance().GetDeviceContextPtrVector();
    for (size_t i = 0U; i < deviceContext.size(); ++i) {
        deviceList.push_back(i);
    }
    const size_t defaultDeviceID = OCLEngine::Instance().GetDefaultDeviceID();
    std::swap(deviceList[0U], deviceList[defaultDeviceID]);
#elif ENABLE_CUDA
    std::vector<DeviceContextPtr> deviceContext = CUDAEngine::Instance().GetDeviceContextPtrVector();
    for (size_t i = 0U; i < deviceContext.size(); ++i) {
        deviceList.push_back(i);
    }
    const size_t defaultDeviceID = CUDAEngine::Instance().GetDefaultDeviceID();
    std::swap(deviceList[0U], deviceList[defaultDeviceID]);
#endif

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator = CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true,
                hp, -1, true, false, REAL1_EPSILON, deviceList, 0, FP_NORM_EPSILON_F);
        } catch (const std::exception& ex) {
            std::cout << ex.what() << std::endl;
            isSuccess = false;
        }
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(hp);
        simulatorErrors.push_back(isSuccess ? 0 : 1);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = hp;
        simulatorErrors[sid] = isSuccess ? 0 : 1;
    }

    if (!q) {
        return sid;
    }

    shards[simulator.get()] = {};
    for (uintq i = 0U; i < q; ++i) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID that clones simulator ID "sid"
 */
MICROSOFT_QUANTUM_DECL uintq init_clone(_In_ uintq sid)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;
        metaError = 2;
        return 0U;
    }
    QInterfacePtr oSimulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[oSimulator.get()]));

    uintq nsid = (uintq)simulators.size();

    for (uintq i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            nsid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    bool isSuccess = true;
    QInterfacePtr simulator;
    try {
        simulator = oSimulator->Clone();
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        isSuccess = false;
    }

    if (nsid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorTypes[sid]);
        simulatorHostPointer.push_back(simulatorHostPointer[sid]);
        simulatorErrors.push_back(isSuccess ? 0 : 1);
        shards[simulator.get()] = {};
    } else {
        simulatorReservations[nsid] = true;
        simulators[nsid] = simulator;
        simulatorTypes[nsid] = simulatorTypes[sid];
        simulatorHostPointer[nsid] = simulatorHostPointer[sid];
        simulatorErrors[nsid] = isSuccess ? 0 : 1;
    }

    shards[simulator.get()] = {};
    for (uintq i = 0U; i < simulator->GetQubitCount(); ++i) {
        shards[simulator.get()][i] = shards[simulators[sid].get()][i];
    }

    return nsid;
}

/**
 * (External API) Destroy a simulator (ID will not be reused)
 */
MICROSOFT_QUANTUM_DECL void destroy(_In_ uintq sid)
{
    META_LOCK_GUARD()

    shards.erase(simulators[sid].get());
    simulatorMutexes.erase(simulators[sid].get());
    simulators[sid] = NULL;
    simulatorErrors[sid] = 0;
    simulatorReservations[sid] = false;
}

/**
 * (External API) Set RNG seed for simulator ID
 */
MICROSOFT_QUANTUM_DECL void seed(_In_ uintq sid, _In_ uintq s)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulators[sid]->SetRandomSeed((unsigned)s);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Set concurrency level per QEngine shard
 */
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ uintq sid, _In_ uintq p)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulators[sid]->SetConcurrency((unsigned)p);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void qstabilizer_out_to_file(_In_ uintq sid, _In_ char* f)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    if (simulatorTypes[sid][0] != QINTERFACE_STABILIZER_HYBRID) {
        simulatorErrors[sid] = 1;
        std::cout << "Cannot write any simulator but QStabilizerHybrid out to file!" << std::endl;
        return;
    }

    std::ofstream ofile;
    ofile.open(f);

    try {
        ofile << std::dynamic_pointer_cast<QStabilizerHybrid>(simulators[sid]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    ofile.close();
}
MICROSOFT_QUANTUM_DECL void qstabilizer_in_from_file(_In_ uintq sid, _In_ char* f)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    if (simulatorTypes[sid][0] != QINTERFACE_STABILIZER_HYBRID) {
        simulatorErrors[sid] = 1;
        std::cout << "Cannot read any simulator but QStabilizerHybrid in from file!" << std::endl;
        return;
    }

    std::ifstream ifile;
    ifile.open(f);

    try {
        ifile >> std::dynamic_pointer_cast<QStabilizerHybrid>(simulators[sid]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    ifile.close();

    shards[simulator.get()] = {};
    for (uintq i = 0U; i < simulator->GetQubitCount(); ++i) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void DumpIds(_In_ uintq sid, _In_ IdCallback callback)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    std::map<uintq, bitLenInt>::iterator it;
    for (it = shards[simulator.get()].begin(); it != shards[simulator.get()].end(); ++it) {
        callback(it->first);
    }
}

/**
 * (External API) "Dump" state vector from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void Dump(_In_ uintq sid, _In_ ProbAmpCallback callback)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    bitCapIntOcl wfnl = (bitCapIntOcl)simulator->GetMaxQPower();

    for (size_t i = 0U; i < wfnl; ++i) {
        complex amp;
        try {
            amp = simulator->GetAmplitude(i);
        } catch (const std::exception& ex) {
            simulatorErrors[sid] = 1;
            std::cout << ex.what() << std::endl;
            break;
        }

        if (!callback(i, (double)real(amp), (double)imag(amp))) {
            break;
        }
    }
}

/**
 * (External API) Set state vector for the selected simulator ID.
 */
MICROSOFT_QUANTUM_DECL void InKet(_In_ uintq sid, _In_ real1_f* ket)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    simulator->SetQuantumState(reinterpret_cast<complex*>(ket));
}

/**
 * (External API) Set state vector for the selected simulator ID.
 */
MICROSOFT_QUANTUM_DECL void OutKet(_In_ uintq sid, _In_ real1_f* ket)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    simulator->GetQuantumState(reinterpret_cast<complex*>(ket));
}

/**
 * (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
 */
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ uintq sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
    std::discrete_distribution<std::size_t> dist(p, p + n);
    return dist(*randNumGen.get());
}

MICROSOFT_QUANTUM_DECL void PhaseParity(_In_ uintq sid, _In_ double lambda, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    bitCapInt mask = ZERO_BCI;
    for (uintq i = 0U; i < n; ++i) {
        bi_or_ip(&mask, pow2(shards[simulator.get()][q[i]]));
    }

    try {
        simulator->PhaseParity((real1_f)lambda, mask);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

double _JointEnsembleProbabilityHelper(QInterfacePtr simulator, uintq n, int* b, uintq* q, bool doMeasure)
{

    if (!n) {
        return 0.0;
    }

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    removeIdentities(&bVec, &qVec);
    n = (uintq)qVec.size();

    if (!n) {
        return 0.0;
    }

    bitCapInt mask = ZERO_BCI;
    for (bitLenInt i = 0U; i < (bitLenInt)n; ++i) {
        bi_or_ip(&mask, pow2(shards[simulator.get()][qVec[i]]));
    }

    return (double)(doMeasure ? (QPARITY(simulator)->MParity(mask) ? ONE_R1 : ZERO_R1)
                              : QPARITY(simulator)->ProbParity(mask));
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    double jointProb = (double)REAL1_DEFAULT_ARG;

    try {
        TransformPauliBasis(simulator, n, b, q);

        jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, false);

        RevertPauliBasis(simulator, n, b, q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    return jointProb;
}

/**
 * (External API) Set the simulator to a computational basis permutation.
 */
MICROSOFT_QUANTUM_DECL void ResetAll(_In_ uintq sid)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->SetPermutation(ZERO_BCI);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ uintq sid, _In_ uintq qid)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;
        metaError = 2;
        return;
    }

    QInterfacePtr nQubit = CreateQuantumInterface(
        simulatorTypes[sid], 1U, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, simulatorHostPointer[sid]);

    if (simulators[sid] == NULL) {
        simulators[sid] = nQubit;
        shards[nQubit.get()] = {};
        shards[nQubit.get()][qid] = 0;

        return;
    }

    QInterfacePtr oSimulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[oSimulator.get()]));

    bitLenInt qubitCount = -1;
    try {
        oSimulator->Compose(nQubit);
        qubitCount = simulators[sid]->GetQubitCount();
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    shards[simulators[sid].get()][qid] = (qubitCount - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL bool release(_In_ uintq sid, _In_ uintq q)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;
        metaError = 2;
        return 0U;
    }
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()]));

    // Check that the qubit is in the |0> state, to within a small tolerance.
    bool toRet = simulator->Prob(shards[simulator.get()][q]) < (ONE_R1 / 100);

    if (simulator->GetQubitCount() == 1U) {
        shards[simulator.get()] = {};
        simulators[sid] = NULL;
    } else {
        bitLenInt oIndex = shards[simulator.get()][q];
        simulator->Dispose(oIndex, 1U);
        for (uintq i = 0U; i < shards[simulator.get()].size(); ++i) {
            if (shards[simulator.get()][i] > oIndex) {
                --(shards[simulator.get()][i]);
            }
        }
        shards[simulator.get()].erase(q);
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL uintq num_qubits(_In_ uintq sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    try {
        return (uintq)simulator->GetQubitCount();
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return -1;
    }
}

/**
 * (External API) "X" Gate
 */
MICROSOFT_QUANTUM_DECL void X(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->X(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void Y(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    try {
        simulator->Y(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void Z(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->Z(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
MICROSOFT_QUANTUM_DECL void H(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->H(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "S" Gate
 */
MICROSOFT_QUANTUM_DECL void S(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->S(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Square root of X gate
 */
MICROSOFT_QUANTUM_DECL void SX(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->SqrtX(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Square root of Y gate
 */
MICROSOFT_QUANTUM_DECL void SY(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->SqrtY(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "T" Gate
 */
MICROSOFT_QUANTUM_DECL void T(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->T(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjS(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->IS(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Inverse square root of X gate
 */
MICROSOFT_QUANTUM_DECL void AdjSX(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->ISqrtX(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Inverse square root of Y gate
 */
MICROSOFT_QUANTUM_DECL void AdjSY(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->ISqrtY(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjT(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->IT(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void U(_In_ uintq sid, _In_ uintq q, _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->U(shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void Mtrx(_In_ uintq sid, _In_reads_(8) double* m, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    complex mtrx[4U]{ complex((real1)m[0U], (real1)m[1U]), complex((real1)m[2U], (real1)m[3U]),
        complex((real1)m[4U], (real1)m[5U]), complex((real1)m[6U], (real1)m[7U]) };

    try {
        simulator->Mtrx(mtrx, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

#define MAP_CONTROLS_AND_LOCK(sid, numC)                                                                               \
    SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                     \
    std::vector<bitLenInt> ctrlsArray(numC);                                                                           \
    for (uintq i = 0; i < numC; ++i) {                                                                                 \
        ctrlsArray[i] = shards[simulator.get()][c[i]];                                                                 \
    }

/**
 * (External API) Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MCX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCInvert(ctrlsArray, ONE_CMPLX, ONE_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MCY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCInvert(ctrlsArray, -I_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MCZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray, ONE_CMPLX, -ONE_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MCH(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    const complex hGate[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCMtrx(ctrlsArray, hGate, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray, ONE_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray, ONE_CMPLX, -I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCU(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ double theta,
    _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->CU(ctrlsArray, shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q)
{
    complex mtrx[4U]{ complex((real1)m[0U], (real1)m[1U]), complex((real1)m[2U], (real1)m[3U]),
        complex((real1)m[4U], (real1)m[5U]), complex((real1)m[6U], (real1)m[7U]) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCMtrx(ctrlsArray, mtrx, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MACX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACInvert(ctrlsArray, ONE_CMPLX, ONE_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MACY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACInvert(ctrlsArray, -I_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MACZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray, ONE_CMPLX, -ONE_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MACH(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    const complex hGate[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACMtrx(ctrlsArray, hGate, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray, ONE_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray, ONE_CMPLX, -I_CMPLX, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) "Anti-"Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACU(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ double theta,
    _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->AntiCU(ctrlsArray, shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q)
{
    complex mtrx[4U]{ complex((real1)m[0U], (real1)m[1U]), complex((real1)m[2U], (real1)m[3U]),
        complex((real1)m[4U], (real1)m[5U]), complex((real1)m[6U], (real1)m[7U]) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACMtrx(ctrlsArray, mtrx, shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate with arbitrary control permutation
 */
MICROSOFT_QUANTUM_DECL void UCMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q, _In_ uintq p)
{
    complex mtrx[4U]{ complex((real1)m[0U], (real1)m[1U]), complex((real1)m[2U], (real1)m[3U]),
        complex((real1)m[4U], (real1)m[5U]), complex((real1)m[6U], (real1)m[7U]) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->UCMtrx(ctrlsArray, mtrx, shards[simulator.get()][q], p);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void Multiplex1Mtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, double* m)
{
    bitCapIntOcl componentCount = 4U * pow2Ocl(n);
    std::unique_ptr<complex[]> mtrxs(new complex[componentCount]);
    _darray_to_creal1_array(m, componentCount, mtrxs.get());

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->UniformlyControlledSingleBit(ctrlsArray, shards[simulator.get()][q], mtrxs.get());
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

#define MAP_MASK_AND_LOCK(sid, numQ)                                                                                   \
    SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                     \
    bitCapInt mask = ZERO_BCI;                                                                                         \
    for (uintq i = 0U; i < numQ; ++i) {                                                                                \
        bi_or_ip(&mask, pow2(shards[simulator.get()][q[i]]));                                                          \
    }

/**
 * (External API) Multiple "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    MAP_MASK_AND_LOCK(sid, n)
    try {
        simulator->XMask(mask);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Multiple "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    MAP_MASK_AND_LOCK(sid, n)
    try {
        simulator->YMask(mask);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Multiple "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    MAP_MASK_AND_LOCK(sid, n)
    try {
        simulator->ZMask(mask);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void R(_In_ uintq sid, _In_ uintq b, _In_ double phi, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        RHelper(sid, b, phi, q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ uintq sid, _In_ uintq b, _In_ double phi, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        MCRHelper(sid, b, phi, n, c, q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void Exp(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_ double phi, _In_reads_(n) uintq* q)
{
    if (!n) {
        return;
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    uintq someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    try {
        if (!bVec.size()) {
            RHelper(sid, PauliI, -2. * phi, someQubit);
        } else if (bVec.size() == 1U) {
            RHelper(sid, bVec.front(), -2. * phi, qVec.front());
        } else {
            TransformPauliBasis(simulator, n, b, q);

            std::size_t mask = make_mask(qVec);
            QPARITY(simulator)->UniformParityRZ(mask, (real1_f)(-phi));

            RevertPauliBasis(simulator, n, b, q);
        }
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void MCExp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_ double phi, _In_ uintq nc,
    _In_reads_(nc) uintq* cs, _In_reads_(n) uintq* q)
{
    if (!n) {
        return;
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    uintq someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    try {
        if (!bVec.size()) {
            MCRHelper(sid, PauliI, -2. * phi, nc, cs, someQubit);
        } else if (bVec.size() == 1U) {
            MCRHelper(sid, bVec.front(), -2. * phi, nc, cs, qVec.front());
        } else {
            std::vector<bitLenInt> csVec(cs, cs + nc);

            TransformPauliBasis(simulator, n, b, q);

            std::size_t mask = make_mask(qVec);
            QPARITY(simulator)->CUniformParityRZ(csVec, mask, (real1_f)(-phi));

            RevertPauliBasis(simulator, n, b, q);
        }
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
MICROSOFT_QUANTUM_DECL uintq M(_In_ uintq sid, _In_ uintq q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    try {
        return simulator->M(shards[simulator.get()][q]) ? 1U : 0U;
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return -1;
    }
}

/**
 * (External API) PSEUDO-QUANTUM: Post-select bit in |0>/|1> basis
 */
MICROSOFT_QUANTUM_DECL uintq ForceM(_In_ uintq sid, _In_ uintq q, _In_ bool r)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    try {
        return simulator->ForceM(shards[simulator.get()][q], r) ? 1U : 0U;
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return -1;
    }
}

/**
 * (External API) Measure all bits separately in |0>/|1> basis, and return the result in low-to-high order corresponding
 * with first-to-last in original order of allocation.
 */
MICROSOFT_QUANTUM_DECL uintq MAll(_In_ uintq sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    try {
        return (bitCapIntOcl)simulators[sid]->MAll();
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return -1;
    }
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
MICROSOFT_QUANTUM_DECL uintq Measure(_In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    uintq toRet = -1;
    try {
        TransformPauliBasis(simulator, n, b, q);

        double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, true);

        toRet = (jointProb < (ONE_R1 / 2)) ? 0U : 1U;

        RevertPauliBasis(simulator, n, b, q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL void MeasureShots(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq s, _In_reads_(s) uintq* m)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    std::vector<bitCapInt> qPowers(n);
    for (uintq i = 0U; i < n; ++i) {
        qPowers[i] = Qrack::pow2(shards[simulator.get()][q[i]]);
    }

    try {
        simulator->MultiShotMeasureMask(qPowers, (unsigned)s, m);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void SWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->Swap(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void ISWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->ISwap(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void AdjISWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->IISwap(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void FSim(_In_ uintq sid, _In_ double theta, _In_ double phi, _In_ uintq qi1, _In_ uintq qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->FSim((real1_f)theta, (real1_f)phi, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CSWAP(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq qi1, _In_ uintq qi2)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->CSwap(ctrlsArray, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void ACSWAP(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq qi1, _In_ uintq qi2)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->AntiCSwap(ctrlsArray, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void Compose(_In_ uintq sid1, _In_ uintq sid2, uintq* q)
{
    if (!simulators[sid1] || !simulators[sid2]) {
        return;
    }
    const std::lock_guard<std::mutex> simulatorLock1(simulatorMutexes[simulators[sid1].get()]);
    const std::lock_guard<std::mutex> simulatorLock2(simulatorMutexes[simulators[sid2].get()]);

    if (simulatorTypes[sid1].size() != simulatorTypes[sid2].size()) {
        metaError = 2;
        std::cout << "Cannot 'Compose()' simulators of different layer stack types!" << std::endl;
        return;
    }

    for (size_t i = 0U; i < simulatorTypes[sid1].size(); ++i) {
        if (simulatorTypes[sid1][i] != simulatorTypes[sid2][i]) {
            metaError = 2;
            std::cout << "Cannot 'Compose()' simulators of different layer stack types!" << std::endl;
            return;
        }
    }

    QInterfacePtr simulator1 = simulators[sid1];
    QInterfacePtr simulator2 = simulators[sid2];
    bitLenInt oQubitCount = 0U;
    bitLenInt pQubitCount = 0U;
    try {
        oQubitCount = simulator1->GetQubitCount();
        pQubitCount = simulator2->GetQubitCount();
        simulator1->Compose(simulator2);
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        simulatorErrors[sid1] = 1;
        simulatorErrors[sid2] = 1;
    }

    for (bitLenInt i = 0; i < pQubitCount; ++i) {
        shards[simulator1.get()][q[i]] = oQubitCount + i;
    }
}

MICROSOFT_QUANTUM_DECL uintq Decompose(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    uintq nSid = init_count(n, false);

    SIMULATOR_LOCK_GUARD_INT(sid)

    bitLenInt nQubitIndex = 0U;

    try {
        nQubitIndex = simulator->GetQubitCount() - n;

        for (uintq i = 0U; i < n; ++i) {
            simulator->Swap(shards[simulator.get()][q[i]], i + nQubitIndex);
        }

        simulator->Decompose(nQubitIndex, simulators[nSid]);
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        simulatorErrors[sid] = 1;
        simulatorErrors[nSid] = 1;
    }

    bitLenInt oIndex;
    for (uintq j = 0U; j < n; ++j) {
        oIndex = shards[simulator.get()][q[j]];
        for (uintq i = 0U; i < shards[simulator.get()].size(); ++i) {
            if (shards[simulator.get()][i] > oIndex) {
                --(shards[simulator.get()][i]);
            }
        }
        shards[simulator.get()].erase(q[j]);
    }

    simulatorTypes[nSid] = simulatorTypes[sid];
    simulatorHostPointer[nSid] = simulatorHostPointer[sid];

    return nSid;
}

MICROSOFT_QUANTUM_DECL void Dispose(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    bitLenInt nQubitIndex = 0U;

    try {
        nQubitIndex = simulator->GetQubitCount() - n;

        for (uintq i = 0U; i < n; ++i) {
            simulator->Swap(shards[simulator.get()][q[i]], i + nQubitIndex);
        }

        simulator->Dispose(nQubitIndex, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }

    bitLenInt oIndex;
    for (uintq j = 0U; j < n; ++j) {
        oIndex = shards[simulator.get()][q[j]];
        for (uintq i = 0U; i < shards[simulator.get()].size(); ++i) {
            if (shards[simulator.get()][i] > oIndex) {
                --(shards[simulator.get()][i]);
            }
        }
        shards[simulator.get()].erase(q[j]);
    }
}

MICROSOFT_QUANTUM_DECL void AND(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->AND(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void OR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->OR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void XOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->XOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void NAND(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->NAND(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void NOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->NOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void XNOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->XNOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLAND(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLAND(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLXOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLXOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLNAND(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLNAND(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLNOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLNOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->CLXNOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

double _Prob(_In_ uintq sid, _In_ uintq q, bool isRdm)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    try {
        return isRdm ? (double)simulator->ProbRdm(shards[simulator.get()][q])
                     : (double)simulator->Prob(shards[simulator.get()][q]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return (double)REAL1_DEFAULT_ARG;
    }
}

/**
 * (External API) Get the probability that a qubit is in the |1> state.
 */
MICROSOFT_QUANTUM_DECL double Prob(_In_ uintq sid, _In_ uintq q) { return _Prob(sid, q, false); }

/**
 * (External API) Get the probability that a qubit is in the |1> state, treating all ancillary qubits as post-selected T
 * gate gadgets.
 */
MICROSOFT_QUANTUM_DECL double ProbRdm(_In_ uintq sid, _In_ uintq q) { return _Prob(sid, q, true); }

double _PermutationProb(uintq sid, uintq n, uintq* q, bool* c, bool isRdm, bool r)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    bitCapInt mask = ZERO_BCI;
    bitCapInt perm = ZERO_BCI;
    for (uintq i = 0U; i < n; ++i) {
        const bitCapInt p = pow2(shards[simulators[sid].get()][q[i]]);
        bi_or_ip(&mask, p);
        if (c[i]) {
            bi_or_ip(&perm, p);
        }
    }

    try {
        return isRdm ? (double)simulator->ProbMaskRdm(r, mask, perm) : (double)simulator->ProbMask(mask, perm);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return (double)REAL1_DEFAULT_ARG;
    }
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double PermutationProb(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) bool* c)
{
    return _PermutationProb(sid, n, q, c, false, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
MICROSOFT_QUANTUM_DECL double PermutationProbRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) bool* c, _In_ bool r)
{
    return _PermutationProb(sid, n, q, c, true, r);
}

double _PermutationExpectation(uintq sid, uintq n, uintq* q, bool r, bool isRdm)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    std::vector<bitLenInt> _q;
    _q.reserve(n);
    for (uintq i = 0U; i < n; ++i) {
        _q.push_back(shards[simulators[sid].get()][q[i]]);
    }

    try {
        return isRdm ? (double)simulator->ExpectationBitsAllRdm(r, _q) : (double)simulator->ExpectationBitsAll(_q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return (double)REAL1_DEFAULT_ARG;
    }
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double PermutationExpectation(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    return _PermutationExpectation(sid, n, q, false, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
MICROSOFT_QUANTUM_DECL double PermutationExpectationRdm(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, bool r)
{
    return _PermutationExpectation(sid, n, q, r, true);
}

double _FactorizedExpectation(uintq sid, uintq n, uintq* q, uintq m, uintq* c, real1_f* f, bool r, bool isRdm)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    std::vector<bitLenInt> _q;
    _q.reserve(n);
    for (uintq i = 0U; i < n; ++i) {
        _q.push_back(shards[simulators[sid].get()][q[i]]);
    }

    const uintq n2 = n << 1U;

    std::vector<bitCapInt> _c;
    if (c) {
        _c.reserve(n2);
#if QBCAPPOW < 7
        for (uintq i = 0U; i < n2; ++i) {
            _c.push_back(c[i]);
        }
#else
        for (uintq i = 0U; i < n2; ++i) {
            bitCapInt perm = ZERO_BCI;
            for (uintq j = 0U; j < m; ++j) {
                bi_lshift_ip(&perm, 64U);
                bi_or_ip(&perm, c[i * m + j]);
            }
            _c.push_back(perm);
        }
#endif
    }

    std::vector<real1_f> _f;
    if (f) {
        _f.reserve(n2);
        for (uintq i = 0U; i < n2; ++i) {
            _f.push_back(f[i]);
        }
    }

    try {
        return c    ? isRdm ? (double)simulator->ExpectationBitsFactorizedRdm(r, _q, _c)
                            : (double)simulator->ExpectationBitsFactorized(_q, _c)
               : isRdm ? (double)simulator->ExpectationFloatsFactorizedRdm(r, _q, _f)
                    : (double)simulator->ExpectationFloatsFactorized(_q, _f);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return (double)REAL1_DEFAULT_ARG;
    }
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double FactorizedExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c)
{
    return _FactorizedExpectation(sid, n, q, m, c, NULL, false, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
MICROSOFT_QUANTUM_DECL double FactorizedExpectationRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c, _In_ bool r)
{
    return _FactorizedExpectation(sid, n, q, m, c, NULL, r, true);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, real1_f* c)
{
    return _FactorizedExpectation(sid, n, q, 0U, NULL, c, false, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFpRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, real1_f* c, _In_ bool r)
{
    return _FactorizedExpectation(sid, n, q, 0U, NULL, c, r, true);
}

MICROSOFT_QUANTUM_DECL void QFT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        std::vector<bitLenInt> q(n);
        for (uintq i = 0U; i < n; ++i) {
            q[i] = shards[simulators[sid].get()][c[i]];
        }
        simulator->QFTR(q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void IQFT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        std::vector<bitLenInt> q(n);
        for (uintq i = 0U; i < n; ++i) {
            q[i] = shards[simulators[sid].get()][c[i]];
        }
        simulator->IQFTR(q);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

#if ENABLE_ALU
MICROSOFT_QUANTUM_DECL void ADD(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, n, q);
        simulator->INC(aTot, start, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void SUB(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, n, q);
        simulator->DEC(aTot, start, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void ADDS(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, uintq s, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, n, q);
        simulator->INCS(aTot, start, n, shards[simulator.get()][s]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void SUBS(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, uintq s, _In_ uintq n, _In_reads_(n) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, n, q);
        simulator->DECS(aTot, start, n, shards[simulator.get()][s]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void MCADD(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq nq, _In_reads_(nq) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, nq, q);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        simulator->CINC(aTot, start, nq, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MCSUB(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq nq, _In_reads_(nq) uintq* q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const uintq start = MapArithmetic(simulator, nq, q);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        simulator->CDEC(aTot, start, nq, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void MUL(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n,
    _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->MUL(aTot, starts.start1, starts.start2, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void DIV(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n,
    _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->DIV(aTot, starts.start1, starts.start2, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MULN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        simulator->MULModNOut(aTot, mTot, starts.start1, starts.start2, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void DIVN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        simulator->IMULModNOut(aTot, mTot, starts.start1, starts.start2, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void POWN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->POWModNOut(aTot, mTot, starts.start1, starts.start2, n);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void MCMUL(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        QALU(simulator)->CMUL(aTot, starts.start1, starts.start2, n, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MCDIV(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        QALU(simulator)->CDIV(aTot, starts.start1, starts.start2, n, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MCMULN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        simulator->CMULModNOut(aTot, mTot, starts.start1, starts.start2, n, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MCDIVN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        simulator->CIMULModNOut(aTot, mTot, starts.start1, starts.start2, n, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void MCPOWN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const bitCapInt aTot = _combineA(na, a);
        const bitCapInt mTot = _combineA(na, m);
        const MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        std::vector<bitLenInt> ctrlsArray(nc);
        for (uintq i = 0; i < nc; ++i) {
            ctrlsArray[i] = shards[simulator.get()][c[i]];
        }
        QALU(simulator)->CPOWModNOut(aTot, mTot, starts.start1, starts.start2, n, ctrlsArray);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void LDA(
    _In_ uintq sid, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv, _In_reads_(nv) uintq* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedLDA(starts.start1, ni, starts.start2, nv, t, true);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void ADC(_In_ uintq sid, uintq s, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv,
    _In_reads_(nv) uintq* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedADC(starts.start1, ni, starts.start2, nv, shards[simulator.get()][s], t);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void SBC(_In_ uintq sid, uintq s, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv,
    _In_reads_(nv) uintq* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedSBC(starts.start1, ni, starts.start2, nv, shards[simulator.get()][s], t);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
MICROSOFT_QUANTUM_DECL void Hash(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        const uintq start = MapArithmetic(simulator, n, q);
        QALU(simulator)->Hash(start, n, t);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
#endif

MICROSOFT_QUANTUM_DECL bool TrySeparate1Qb(_In_ uintq sid, _In_ uintq qi1)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    try {
        return simulators[sid]->TrySeparate(shards[simulator.get()][qi1]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return false;
    }
}

MICROSOFT_QUANTUM_DECL bool TrySeparate2Qb(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    try {
        return simulators[sid]->TrySeparate(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return false;
    }
}

MICROSOFT_QUANTUM_DECL bool TrySeparateTol(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ double tol)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    std::vector<bitLenInt> bitArray(n);
    for (uintq i = 0U; i < n; ++i) {
        bitArray[i] = shards[simulator.get()][q[i]];
    }

    try {
        return simulator->TrySeparate(bitArray, (real1_f)tol);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return false;
    }
}

MICROSOFT_QUANTUM_DECL double GetUnitaryFidelity(_In_ uintq sid)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    try {
        return simulator->GetUnitaryFidelity();
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
        return -1.0;
    }
}

MICROSOFT_QUANTUM_DECL void ResetUnitaryFidelity(_In_ uintq sid)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->ResetUnitaryFidelity();
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void SetSdrp(_In_ uintq sid, _In_ double sdrp)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->SetSdrp(sdrp);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void SetNcrp(_In_ uintq sid, _In_ double ncrp)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->SetNcrp(ncrp);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void SetReactiveSeparate(_In_ uintq sid, _In_ bool irs)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    try {
        simulator->SetReactiveSeparate(irs);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void SetTInjection(_In_ uintq sid, _In_ bool irs)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    try {
        simulator->SetTInjection(irs);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

#if !(FPPOW < 6 && !defined(ENABLE_COMPLEX_X2))
/**
 * (External API) Simulate a Hamiltonian
 */
MICROSOFT_QUANTUM_DECL void TimeEvolve(_In_ uintq sid, _In_ double t, _In_ uintq n,
    _In_reads_(n) _QrackTimeEvolveOpHeader* teos, uintq mn, _In_reads_(mn) double* mtrx)
{
    bitCapIntOcl mtrxOffset = 0U;
    Hamiltonian h(n);
    for (uintq i = 0U; i < n; ++i) {
        h[i] = std::make_shared<UniformHamiltonianOp>(teos[i], mtrx + mtrxOffset);
        mtrxOffset += pow2Ocl(teos[i].controlLen) * 8U;
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)

    try {
        simulator->TimeEvolve(h, (real1_f)t);
    } catch (const std::exception& ex) {
        simulatorErrors[sid] = 1;
        std::cout << ex.what() << std::endl;
    }
}
#endif

MICROSOFT_QUANTUM_DECL uintq init_qneuron(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ uintq f, _In_ double a, _In_ double tol)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;
        metaError = 2;
        return 0U;
    }
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()]));
    if (!simulator) {
        std::cout << "Invalid argument: simulator ID not found!" << std::endl;
        metaError = 2;
        return -1;
    }

    std::vector<bitLenInt> ctrlsArray(n);
    for (uintq i = 0; i < n; ++i) {
        ctrlsArray[i] = shards[simulator.get()][c[i]];
    }

    uintq nid = (uintq)neurons.size();

    for (uintq i = 0U; i < neurons.size(); ++i) {
        if (neuronReservations[i] == false) {
            nid = i;
            neuronReservations[i] = true;
            break;
        }
    }

    QNeuronPtr neuron = std::make_shared<QNeuron>(
        simulator, ctrlsArray, shards[simulator.get()][q], (QNeuronActivationFn)f, (real1_f)a, (real1_f)tol);
    neuronSimulators[neuron] = simulator.get();

    if (nid == neurons.size()) {
        neuronReservations.push_back(true);
        neurons.push_back(neuron);
        neuronErrors.push_back(0);
    } else {
        neuronReservations[nid] = true;
        neurons[nid] = neuron;
        neuronErrors[nid] = 0;
    }

    return nid;
}

MICROSOFT_QUANTUM_DECL uintq clone_qneuron(_In_ uintq nid)
{
    META_LOCK_GUARD()

    if (nid > neurons.size()) {
        std::cout << "Invalid argument: neuron ID not found!" << std::endl;
        metaError = 2;
        return 0U;
    }
    QNeuronPtr neuron = neurons[nid];
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock(
        new const std::lock_guard<std::mutex>(neuronMutexes[neuron.get()]));

    uintq nnid = (uintq)neurons.size();

    for (uintq i = 0U; i < neurons.size(); ++i) {
        if (neuronReservations[i] == false) {
            nnid = i;
            neuronReservations[i] = true;
            break;
        }
    }

    QNeuronPtr nNeuron = std::make_shared<QNeuron>(*neuron);
    neuronSimulators[nNeuron] = neuronSimulators[neuron];

    if (nnid == neurons.size()) {
        neuronReservations.push_back(true);
        neurons.push_back(nNeuron);
        neuronErrors.push_back(0);
    } else {
        neuronReservations[nnid] = true;
        neurons[nnid] = nNeuron;
        neuronErrors[nnid] = 0;
    }

    return nnid;
}

MICROSOFT_QUANTUM_DECL void destroy_qneuron(_In_ uintq nid)
{
    META_LOCK_GUARD()

    neuronMutexes.erase(neurons[nid].get());
    neurons[nid] = NULL;
    neuronErrors[nid] = 0;
    neuronReservations[nid] = false;
}

#if FPPOW < 6
MICROSOFT_QUANTUM_DECL void set_qneuron_angles(_In_ uintq nid, _In_ float* angles)
#else
MICROSOFT_QUANTUM_DECL void set_qneuron_angles(_In_ uintq nid, _In_ double* angles)
#endif
{
    NEURON_LOCK_GUARD_VOID(nid)
#if (FPPOW == 5) || (FPPOW == 6)
    neuron->SetAngles(angles);
#else
    const bitCapIntOcl inputPower = (bitCapIntOcl)neuron->GetInputPower();
    std::unique_ptr<real1[]> _angles(new real1[inputPower]);
#if (FPPOW == 4)
    std::copy(angles, angles + inputPower, _angles.get());
#else
    std::transform(angles, angles + inputPower, _angles.get(), [](double d) { return (real1)d; });
#endif
    neuron->SetAngles(_angles.get());
#endif
}

#if FPPOW < 6
MICROSOFT_QUANTUM_DECL void get_qneuron_angles(_In_ uintq nid, _In_ float* angles)
#else
MICROSOFT_QUANTUM_DECL void get_qneuron_angles(_In_ uintq nid, _In_ double* angles)
#endif
{
    NEURON_LOCK_GUARD_VOID(nid)
#if (FPPOW == 5) || (FPPOW == 6)
    neuron->GetAngles(angles);
#else
    const bitCapIntOcl inputPower = (bitCapIntOcl)neuron->GetInputPower();
    std::unique_ptr<real1[]> _angles(new real1[inputPower]);
    neuron->GetAngles(_angles.get());
#if (FPPOW == 4)
    std::copy(_angles.get(), _angles.get() + inputPower, angles);
#else
    std::transform(_angles.get(), _angles.get() + inputPower, angles, [](real1 d) { return (double)d; });
#endif
#endif
}

MICROSOFT_QUANTUM_DECL void set_qneuron_alpha(_In_ uintq nid, _In_ double alpha)
{
    NEURON_LOCK_GUARD_VOID(nid)
    neuron->SetAlpha((real1_f)alpha);
}

MICROSOFT_QUANTUM_DECL double get_qneuron_alpha(_In_ uintq nid)
{
    NEURON_LOCK_GUARD_DOUBLE(nid)
    return (double)neuron->GetAlpha();
}

MICROSOFT_QUANTUM_DECL void set_qneuron_activation_fn(_In_ uintq nid, _In_ uintq f)
{
    NEURON_LOCK_GUARD_VOID(nid)
    neuron->SetActivationFn((QNeuronActivationFn)f);
}

MICROSOFT_QUANTUM_DECL uintq get_qneuron_activation_fn(_In_ uintq nid)
{
    NEURON_LOCK_GUARD_INT(nid)
    return (uintq)neuron->GetActivationFn();
}

MICROSOFT_QUANTUM_DECL double qneuron_predict(_In_ uintq nid, _In_ bool e, _In_ bool r)
{
    NEURON_LOCK_GUARD_DOUBLE(nid)
    try {
        return (double)neuron->Predict(e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

MICROSOFT_QUANTUM_DECL double qneuron_unpredict(_In_ uintq nid, _In_ bool e)
{
    NEURON_LOCK_GUARD_DOUBLE(nid)
    try {
        return (double)neuron->Unpredict(e);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

MICROSOFT_QUANTUM_DECL double qneuron_learn_cycle(_In_ uintq nid, _In_ bool e)
{
    NEURON_LOCK_GUARD_DOUBLE(nid)
    try {
        return (double)neuron->LearnCycle(e);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

MICROSOFT_QUANTUM_DECL void qneuron_learn(_In_ uintq nid, _In_ double eta, _In_ bool e, _In_ bool r)
{
    NEURON_LOCK_GUARD_VOID(nid)
    try {
        neuron->Learn(eta, e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL void qneuron_learn_permutation(_In_ uintq nid, _In_ double eta, _In_ bool e, _In_ bool r)
{
    NEURON_LOCK_GUARD_VOID(nid)
    try {
        neuron->LearnPermutation(eta, e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

MICROSOFT_QUANTUM_DECL uintq init_qcircuit(_In_ bool collapse, _In_ bool clifford)
{
    META_LOCK_GUARD()
    uintq cid = (uintq)circuits.size();

    for (uintq i = 0U; i < circuits.size(); ++i) {
        if (circuitReservations[i] == false) {
            cid = i;
            circuitReservations[i] = true;
            break;
        }
    }

    QCircuitPtr circuit = std::make_shared<QCircuit>(collapse, clifford);

    if (cid == circuits.size()) {
        circuitReservations.push_back(true);
        circuits.push_back(circuit);
    } else {
        circuitReservations[cid] = true;
        circuits[cid] = circuit;
    }

    return cid;
}

uintq _init_qcircuit_copy(uintq cid, bool isInverse, std::set<bitLenInt> q)
{
    META_LOCK_GUARD()

    if (cid > circuits.size()) {
        std::cout << "Invalid argument: circuit ID not found!" << std::endl;
        metaError = 2;
        return 0U;
    }
    QCircuitPtr circuit = circuits[cid];
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock(
        new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()]));

    uintq ncid = (uintq)circuits.size();

    for (uintq i = 0U; i < circuits.size(); ++i) {
        if (circuitReservations[i] == false) {
            ncid = i;
            circuitReservations[i] = true;
            break;
        }
    }

    QCircuitPtr nCircuit = isInverse ? circuit->Inverse() : (q.size() ? circuit->PastLightCone(q) : circuit->Clone());

    if (ncid == circuits.size()) {
        circuitReservations.push_back(true);
        circuits.push_back(nCircuit);
    } else {
        circuitReservations[ncid] = true;
        circuits[ncid] = nCircuit;
    }

    return ncid;
}

MICROSOFT_QUANTUM_DECL uintq init_qcircuit_clone(_In_ uintq cid) { return _init_qcircuit_copy(cid, false, {}); }

MICROSOFT_QUANTUM_DECL uintq qcircuit_inverse(_In_ uintq cid) { return _init_qcircuit_copy(cid, true, {}); }

MICROSOFT_QUANTUM_DECL uintq qcircuit_past_light_cone(_In_ uintq cid, _In_ uintq n, _In_reads_(n) uintq* q)
{
    std::set<bitLenInt> qubits;
    for (uintq i = 0U; i < n; ++i) {
        qubits.insert((bitLenInt)q[i]);
    }
    return _init_qcircuit_copy(cid, false, qubits);
}

MICROSOFT_QUANTUM_DECL void destroy_qcircuit(_In_ uintq cid)
{
    META_LOCK_GUARD()

    circuitMutexes.erase(circuits[cid].get());
    circuits[cid] = NULL;
    circuitReservations[cid] = false;
}

MICROSOFT_QUANTUM_DECL uintq get_qcircuit_qubit_count(_In_ uintq cid)
{
    CIRCUIT_LOCK_GUARD_INT(cid)
    return circuit->GetQubitCount();
}
MICROSOFT_QUANTUM_DECL void qcircuit_swap(_In_ uintq cid, _In_ uintq q1, _In_ uintq q2)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    circuit->Swap(q1, q2);
}
MICROSOFT_QUANTUM_DECL void qcircuit_append_1qb(_In_ uintq cid, _In_reads_(8) double* m, _In_ uintq q)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };
    circuit->AppendGate(std::make_shared<QCircuitGate>(q, mtrx));
}

MICROSOFT_QUANTUM_DECL void qcircuit_append_mc(
    _In_ uintq cid, _In_reads_(8) double* m, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ uintq p)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)

    std::vector<bitLenInt> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](bitLenInt a, bitLenInt b) -> bool { return c[a] < c[b]; });

    bitCapInt _p = ZERO_BCI;
    std::set<bitLenInt> ctrls;
    for (uintq i = 0U; i < n; ++i) {
        bi_or_ip(&_p, ((p >> i) & 1U) << indices[i]);
        ctrls.insert(c[i]);
    }

    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };
    circuit->AppendGate(std::make_shared<QCircuitGate>(q, mtrx, ctrls, _p));
}

MICROSOFT_QUANTUM_DECL void qcircuit_run(_In_ uintq cid, _In_ uintq sid)
{
    CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)
    circuit->Run(simulator);
}

MICROSOFT_QUANTUM_DECL void qcircuit_out_to_file(_In_ uintq cid, _In_ char* f)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    std::ofstream ofile;
    ofile.open(f);
    ofile << circuit;
    ofile.close();
}
MICROSOFT_QUANTUM_DECL void qcircuit_in_from_file(_In_ uintq cid, _In_ char* f)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    std::ifstream ifile;
    ifile.open(f);
    ifile >> circuit;
    ifile.close();
}
}
