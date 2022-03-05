//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "pinvoke_api.hpp"

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#if !(FPPOW < 6 && !ENABLE_COMPLEX_X2)
#include "hamiltonian.hpp"
#endif

#include <map>
#include <mutex>
#include <vector>

#define META_LOCK_GUARD()                                                                                              \
    const std::lock_guard<std::mutex> metaLock(metaOperationMutex);                                                    \
    std::map<QInterface*, std::mutex>::iterator mutexLockIt;                                                           \
    std::vector<std::unique_ptr<const std::lock_guard<std::mutex>>> simulatorLocks;                                    \
    for (mutexLockIt = simulatorMutexes.begin(); mutexLockIt != simulatorMutexes.end(); mutexLockIt++) {               \
        simulatorLocks.push_back(std::unique_ptr<const std::lock_guard<std::mutex>>(                                   \
            new const std::lock_guard<std::mutex>(mutexLockIt->second)));                                              \
    }

// SIMULATOR_LOCK_GUARD variants will lock simulatorMutexes[NULL], if the requested simulator doesn't exist.
// This is CORRECT behavior. This will effectively emplace a mutex for NULL key.
#define SIMULATOR_LOCK_GUARD(sid)                                                                                      \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex);                                                \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[simulators[sid].get()]));                           \
    }

#define SIMULATOR_LOCK_GUARD_DOUBLE(sid)                                                                               \
    SIMULATOR_LOCK_GUARD(sid)                                                                                          \
    if (!simulators[sid]) {                                                                                            \
        return 0.0;                                                                                                    \
    }

#define SIMULATOR_LOCK_GUARD_BOOL(sid)                                                                                 \
    SIMULATOR_LOCK_GUARD(sid)                                                                                          \
    if (!simulators[sid]) {                                                                                            \
        return false;                                                                                                  \
    }

#define SIMULATOR_LOCK_GUARD_INT(sid)                                                                                  \
    SIMULATOR_LOCK_GUARD(sid)                                                                                          \
    if (!simulators[sid]) {                                                                                            \
        return 0U;                                                                                                     \
    }

#define QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)

using namespace Qrack;

qrack_rand_gen_ptr randNumGen = std::make_shared<qrack_rand_gen>(time(0));
std::mutex metaOperationMutex;
std::vector<int> simulatorErrors;
std::vector<QInterfacePtr> simulators;
std::vector<std::vector<QInterfaceEngine>> simulatorTypes;
std::vector<bool> simulatorHostPointer;
std::map<QInterface*, std::mutex> simulatorMutexes;
std::vector<bool> simulatorReservations;
std::map<QInterface*, std::map<unsigned, bitLenInt>> shards;
bitLenInt _maxShardQubits = 0;
bitLenInt MaxShardQubits()
{
    if (_maxShardQubits == 0) {
#if ENABLE_ENV_VARS
        _maxShardQubits =
            (bitLenInt)(getenv("QRACK_MAX_PAGING_QB") ? std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB"))) : -1);
#else
        _maxShardQubits = -1;
#endif
    }

    return _maxShardQubits;
}

void TransformPauliBasis(QInterfacePtr simulator, unsigned len, int* bases, unsigned* qubitIds)
{
    for (unsigned i = 0; i < len; i++) {
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

void RevertPauliBasis(QInterfacePtr simulator, unsigned len, int* bases, unsigned* qubitIds)
{
    for (unsigned i = 0; i < len; i++) {
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
    unsigned i = 0;
    while (i != b->size()) {
        if ((*b)[i] == PauliI) {
            b->erase(b->begin() + i);
            qs->erase(qs->begin() + i);
        } else {
            ++i;
        }
    }
}

void RHelper(unsigned sid, unsigned b, double phi, unsigned q)
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

void MCRHelper(unsigned sid, unsigned b, double phi, unsigned n, unsigned* c, unsigned q)
{
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<bitLenInt[]> ctrlsArray(new bitLenInt[n]);
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator.get()][c[i]];
    }

    if (b == PauliI) {
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->MCPhase(ctrlsArray.get(), n, phaseFac, phaseFac, shards[simulator.get()][q]);
        return;
    }

    real1 cosine = (real1)cos(phi / 2);
    real1 sine = (real1)sin(phi / 2);
    complex pauliR[4];

    switch (b) {
    case PauliX:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(ZERO_R1, -sine);
        pauliR[2] = complex(ZERO_R1, -sine);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(ctrlsArray.get(), n, pauliR, shards[simulator.get()][q]);
        break;
    case PauliY:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(-sine, ZERO_R1);
        pauliR[2] = complex(sine, ZERO_R1);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(ctrlsArray.get(), n, pauliR, shards[simulator.get()][q]);
        break;
    case PauliZ:
        simulator->MCPhase(
            ctrlsArray.get(), n, complex(cosine, -sine), complex(cosine, sine), shards[simulator.get()][q]);
        break;
    case PauliI:
    default:
        break;
    }
}

inline std::size_t make_mask(std::vector<bitLenInt> const& qs)
{
    std::size_t mask = 0;
    for (std::size_t q : qs)
        mask = mask | pow2Ocl(q);
    return mask;
}

std::map<unsigned, bitLenInt>::iterator FindShardValue(bitLenInt v, std::map<unsigned, bitLenInt>& simMap)
{
    for (auto it = simMap.begin(); it != simMap.end(); it++) {
        if (it->second == v) {
            // We have the matching it1, if we break.
            return it;
        }
    }

    return simMap.end();
}

void SwapShardValues(bitLenInt v1, bitLenInt v2, std::map<unsigned, bitLenInt>& simMap)
{
    auto it1 = FindShardValue(v1, simMap);
    auto it2 = FindShardValue(v2, simMap);
    std::swap(it1->second, it2->second);
}

unsigned MapArithmetic(QInterfacePtr simulator, unsigned n, unsigned* q)
{
    unsigned start = shards[simulator.get()][q[0]];
    std::unique_ptr<bitLenInt[]> bitArray(new bitLenInt[n]);
    for (unsigned i = 0U; i < n; i++) {
        bitArray[i] = shards[simulator.get()][q[i]];
        if (start > bitArray[i]) {
            start = bitArray[i];
        }
    }
    for (unsigned i = 0U; i < n; i++) {
        simulator->Swap(start + i, bitArray[i]);
        SwapShardValues(start + i, bitArray[i], shards[simulator.get()]);
    }

    return start;
}

struct MapArithmeticResult2 {
    unsigned start1;
    unsigned start2;

    MapArithmeticResult2(unsigned s1, unsigned s2)
        : start1(s1)
        , start2(s2)
    {
    }
};

MapArithmeticResult2 MapArithmetic2(QInterfacePtr simulator, unsigned n, unsigned* q1, unsigned* q2)
{
    unsigned start1 = shards[simulator.get()][q1[0]];
    unsigned start2 = shards[simulator.get()][q2[0]];
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[n]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[n]);
    for (unsigned i = 0; i < n; i++) {
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

    for (unsigned i = 0U; i < n; i++) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }

    if ((start1 + n) > start2) {
        start2 = start1 + n;
    }

    for (unsigned i = 0U; i < n; i++) {
        simulator->Swap(start2 + i, bitArray2[i]);
        SwapShardValues(start2 + i, bitArray2[i], shards[simulator.get()]);
    }

    if (isReversed) {
        std::swap(start1, start2);
    }

    return MapArithmeticResult2(start1, start2);
}

MapArithmeticResult2 MapArithmetic3(QInterfacePtr simulator, unsigned n1, unsigned* q1, unsigned n2, unsigned* q2)
{
    unsigned start1 = shards[simulator.get()][q1[0]];
    unsigned start2 = shards[simulator.get()][q2[0]];
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[n1]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[n2]);
    for (unsigned i = 0; i < n1; i++) {
        bitArray1[i] = shards[simulator.get()][q1[i]];
        if (start1 > bitArray1[i]) {
            start1 = bitArray1[i];
        }
    }

    for (unsigned i = 0; i < n2; i++) {
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

    for (unsigned i = 0U; i < n1; i++) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }

    if ((start1 + n1) > start2) {
        start2 = start1 + n1;
    }

    for (unsigned i = 0U; i < n2; i++) {
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
    for (bitCapIntOcl j = 0; j < componentCount; j++) {
        amps[j] = complex(real1(params[2 * j]), real1(params[2 * j + 1]));
    }
}

extern "C" {

/**
 * (External API) Poll after each operation to check whether error occurred.
 */
MICROSOFT_QUANTUM_DECL int get_error(_In_ unsigned sid) { return simulatorErrors[sid]; }

/**
 * (External API) Initialize a simulator ID with "q" qubits and explicit layer options on/off
 */
MICROSOFT_QUANTUM_DECL unsigned init_count_type(_In_ unsigned q, _In_ bool md, _In_ bool sd, _In_ bool sh,
    _In_ bool bdt, _In_ bool pg, _In_ bool zxf, _In_ bool hy, _In_ bool oc, _In_ bool hp)
{
    META_LOCK_GUARD()

    unsigned sid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

#if ENABLE_OPENCL
    bool isOcl = oc && (OCLEngine::Instance().GetDeviceCount() > 0);
    bool isOclMulti = oc && md && (OCLEngine::Instance().GetDeviceCount() > 1);
#else
    bool isOclMulti = false;
#endif

    // Construct backwards, then reverse:
    std::vector<QInterfaceEngine> simulatorType;

#if ENABLE_OPENCL
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
    }
#endif

    if (zxf && (!pg || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_MASK_FUSION);
    }

    if (pg && (bdt || !sh || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_QPAGER);
    }

    if (bdt) {
        simulatorType.push_back(QINTERFACE_BDT);
    }

    if (sh && (!sd || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

    if (sd) {
        simulatorType.push_back(isOclMulti ? QINTERFACE_QUNIT_MULTI : QINTERFACE_QUNIT);
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
#else
        simulatorType.push_back(QINTERFACE_CPU);
#endif
    }

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator = CreateQuantumInterface(simulatorType, q, 0, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
        } catch (...) {
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
    for (unsigned i = 0; i < q; i++) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
MICROSOFT_QUANTUM_DECL unsigned init_count(_In_ unsigned q, _In_ bool hp)
{
    META_LOCK_GUARD()

    unsigned sid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    std::vector<QInterfaceEngine> simulatorType;

#if ENABLE_OPENCL
    simulatorType.push_back(
        (OCLEngine::Instance().GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#else
    simulatorType.push_back(QINTERFACE_OPTIMAL);
#endif

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator = CreateQuantumInterface(simulatorType, q, 0, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
        } catch (...) {
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
    for (unsigned i = 0; i < q; i++) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
MICROSOFT_QUANTUM_DECL unsigned init_count_pager(_In_ unsigned q, _In_ bool hp)
{
    META_LOCK_GUARD()

    unsigned sid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    std::vector<QInterfaceEngine> simulatorType;

    simulatorType.push_back(QINTERFACE_OPTIMAL);

    std::vector<int> deviceList;
#if ENABLE_OPENCL
    std::vector<DeviceContextPtr> deviceContext = OCLEngine::Instance().GetDeviceContextPtrVector();
    for (size_t i = 0; i < deviceContext.size(); i++) {
        deviceList.push_back(i);
    }
    const size_t defaultDeviceID = OCLEngine::Instance().GetDefaultDeviceID();
    std::swap(deviceList[0], deviceList[defaultDeviceID]);
#endif

    bool isSuccess = true;
    QInterfacePtr simulator = NULL;
    if (q) {
        try {
            simulator = CreateQuantumInterface(simulatorType, q, 0, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp, -1,
                true, false, REAL1_EPSILON, deviceList, 0, FP_NORM_EPSILON);
        } catch (...) {
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
    for (unsigned i = 0; i < q; i++) {
        shards[simulator.get()][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID that clones simulator ID "sid"
 */
MICROSOFT_QUANTUM_DECL unsigned init_clone(_In_ unsigned sid)
{
    META_LOCK_GUARD()

    unsigned nsid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            nsid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    bool isSuccess = true;
    QInterfacePtr simulator;
    try {
        simulator = simulators[sid]->Clone();
    } catch (...) {
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
    for (unsigned i = 0; i < simulator->GetQubitCount(); i++) {
        shards[simulator.get()][i] = shards[simulators[sid].get()][i];
    }

    return nsid;
}

/**
 * (External API) Destroy a simulator (ID will not be reused)
 */
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid)
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
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s)
{
    SIMULATOR_LOCK_GUARD(sid)

    try {
        simulators[sid]->SetRandomSeed(s);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Set concurrency level per QEngine shard
 */
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ unsigned sid, _In_ unsigned p)
{
    SIMULATOR_LOCK_GUARD(sid)

    try {
        simulators[sid]->SetConcurrency(p);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void DumpIds(_In_ unsigned sid, _In_ IdCallback callback)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    std::map<unsigned, bitLenInt>::iterator it;
    for (it = shards[simulator.get()].begin(); it != shards[simulator.get()].end(); it++) {
        callback(it->first);
    }
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void Dump(_In_ unsigned sid, _In_ ProbAmpCallback callback)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitCapIntOcl wfnl = (bitCapIntOcl)simulator->GetMaxQPower();
    std::unique_ptr<complex[]> wfn(new complex[wfnl]);
    try {
        simulator->GetQuantumState(wfn.get());
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
    for (size_t i = 0; i < wfnl; i++) {
        if (!callback(i, real(wfn[i]), imag(wfn[i]))) {
            break;
        }
    }
}

/**
 * (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
 */
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
    std::discrete_distribution<std::size_t> dist(p, p + n);
    return dist(*randNumGen.get());
}

double _JointEnsembleProbabilityHelper(QInterfacePtr simulator, unsigned n, int* b, unsigned* q, bool doMeasure)
{

    if (n == 0) {
        return 0.0;
    }

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    removeIdentities(&bVec, &qVec);
    n = (unsigned)qVec.size();

    if (n == 0) {
        return 0.0;
    }

    bitCapInt mask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)n; i++) {
        bitCapInt bit = pow2(shards[simulator.get()][qVec[i]]);
        mask |= bit;
    }

    return (double)(doMeasure ? (simulator->MParity(mask) ? ONE_R1 : ZERO_R1) : simulator->ProbParity(mask));
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    QInterfacePtr simulator = simulators[sid];

    double jointProb = (double)REAL1_DEFAULT_ARG;

    try {
        TransformPauliBasis(simulator, n, b, q);

        jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, false);

        RevertPauliBasis(simulator, n, b, q);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }

    return jointProb;
}

MICROSOFT_QUANTUM_DECL void PhaseParity(
    _In_ unsigned sid, _In_ double lambda, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];
    bitCapInt mask = 0;
    for (unsigned i = 0; i < n; i++) {
        mask |= pow2(shards[simulator.get()][q[i]]);
    }

    try {
        simulator->PhaseParity((real1_f)lambda, mask);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Set the simulator to a computational basis permutation.
 */
MICROSOFT_QUANTUM_DECL void ResetAll(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD(sid)

    try {
        simulators[sid]->SetPermutation(0);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
    META_LOCK_GUARD()

    QInterfacePtr nQubit = CreateQuantumInterface(
        simulatorTypes[sid], 1, 0, randNumGen, CMPLX_DEFAULT_ARG, false, true, simulatorHostPointer[sid]);

    if (simulators[sid] == NULL) {
        simulators[sid] = nQubit;
        shards[nQubit.get()] = {};
        shards[nQubit.get()][qid] = 0;

        return;
    }

    bitLenInt qubitCount = -1;
    try {
        simulators[sid]->Compose(nQubit);
        qubitCount = simulators[sid]->GetQubitCount();
    } catch (...) {
        simulatorErrors[sid] = 1;
    }

    shards[simulators[sid].get()][qid] = (qubitCount - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL bool release(_In_ unsigned sid, _In_ unsigned q)
{
    META_LOCK_GUARD()

    QInterfacePtr simulator = simulators[sid];

    // Check that the qubit is in the |0> state, to within a small tolerance.
    bool toRet = simulator->Prob(shards[simulator.get()][q]) < (ONE_R1 / 100);

    if (simulator->GetQubitCount() == 1U) {
        shards[simulator.get()] = {};
        simulators[sid] = NULL;
    } else {
        bitLenInt oIndex = shards[simulator.get()][q];
        simulator->Dispose(oIndex, 1U);
        for (unsigned i = 0; i < shards[simulator.get()].size(); i++) {
            if (shards[simulator.get()][i] > oIndex) {
                shards[simulator.get()][i]--;
            }
        }
        shards[simulator.get()].erase(q);
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    try {
        return (unsigned)simulators[sid]->GetQubitCount();
    } catch (...) {
        simulatorErrors[sid] = 1;
        return -1;
    }
}

/**
 * (External API) "X" Gate
 */
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->X(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->Y(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->Z(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->H(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "S" Gate
 */
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->S(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "T" Gate
 */
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->T(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->IS(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->IT(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void U(
    _In_ unsigned sid, _In_ unsigned q, _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->U(shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void Mtrx(_In_ unsigned sid, _In_reads_(8) double* m, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->Mtrx(mtrx, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

#define MAP_CONTROLS_AND_LOCK(sid, numC)                                                                               \
    SIMULATOR_LOCK_GUARD(sid)                                                                                          \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    std::unique_ptr<bitLenInt[]> ctrlsArray(new bitLenInt[numC]);                                                      \
    for (unsigned i = 0; i < numC; i++) {                                                                              \
        ctrlsArray[i] = shards[simulator.get()][c[i]];                                                                 \
    }

/**
 * (External API) Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCInvert(ctrlsArray.get(), n, ONE_CMPLX, ONE_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCInvert(ctrlsArray.get(), n, -I_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray.get(), n, ONE_CMPLX, -ONE_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    const complex hGate[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCMtrx(ctrlsArray.get(), n, hGate, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray.get(), n, ONE_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray.get(), n, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(ctrlsArray.get(), n, ONE_CMPLX, -I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCPhase(
            ctrlsArray.get(), n, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->CU(ctrlsArray.get(), n, shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q)
{
    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MCMtrx(ctrlsArray.get(), n, mtrx, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MACX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACInvert(ctrlsArray.get(), n, ONE_CMPLX, ONE_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MACY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACInvert(ctrlsArray.get(), n, -I_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MACZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray.get(), n, ONE_CMPLX, -ONE_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MACH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    const complex hGate[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACMtrx(ctrlsArray.get(), n, hGate, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray.get(), n, ONE_CMPLX, I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(
            ctrlsArray.get(), n, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(ctrlsArray.get(), n, ONE_CMPLX, -I_CMPLX, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) "Anti-"Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACPhase(
            ctrlsArray.get(), n, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->AntiCU(
            ctrlsArray.get(), n, shards[simulator.get()][q], (real1_f)theta, (real1_f)phi, (real1_f)lambda);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q)
{
    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->MACMtrx(ctrlsArray.get(), n, mtrx, shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void Multiplex1Mtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q, double* m)
{
    bitCapIntOcl componentCount = 4U * pow2Ocl(n);
    std::unique_ptr<complex[]> mtrxs(new complex[componentCount]);
    _darray_to_creal1_array(m, componentCount, mtrxs.get());

    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->UniformlyControlledSingleBit(ctrlsArray.get(), n, shards[simulator.get()][q], mtrxs.get());
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    try {
        RHelper(sid, b, phi, q);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    try {
        MCRHelper(sid, b, phi, n, c, q);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void Exp(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    try {
        if (bVec.size() == 0) {
            RHelper(sid, PauliI, -2. * phi, someQubit);
        } else if (bVec.size() == 1) {
            RHelper(sid, bVec.front(), -2. * phi, qVec.front());
        } else {
            QInterfacePtr simulator = simulators[sid];

            TransformPauliBasis(simulator, n, b, q);

            std::size_t mask = make_mask(qVec);
            simulator->UniformParityRZ((bitCapInt)mask, (real1_f)(-phi));

            RevertPauliBasis(simulator, n, b, q);
        }
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi,
    _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    try {
        if (bVec.size() == 0) {
            MCRHelper(sid, PauliI, -2. * phi, nc, cs, someQubit);
        } else if (bVec.size() == 1) {
            MCRHelper(sid, bVec.front(), -2. * phi, nc, cs, qVec.front());
        } else {
            QInterfacePtr simulator = simulators[sid];
            std::vector<bitLenInt> csVec(cs, cs + nc);

            TransformPauliBasis(simulator, n, b, q);

            std::size_t mask = make_mask(qVec);
            simulator->CUniformParityRZ(&(csVec[0]), csVec.size(), (bitCapInt)mask, (real1_f)(-phi));

            RevertPauliBasis(simulator, n, b, q);
        }
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        return simulator->M(shards[simulator.get()][q]) ? 1U : 0U;
    } catch (...) {
        simulatorErrors[sid] = 1;
        return -1;
    }
}

/**
 * (External API) Measure all bits separately in |0>/|1> basis, and return the result in low-to-high order corresponding
 * with first-to-last in original order of allocation.
 */
MICROSOFT_QUANTUM_DECL unsigned MAll(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    try {
        return (unsigned)simulators[sid]->MAll();
    } catch (...) {
        simulatorErrors[sid] = 1;
        return -1;
    }
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
MICROSOFT_QUANTUM_DECL unsigned Measure(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    QInterfacePtr simulator = simulators[sid];

    unsigned toRet = -1;
    try {
        TransformPauliBasis(simulator, n, b, q);

        double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, true);

        toRet = (jointProb < (ONE_R1 / 2)) ? 0U : 1U;

        RevertPauliBasis(simulator, n, b, q);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL void MeasureShots(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ unsigned s, _In_reads_(s) unsigned* m)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<bitCapInt[]> qPowers(new bitCapInt[n]);
    for (unsigned i = 0; i < n; i++) {
        qPowers[i] = Qrack::pow2(shards[simulator.get()][q[i]]);
    }

    try {
        simulator->MultiShotMeasureMask(qPowers.get(), n, s, m);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void SWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->Swap(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void ISWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->ISwap(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void FSim(
    _In_ unsigned sid, _In_ double theta, _In_ double phi, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->FSim((real1_f)theta, (real1_f)phi, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->CSwap(ctrlsArray.get(), n, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void ACSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2)
{
    MAP_CONTROLS_AND_LOCK(sid, n)
    try {
        simulator->AntiCSwap(ctrlsArray.get(), n, shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void Compose(_In_ unsigned sid1, _In_ unsigned sid2, unsigned* q)
{
    if (!simulators[sid1] || !simulators[sid2]) {
        return;
    }
    const std::lock_guard<std::mutex> simulatorLock1(simulatorMutexes[simulators[sid1].get()]);
    const std::lock_guard<std::mutex> simulatorLock2(simulatorMutexes[simulators[sid2].get()]);

    if (simulatorTypes[sid1].size() != simulatorTypes[sid2].size()) {
        throw std::runtime_error("Cannot 'Compose()' simulators of different layer stack types.");
    }

    for (size_t i = 0; i < simulatorTypes[sid1].size(); i++) {
        if (simulatorTypes[sid1][i] != simulatorTypes[sid2][i]) {
            throw std::runtime_error("Cannot 'Compose()' simulators of different layer stack types.");
        }
    }

    QInterfacePtr simulator1 = simulators[sid1];
    QInterfacePtr simulator2 = simulators[sid2];
    bitLenInt oQubitCount = 0;
    bitLenInt pQubitCount = 0;
    try {
        oQubitCount = simulator1->GetQubitCount();
        pQubitCount = simulator2->GetQubitCount();
        simulator1->Compose(simulator2);
    } catch (...) {
        simulatorErrors[sid1] = 1;
        simulatorErrors[sid2] = 1;
    }

    for (bitLenInt i = 0; i < pQubitCount; i++) {
        shards[simulator1.get()][q[i]] = oQubitCount + i;
    }
}

MICROSOFT_QUANTUM_DECL unsigned Decompose(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    unsigned nSid = init_count(n, false);

    SIMULATOR_LOCK_GUARD_INT(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt nQubitIndex = 0;

    try {
        nQubitIndex = simulator->GetQubitCount() - n;

        for (unsigned i = 0; i < n; i++) {
            simulator->Swap(shards[simulator.get()][q[i]], i + nQubitIndex);
        }

        simulator->Decompose(nQubitIndex, simulators[nSid]);
    } catch (...) {
        simulatorErrors[sid] = 1;
        simulatorErrors[nSid] = 1;
    }

    bitLenInt oIndex;
    for (unsigned j = 0; j < n; j++) {
        oIndex = shards[simulator.get()][q[j]];
        for (unsigned i = 0; i < shards[simulator.get()].size(); i++) {
            if (shards[simulator.get()][i] > oIndex) {
                shards[simulator.get()][i]--;
            }
        }
        shards[simulator.get()].erase(q[j]);
    }

    simulatorTypes[nSid] = simulatorTypes[sid];
    simulatorHostPointer[nSid] = simulatorHostPointer[sid];

    return nSid;
}

MICROSOFT_QUANTUM_DECL void Dispose(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt nQubitIndex = 0;

    try {
        nQubitIndex = simulator->GetQubitCount() - n;

        for (unsigned i = 0; i < n; i++) {
            simulator->Swap(shards[simulator.get()][q[i]], i + nQubitIndex);
        }

        simulator->Dispose(nQubitIndex, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }

    bitLenInt oIndex;
    for (unsigned j = 0; j < n; j++) {
        oIndex = shards[simulator.get()][q[j]];
        for (unsigned i = 0; i < shards[simulator.get()].size(); i++) {
            if (shards[simulator.get()][i] > oIndex) {
                shards[simulator.get()][i]--;
            }
        }
        shards[simulator.get()].erase(q[j]);
    }
}

MICROSOFT_QUANTUM_DECL void AND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->AND(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void OR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->OR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void XOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->XOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void NAND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->NAND(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void NOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->NOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void XNOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->XNOR(shards[simulator.get()][qi1], shards[simulator.get()][qi2], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLAND(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLXOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLXOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLNAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLNAND(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLNOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
        simulator->CLXNOR(ci, shards[simulator.get()][qi], shards[simulator.get()][qo]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

/**
 * (External API) Get the probability that a qubit is in the |1> state.
 */
MICROSOFT_QUANTUM_DECL double Prob(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    try {
        QInterfacePtr simulator = simulators[sid];
        return simulator->Prob(shards[simulator.get()][q]);
    } catch (...) {
        simulatorErrors[sid] = 1;
        return (double)REAL1_DEFAULT_ARG;
    }
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double PermutationExpectation(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD_DOUBLE(sid)

    std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
    std::copy(c, c + n, q.get());

    try {
        QInterfacePtr simulator = simulators[sid];
        double result = simulator->ExpectationBitsAll(q.get(), n);
        return result;
    } catch (...) {
        simulatorErrors[sid] = 1;
        return (double)REAL1_DEFAULT_ARG;
    }
}

MICROSOFT_QUANTUM_DECL void QFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
#if (QBCAPPOW >= 16) && (QBCAPPOW < 32)
        simulator->QFTR(c, n);
#else
        std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
        std::copy(c, c + n, q.get());
        simulator->QFTR(q.get(), n);
#endif
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void IQFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    try {
#if (QBCAPPOW >= 16) && (QBCAPPOW < 32)
        simulator->IQFTR(c, n);
#else
        std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
        std::copy(c, c + n, q.get());
        simulator->IQFTR(q.get(), n);
#endif
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

#if ENABLE_ALU
MICROSOFT_QUANTUM_DECL void ADD(_In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        unsigned start = MapArithmetic(simulator, n, q);
        simulator->INC(a, start, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void SUB(_In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        unsigned start = MapArithmetic(simulator, n, q);
        simulator->DEC(a, start, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void ADDS(_In_ unsigned sid, unsigned a, unsigned s, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        unsigned start = MapArithmetic(simulator, n, q);
        simulator->INCS(a, start, n, shards[simulator.get()][s]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void SUBS(_In_ unsigned sid, unsigned a, unsigned s, _In_ unsigned n, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        unsigned start = MapArithmetic(simulator, n, q);
        simulator->DECS(a, start, n, shards[simulator.get()][s]);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void MCADD(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned nq, _In_reads_(nq) unsigned* q)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        unsigned start = MapArithmetic(simulator, nq, q);
        simulator->CINC(a, start, nq, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MCSUB(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned nq, _In_reads_(nq) unsigned* q)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        unsigned start = MapArithmetic(simulator, nq, q);
        simulator->CDEC(a, start, nq, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void MUL(
    _In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->MUL(a, starts.start1, starts.start2, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void DIV(
    _In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->DIV(a, starts.start1, starts.start2, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MULN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->MULModNOut(a, m, starts.start1, starts.start2, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void DIVN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->IMULModNOut(a, m, starts.start1, starts.start2, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void POWN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->POWModNOut(a, m, starts.start1, starts.start2, n);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void MCMUL(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->CMUL(a, starts.start1, starts.start2, n, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MCDIV(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->CDIV(a, starts.start1, starts.start2, n, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MCMULN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->CMULModNOut(a, m, starts.start1, starts.start2, n, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MCDIVN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->CIMULModNOut(a, m, starts.start1, starts.start2, n, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void MCPOWN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o)
{
    MAP_CONTROLS_AND_LOCK(sid, nc)

    try {
        MapArithmeticResult2 starts = MapArithmetic2(simulator, n, q, o);
        QALU(simulator)->CPOWModNOut(a, m, starts.start1, starts.start2, n, ctrlsArray.get(), nc);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

MICROSOFT_QUANTUM_DECL void LDA(_In_ unsigned sid, _In_ unsigned ni, _In_reads_(ni) unsigned* qi, _In_ unsigned nv,
    _In_reads_(nv) unsigned* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedLDA(starts.start1, ni, starts.start2, nv, t, true);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void ADC(_In_ unsigned sid, unsigned s, _In_ unsigned ni, _In_reads_(ni) unsigned* qi,
    _In_ unsigned nv, _In_reads_(nv) unsigned* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedADC(starts.start1, ni, starts.start2, nv, shards[simulator.get()][s], t);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void SBC(_In_ unsigned sid, unsigned s, _In_ unsigned ni, _In_reads_(ni) unsigned* qi,
    _In_ unsigned nv, _In_reads_(nv) unsigned* qv, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        MapArithmeticResult2 starts = MapArithmetic3(simulator, ni, qi, nv, qv);
        QALU(simulator)->IndexedSBC(starts.start1, ni, starts.start2, nv, shards[simulator.get()][s], t);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
MICROSOFT_QUANTUM_DECL void Hash(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, unsigned char* t)
{
    SIMULATOR_LOCK_GUARD(sid)
    QInterfacePtr simulator = simulators[sid];

    try {
        unsigned start = MapArithmetic(simulator, n, q);
        QALU(simulator)->Hash(start, n, t);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
#endif

MICROSOFT_QUANTUM_DECL bool TrySeparate1Qb(_In_ unsigned sid, _In_ unsigned qi1)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    try {
        QInterfacePtr simulator = simulators[sid];
        return simulators[sid]->TrySeparate(shards[simulator.get()][qi1]);
    } catch (...) {
        simulatorErrors[sid] = 1;
        return 1;
    }
}

MICROSOFT_QUANTUM_DECL bool TrySeparate2Qb(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    try {
        QInterfacePtr simulator = simulators[sid];
        return simulators[sid]->TrySeparate(shards[simulator.get()][qi1], shards[simulator.get()][qi2]);
    } catch (...) {
        simulatorErrors[sid] = 1;
        return 1;
    }
}

MICROSOFT_QUANTUM_DECL bool TrySeparateTol(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ double tol)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)

    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<bitLenInt[]> bitArray(new bitLenInt[n]);
    for (unsigned i = 0; i < n; i++) {
        bitArray[i] = shards[simulator.get()][q[i]];
    }

    try {
        QInterfacePtr simulator = simulators[sid];
        return simulator->TrySeparate(bitArray.get(), (bitLenInt)n, (real1_f)tol);
    } catch (...) {
        simulatorErrors[sid] = 1;
        return 1;
    }
}

MICROSOFT_QUANTUM_DECL void SetReactiveSeparate(_In_ unsigned sid, _In_ bool irs)
{
    SIMULATOR_LOCK_GUARD(sid)
    try {
        simulators[sid]->SetReactiveSeparate(irs);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}

#if !(FPPOW < 6 && !ENABLE_COMPLEX_X2)
/**
 * (External API) Simulate a Hamiltonian
 */
MICROSOFT_QUANTUM_DECL void TimeEvolve(_In_ unsigned sid, _In_ double t, _In_ unsigned n,
    _In_reads_(n) _QrackTimeEvolveOpHeader* teos, unsigned mn, _In_reads_(mn) double* mtrx)
{
    bitCapIntOcl mtrxOffset = 0;
    Hamiltonian h(n);
    for (unsigned i = 0; i < n; i++) {
        h[i] = std::make_shared<UniformHamiltonianOp>(teos[i], mtrx + mtrxOffset);
        mtrxOffset += pow2Ocl(teos[i].controlLen) * 8U;
    }

    SIMULATOR_LOCK_GUARD(sid)

    try {
        QInterfacePtr simulator = simulators[sid];
        simulator->TimeEvolve(h, (real1)t);
    } catch (...) {
        simulatorErrors[sid] = 1;
    }
}
#endif
}
