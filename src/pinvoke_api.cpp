//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "pinvoke_api.hpp"
#include "hamiltonian.hpp"

// for details.

#include <map>
#include <mutex>
#include <vector>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

std::mutex metaOperationMutex;

#define META_LOCK_GUARD() const std::lock_guard<std::mutex> metaLock(metaOperationMutex);
// TODO: By design, Qrack should be able to support per-simulator lock guards, in a multithreaded OCL environment. This
// feature might not yet be fully realized.
#define SIMULATOR_LOCK_GUARD(sid) const std::lock_guard<std::mutex> metaLock(metaOperationMutex);

using namespace Qrack;

enum Pauli {
    /// Pauli Identity operator. Corresponds to Q# constant "PauliI."
    PauliI = 0,
    /// Pauli X operator. Corresponds to Q# constant "PauliX."
    PauliX = 1,
    /// Pauli Y operator. Corresponds to Q# constant "PauliY."
    PauliY = 3,
    /// Pauli Z operator. Corresponds to Q# constant "PauliZ."
    PauliZ = 2
};

qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>(time(0));
std::vector<QInterfacePtr> simulators;
std::vector<bool> simulatorReservations;
std::map<QInterfacePtr, std::map<unsigned, bitLenInt>> shards;

void TransformPauliBasis(QInterfacePtr simulator, unsigned len, int* bases, unsigned* qubitIds)
{
    for (unsigned i = 0; i < len; i++) {
        switch (bases[i]) {
        case PauliX:
            simulator->H(shards[simulator][qubitIds[i]]);
            break;
        case PauliY:
            simulator->IS(shards[simulator][qubitIds[i]]);
            simulator->H(shards[simulator][qubitIds[i]]);
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
            simulator->H(shards[simulator][qubitIds[i]]);
            break;
        case PauliY:
            simulator->H(shards[simulator][qubitIds[i]]);
            simulator->S(shards[simulator][qubitIds[i]]);
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
        complex phaseFac = std::exp(complex(ZERO_R1, phi / 4));
        simulator->ApplySinglePhase(phaseFac, phaseFac, shards[simulator][q]);
        break;
    }
    case PauliX:
        simulator->RX(phi, shards[simulator][q]);
        break;
    case PauliY:
        simulator->RY(phi, shards[simulator][q]);
        break;
    case PauliZ:
        simulator->RZ(phi, shards[simulator][q]);
        break;
    default:
        break;
    }
}

void MCRHelper(unsigned sid, unsigned b, double phi, unsigned n, unsigned* c, unsigned q)
{
    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    if (b == PauliI) {
        complex phaseFac = std::exp(complex(ZERO_R1, phi / 4));
        simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], phaseFac, phaseFac);
        return;
    }

    real1 cosine = cos(phi / 2);
    real1 sine = sin(phi / 2);
    complex pauliR[4];

    switch (b) {
    case PauliX:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(ZERO_R1, -sine);
        pauliR[2] = complex(ZERO_R1, -sine);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
        break;
    case PauliY:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(-sine, ZERO_R1);
        pauliR[2] = complex(sine, ZERO_R1);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
        break;
    case PauliZ:
        simulator->ApplyControlledSinglePhase(
            ctrlsArray, n, shards[simulator][q], complex(cosine, -sine), complex(cosine, sine));
        break;
    case PauliI:
    default:
        break;
    }

    delete[] ctrlsArray;
}

inline std::size_t make_mask(std::vector<bitLenInt> const& qs)
{
    std::size_t mask = 0;
    for (std::size_t q : qs)
        mask = mask | pow2Ocl(q);
    return mask;
}

extern "C" {

/**
 * (External API) Initialize a simulator ID with 0 qubits
 */
MICROSOFT_QUANTUM_DECL unsigned init() { return init_count(0); }

/**
 * (External API) Initialize a simulator ID with "q" qubits
 */
MICROSOFT_QUANTUM_DECL unsigned init_count(_In_ unsigned q)
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

    QInterfacePtr simulator = q ? CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, q, 0, rng) : NULL;
    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
    }

    if (!q) {
        return sid;
    }

    shards[simulator] = {};
    for (unsigned i = 0; i < q; i++) {
        shards[simulator][i] = (bitLenInt)i;
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

    QInterfacePtr simulator = simulators[sid]->Clone();
    if (nsid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
    } else {
        simulatorReservations[nsid] = true;
        simulators[nsid] = simulator;
    }

    shards[simulator] = {};
    for (unsigned i = 0; i < simulator->GetQubitCount(); i++) {
        shards[simulator][i] = shards[simulators[sid]][i];
    }

    return nsid;
}

/**
 * (External API) Destroy a simulator (ID will not be reused)
 */
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid)
{
    META_LOCK_GUARD()
    // SIMULATOR_LOCK_GUARD(sid)

    shards.erase(simulators[sid]);
    simulators[sid] = NULL;
    simulatorReservations[sid] = false;
}

/**
 * (External API) Set RNG seed for simulator ID
 */
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] != NULL) {
        simulators[sid]->SetRandomSeed(s);
    }
}

/**
 * (External API) Set concurrency level per QEngine shard
 */
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ unsigned sid, _In_ unsigned p)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] != NULL) {
        simulators[sid]->SetConcurrency(p);
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

    for (it = shards[simulator].begin(); it != shards[simulator].end(); it++) {
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
    complex* wfn = new complex[wfnl];
    simulator->GetQuantumState(wfn);
    for (size_t i = 0; i < wfnl; i++) {
        if (!callback(i, real(wfn[i]), imag(wfn[i]))) {
            break;
        }
    }
    delete[] wfn;
}

/**
 * (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
 */
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
    std::discrete_distribution<std::size_t> dist(p, p + n);
    return dist(*rng.get());
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
    for (bitLenInt i = 0; i < n; i++) {
        bitCapInt bit = pow2(shards[simulator][qVec[i]]);
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
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, false);

    RevertPauliBasis(simulator, n, b, q);

    return jointProb;
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr nQubit = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, 1, 0, rng);
    if (simulators[sid] == NULL) {
        simulators[sid] = nQubit;
        shards[simulators[sid]] = {};
    } else {
        simulators[sid]->Compose(nQubit);
    }
    shards[simulators[sid]][qid] = (simulators[sid]->GetQubitCount() - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL bool release(_In_ unsigned sid, _In_ unsigned q)
{
    QInterfacePtr simulator = simulators[sid];

    // Check that the qubit is in the |0> state, to within a small tolerance.
    bool toRet = simulator->Prob(shards[simulator][q]) < (ONE_R1 / 100);

    if (simulator->GetQubitCount() == 1U) {
        shards.erase(simulator);
        simulators[sid] = NULL;
    } else {
        SIMULATOR_LOCK_GUARD(sid)
        bitLenInt oIndex = shards[simulator][q];
        simulator->Dispose(oIndex, 1U);
        for (unsigned i = 0; i < shards[simulator].size(); i++) {
            if (shards[simulator][i] > oIndex) {
                shards[simulator][i]--;
            }
        }
        shards[simulator].erase(q);
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] == NULL) {
        return 0U;
    }

    return (unsigned)simulators[sid]->GetQubitCount();
}

/**
 * (External API) "X" Gate
 */
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->X(shards[simulator][q]);
}

/**
 * (External API) "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Y(shards[simulator][q]);
}

/**
 * (External API) "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Z(shards[simulator][q]);
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->H(shards[simulator][q]);
}

/**
 * (External API) "S" Gate
 */
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->S(shards[simulator][q]);
}

/**
 * (External API) "T" Gate
 */
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->T(shards[simulator][q]);
}

/**
 * (External API) Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->IS(shards[simulator][q]);
}

/**
 * (External API) Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->IT(shards[simulator][q]);
}

/**
 * (External API) 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void U(
    _In_ unsigned sid, _In_ unsigned q, _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->U(shards[simulator][q], theta, phi, lambda);
}

/**
 * (External API) Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, ONE_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], -I_CMPLX, I_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, -ONE_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    const complex hGate[4] = { complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
        complex(-M_SQRT1_2, ZERO_R1) };

    simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], hGate);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, I_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(
        ctrlsArray, n, shards[simulator][q], ONE_CMPLX, complex(M_SQRT1_2, M_SQRT1_2));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, -I_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(
        ctrlsArray, n, shards[simulator][q], ONE_CMPLX, complex(M_SQRT1_2, -M_SQRT1_2));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->CU(ctrlsArray, n, shards[simulator][q], theta, phi, lambda);

    delete[] ctrlsArray;
}

/**
 * (External API) Rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    RHelper(sid, b, phi, q);
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    MCRHelper(sid, b, phi, n, c, q);
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

    if (bVec.size() == 0) {
        RHelper(sid, PauliI, -2. * phi, someQubit);
    } else if (bVec.size() == 1) {
        RHelper(sid, bVec.front(), -2. * phi, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];

        TransformPauliBasis(simulator, n, b, q);

        std::size_t mask = make_mask(qVec);
        simulator->UniformParityRZ(mask, -phi);

        RevertPauliBasis(simulator, n, b, q);
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

    if (bVec.size() == 0) {
        MCRHelper(sid, PauliI, -2. * phi, nc, cs, someQubit);
    } else if (bVec.size() == 1) {
        MCRHelper(sid, bVec.front(), -2. * phi, nc, cs, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];
        std::vector<bitLenInt> csVec(cs, cs + nc);

        TransformPauliBasis(simulator, n, b, q);

        std::size_t mask = make_mask(qVec);
        simulator->CUniformParityRZ(&(csVec[0]), csVec.size(), mask, -phi);

        RevertPauliBasis(simulator, n, b, q);
    }
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    return simulator->M(shards[simulator][q]) ? 1U : 0U;
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
MICROSOFT_QUANTUM_DECL unsigned Measure(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    std::vector<unsigned> bVec;
    std::vector<unsigned> qVec;

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, true);

    unsigned toRet = (jointProb < (ONE_R1 / 2)) ? 0U : 1U;

    RevertPauliBasis(simulator, n, b, q);

    return toRet;
}

MICROSOFT_QUANTUM_DECL void SWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Swap(qi1, qi2);
}

MICROSOFT_QUANTUM_DECL void CSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->CSwap(ctrlsArray, n, qi1, qi2);

    delete[] ctrlsArray;
}

MICROSOFT_QUANTUM_DECL void AND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->AND(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void OR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->OR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void XOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->XOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void NAND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->NAND(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void NOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->NOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void XNOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->XNOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void CLAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLAND(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLXOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLXOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLNAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLNAND(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLNOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLXNOR(ci, qi, qo);
}

/**
 * (External API) Get the probability that a qubit is in the |1> state.
 */
MICROSOFT_QUANTUM_DECL double Prob(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    return simulator->Prob(shards[simulator][q]);
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

    QInterfacePtr simulator = simulators[sid];
    simulator->TimeEvolve(h, (real1)t);
}
#endif
}
