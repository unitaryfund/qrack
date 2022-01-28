//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// Adapted from:
//
// CHP: CNOT-Hadamard-Phase
// Stabilizer Quantum Computer Simulator
// by Scott Aaronson
// Last modified June 30, 2004
//
// Thanks to Simon Anders and Andrew Cross for bugfixes
//
// https://www.scottaaronson.com/chp/
//
// Daniel Strano and the Qrack contributers appreciate Scott Aaronson's open sharing of the CHP code, and we hope that
// vm6502q/qrack is one satisfactory framework by which CHP could be adapted to enter the C++ STL. Our project
// philosophy aims to raise the floor of decentralized quantum computing technology access across all modern platforms,
// for all people, not commercialization.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qstabilizer.hpp"

#include <chrono>

#if SEED_DEVRAND
#include <sys/random.h>
#endif

namespace Qrack {

QStabilizer::QStabilizer(const bitLenInt& n, const bitCapInt& perm, bool useHardwareRNG, qrack_rand_gen_ptr rgp)
    : qubitCount(n)
    , x((n << 1U) + 1U, std::vector<bool>(n))
    , z((n << 1U) + 1U, std::vector<bool>(n))
    , r((n << 1U) + 1U)
    , rand_distribution(0, 1)
    , hardware_rand_generator(NULL)
    , rawRandBools(0)
    , rawRandBoolsRemaining(0)
{
#if !ENABLE_RDRAND && !ENABLE_RNDFILE && !ENABLE_DEVRAND
    useHardwareRNG = false;
#endif

    if (useHardwareRNG) {
        hardware_rand_generator = std::make_shared<RdRandom>();
#if !ENABLE_RNDFILE && !ENABLE_DEVRAND
        if (!(hardware_rand_generator->SupportsRDRAND())) {
            hardware_rand_generator = NULL;
        }
#endif
    }

    if ((rgp == NULL) && (hardware_rand_generator == NULL)) {
        rand_generator = std::make_shared<qrack_rand_gen>();
#if SEED_DEVRAND
        // The original author of this code block (Daniel Strano) is NOT a cryptography expert. However, here's the
        // author's justification for preferring /dev/random used to seed Mersenne twister, in this case. We state
        // firstly, our use case is probably more dependent on good statistical randomness than CSPRNG security.
        // Casually, we can list a few reasons our design:
        //
        // * (As a total guess, if clock manipulation isn't a completely obvious problem,) either of /dev/random or
        // /dev/urandom is probably both statistically and cryptographically preferable to the system clock, as a
        // one-time seed.
        //
        // * We need VERY LITTLE entropy for this seeding, even though its repeated a few times depending on the
        // simulation method stack. Tests of 30+ qubits don't run out of random numbers, this way, and there's no
        // detectable slow-down in Qrack.
        //
        // * The blocking behavior of /dev/random (specifically on startup) is GOOD for us, here. We WANT Qrack to block
        // until the entropy pool is ready on virtual machine and container images that start a Qrack-based application
        // on boot. (We're not crypotgraphers; we're quantum computer simulator developers and users.)
        //
        // * (I have a very basic appreciation for the REFUTATION to historical confusion over the quantity of "entropy"
        // in the device pools, but...) If our purpose is PHYSICAL REALISM of quantum computer simulation, rather than
        // cryptography, then we probably should have a tiny preference for higher "true" entropy. Although, even as a
        // developer in the quantum computing field, I must say that there might be no provable empirical difference
        // between "true quantum randomness" and "perfect statistical (whether pseudo-)randomness" as ontological
        // categories, now might there?

        const int max_rdrand_tries = 10;
        int i;
        for (i = 0; i < max_rdrand_tries; ++i) {
            if (sizeof(randomSeed) == getrandom(reinterpret_cast<char*>(&randomSeed), sizeof(randomSeed), GRND_RANDOM))
                break;
        }
        if (i == max_rdrand_tries) {
            throw std::runtime_error("Failed to seed RNG!");
        }
#else
        randomSeed = (uint32_t)std::time(0);
#endif
        SetRandomSeed(randomSeed);
    } else {
        rand_generator = rgp;
    }

#if ENABLE_ENV_VARS
    dispatchThreshold =
        (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
    dispatchThreshold = PSTRIDEPOW;
#endif

    SetPermutation(perm);
}

void QStabilizer::SetPermutation(const bitCapInt& perm)
{
    Dump();

    const bitLenInt rowCount = (qubitCount << 1U);

    std::fill(r.begin(), r.end(), 0);

    for (bitLenInt i = 0; i < rowCount; i++) {
        std::fill(x[i].begin(), x[i].end(), false);
        std::fill(z[i].begin(), z[i].end(), false);

        if (i < qubitCount) {
            x[i][i] = true;
        } else {
            bitLenInt j = i - qubitCount;
            z[i][j] = true;
        }
    }

    if (!perm) {
        return;
    }

    for (bitLenInt j = 0; j < qubitCount; j++) {
        if ((perm >> j) & 1) {
            X(j);
        }
    }
}

/// Sets row i equal to row k
void QStabilizer::rowcopy(const bitLenInt& i, const bitLenInt& k)
{
    if (i == k) {
        return;
    }

    x[i] = x[k];
    z[i] = z[k];
    r[i] = r[k];
}

/// Swaps row i and row k
void QStabilizer::rowswap(const bitLenInt& i, const bitLenInt& k)
{
    if (i == k) {
        return;
    }

    std::swap(x[k], x[i]);
    std::swap(z[k], z[i]);
    std::swap(r[k], r[i]);
}

/// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)
void QStabilizer::rowset(const bitLenInt& i, bitLenInt b)
{
    r[i] = 0;
    std::fill(x[i].begin(), x[i].end(), 0);
    std::fill(z[i].begin(), z[i].end(), 0);

    if (b < qubitCount) {
        z[i][b] = true;
    } else {
        b -= qubitCount;
        x[i][b] = true;
    }
}

/// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
uint8_t QStabilizer::clifford(const bitLenInt& i, const bitLenInt& k)
{
    // Power to which i is raised
    bitLenInt e = 0U;

    for (bitLenInt j = 0; j < qubitCount; j++) {
        // X
        if (x[k][j] && !z[k][j]) {
            // XY=iZ
            e += x[i][j] && z[i][j];
            // XZ=-iY
            e -= !x[i][j] && z[i][j];
        }
        // Y
        if (x[k][j] && z[k][j]) {
            // YZ=iX
            e += !x[i][j] && z[i][j];
            // YX=-iZ
            e -= x[i][j] && !z[i][j];
        }
        // Z
        if (!x[k][j] && z[k][j]) {
            // ZX=iY
            e += x[i][j] && !z[i][j];
            // ZY=-iX
            e -= x[i][j] && z[i][j];
        }
    }

    e = (e + r[i] + r[k]) & 0x3U;

    return e;
}

/// Left-multiply row i by row k
void QStabilizer::rowmult(const bitLenInt& i, const bitLenInt& k)
{
    r[i] = clifford(i, k);
    for (bitLenInt j = 0; j < qubitCount; j++) {
        x[i][j] = x[i][j] ^ x[k][j];
        z[i][j] = z[i][j] ^ z[k][j];
    }
}

/**
 * Do Gaussian elimination to put the stabilizer generators in the following form:
 * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
 * (Return value = number of such generators = log_2 of number of nonzero basis states)
 * At the bottom, generators containing Z's only in quasi-upper-triangular form.
 */
bitLenInt QStabilizer::gaussian()
{
    // For brevity:
    const bitLenInt n = qubitCount;
    const bitLenInt maxLcv = n << 1U;
    bitLenInt i = n;
    bitLenInt k;

    for (bitLenInt j = 0; j < n; j++) {

        // Find a generator containing X in jth column
        for (k = i; k < maxLcv; k++) {
            if (x[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; k2++) {
                if (x[k2][j]) {
                    // Gaussian elimination step:
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            i++;
        }
    }

    const bitLenInt g = i - n;

    for (bitLenInt j = 0; j < n; j++) {

        // Find a generator containing Z in jth column
        for (k = i; k < maxLcv; k++) {
            if (z[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; k2++) {
                if (z[k2][j]) {
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            i++;
        }
    }

    return g;
}

/**
 * Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
 * writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
 * performed on q.  g is the return value from gaussian(q).
 */
void QStabilizer::seed(const bitLenInt& g)
{
    const bitLenInt elemCount = qubitCount << 1U;
    int min = 0;

    // Wipe the scratch space clean
    r[elemCount] = 0;
    std::fill(x[elemCount].begin(), x[elemCount].end(), 0);
    std::fill(z[elemCount].begin(), z[elemCount].end(), 0);

    for (int i = elemCount - 1; i >= (int)(qubitCount + g); i--) {
        int f = r[i];
        for (int j = qubitCount - 1; j >= 0; j--) {
            if (z[i][j]) {
                min = j;
                if (x[elemCount][j]) {
                    f = (f + 2) & 0x3;
                }
            }
        }

        if (f == 2) {
            const int j = min;
            // Make the seed consistent with the ith equation
            x[elemCount][j] = !x[elemCount][j];
        }
    }
}

/// Helper for setBasisState() and setBasisProb()
AmplitudeEntry QStabilizer::getBasisAmp(const real1_f& nrm)
{
    const bitLenInt elemCount = qubitCount << 1U;
    uint8_t e = r[elemCount];

    for (bitLenInt j = 0; j < qubitCount; j++) {
        // Pauli operator is "Y"
        if (x[elemCount][j] && z[elemCount][j]) {
            e = (e + 1) & 0x3U;
        }
    }

    complex amp((real1)nrm, ZERO_R1);
    if (e & 1) {
        amp *= I_CMPLX;
    }
    if (e & 2) {
        amp *= -ONE_CMPLX;
    }

    bitCapIntOcl perm = 0;
    for (bitLenInt j = 0; j < qubitCount; j++) {
        if (x[elemCount][j]) {
            perm |= pow2Ocl(j);
        }
    }

    return AmplitudeEntry(perm, amp);
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1_f& nrm, complex* stateVec, QInterfacePtr eng)
{
    AmplitudeEntry entry = getBasisAmp(nrm);
    if (entry.amplitude == ZERO_CMPLX) {
        return;
    }

    if (stateVec) {
        stateVec[entry.permutation] = entry.amplitude;
    }

    if (eng) {
        eng->SetAmplitude(entry.permutation, entry.amplitude);
    }
}

/// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisProb(const real1_f& nrm, real1* outputProbs)
{
    AmplitudeEntry entry = getBasisAmp(nrm);
    outputProbs[entry.permutation] = norm(entry.amplitude);
}

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(complex* stateVec)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    std::fill(stateVec, stateVec + pow2Ocl(qubitCount), ZERO_CMPLX);

    setBasisState(nrm, stateVec, NULL);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, stateVec, NULL);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(QInterfacePtr eng)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    eng->SetPermutation(0);
    eng->SetAmplitude(0, ZERO_CMPLX);

    setBasisState(nrm, NULL, eng);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1U);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, NULL, eng);
    }
}

/// Get all probabilities corresponding to ket notation
void QStabilizer::GetProbs(real1* outputProbs)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    std::fill(outputProbs, outputProbs + pow2Ocl(qubitCount), ZERO_R1);

    setBasisProb(nrm, outputProbs);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisProb(nrm, outputProbs);
    }
}

/// Apply a CNOT gate with control and target
void QStabilizer::CNOT(const bitLenInt& c, const bitLenInt& t)
{
    Dispatch([this, c, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (x[i][c]) {
                x[i][t] = !x[i][t];
            }

            if (z[i][t]) {
                z[i][c] = !z[i][c];
            }

            if (x[i][c] && z[i][t] && (x[i][t] == z[i][c])) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply a Hadamard gate to target
void QStabilizer::H(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            std::swap(x[i][t], z[i][t]);
            if (x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::S(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
            z[i][t] = z[i][t] ^ x[i][t];
        }
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::IS(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            z[i][t] = z[i][t] ^ x[i][t];
            if (x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::Z(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (x[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply an X (or NOT) gate to target
void QStabilizer::X(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply a Pauli Y gate to target
void QStabilizer::Y(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (z[i][t] ^ x[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply square root of X gate
void QStabilizer::SqrtX(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            x[i][t] = x[i][t] ^ z[i][t];
            if (x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply inverse square root of X gate
void QStabilizer::ISqrtX(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
            x[i][t] = x[i][t] ^ z[i][t];
        }
    });
}

/// Apply square root of Y gate
void QStabilizer::SqrtY(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            std::swap(x[i][t], z[i][t]);
            if (!x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply inverse square root of Y gate
void QStabilizer::ISqrtY(const bitLenInt& t)
{
    Dispatch([this, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (!x[i][t] && z[i][t]) {
                r[i] = (r[i] + 2) & 0x3U;
            }
            std::swap(x[i][t], z[i][t]);
        }
    });
}

/// Apply a CZ gate with control and target
void QStabilizer::CZ(const bitLenInt& c, const bitLenInt& t)
{
    Dispatch([this, c, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            if (x[i][t]) {
                z[i][c] = !z[i][c];

                if (x[i][c] && (z[i][t] == z[i][c])) {
                    r[i] = (r[i] + 2) & 0x3U;
                }
            }

            if (x[i][c]) {
                z[i][t] = !z[i][t];
            }
        }
    });
}

/// Apply a CY gate with control and target
void QStabilizer::CY(const bitLenInt& c, const bitLenInt& t)
{
    Dispatch([this, c, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            z[i][t] = z[i][t] ^ x[i][t];

            if (x[i][c]) {
                x[i][t] = !x[i][t];
            }

            if (z[i][t]) {
                if (x[i][c] && (x[i][t] == z[i][c])) {
                    r[i] = (r[i] + 2) & 0x3U;
                }

                z[i][c] = !z[i][c];
            }

            z[i][t] = z[i][t] ^ x[i][t];
        }
    });
}

void QStabilizer::Swap(const bitLenInt& c, const bitLenInt& t)
{
    if (c == t) {
        return;
    }

    Dispatch([this, c, t] {
        const bitLenInt maxLcv = qubitCount << 1U;

        for (bitLenInt i = 0; i < maxLcv; i++) {
            std::swap(x[i][c], x[i][t]);
            std::swap(z[i][c], z[i][t]);
        }
    });
}

/**
 * Returns "true" if target qubit is a Z basis eigenstate
 */
bool QStabilizer::IsSeparableZ(const bitLenInt& t)
{
    Finish();

    // for brevity
    const bitLenInt n = qubitCount;

    // loop over stabilizer generators
    for (bitLenInt p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t]) {
            return false;
        }
    }

    return true;
}

/**
 * Returns "true" if target qubit is an X basis eigenstate
 */
bool QStabilizer::IsSeparableX(const bitLenInt& t)
{
    H(t);
    const bool isSeparable = IsSeparableZ(t);
    H(t);

    return isSeparable;
}

/**
 * Returns "true" if target qubit is a Y basis eigenstate
 */
bool QStabilizer::IsSeparableY(const bitLenInt& t)
{
    H(t);
    S(t);
    const bool isSeparable = IsSeparableZ(t);
    IS(t);
    H(t);

    return isSeparable;
}

/**
 * Returns:
 * 0 if target qubit is not separable
 * 1 if target qubit is a Z basis eigenstate
 * 2 if target qubit is an X basis eigenstate
 * 3 if target qubit is a Y basis eigenstate
 */
uint8_t QStabilizer::IsSeparable(const bitLenInt& t)
{
    if (IsSeparableZ(t)) {
        return 1;
    }

    H(t);

    if (IsSeparableZ(t)) {
        H(t);
        return 2;
    }

    S(t);

    if (IsSeparableZ(t)) {
        IS(t);
        H(t);
        return 3;
    }

    return 0;
}

/**
 * Measure qubit b
 */
bool QStabilizer::M(const bitLenInt& t, bool result, const bool& doForce, const bool& doApply)
{
    if (doForce && !doApply) {
        return result;
    }

    Finish();

    const bitLenInt elemCount = qubitCount << 1U;
    // for brevity
    const bitLenInt n = qubitCount;

    // pivot row in stabilizer
    bitLenInt p;
    // pivot row in destabilizer
    bitLenInt m;

    // loop over stabilizer generators
    for (p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t]) {
            // The outcome is random
            break;
        }
    }

    // If outcome is indeterminate
    if (p < n) {
        // moment of quantum randomness
        if (!doForce) {
            result = Rand();
        }

        if (!doApply) {
            return result;
        }

        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, t + n);

        r[p + n] = result ? 2U : 0U;
        // Now update the Xbar's and Zbar's that don't commute with Z_b
        for (bitLenInt i = 0; i < elemCount; i++) {
            if ((i != p) && x[i][t]) {
                rowmult(i, p);
            }
        }

        return result;
    }

    // If outcome is determinate

    // Before, we were checking if stabilizer generators commute with Z_b; now, we're checking destabilizer
    // generators
    for (m = 0; m < n; m++) {
        if (x[m][t]) {
            break;
        }
    }

    if (m >= n) {
        return r[elemCount];
    }

    rowcopy(elemCount, m + n);
    for (bitLenInt i = m + 1U; i < n; i++) {
        if (x[i][t]) {
            rowmult(elemCount, i + n);
        }
    }

    return r[elemCount];
}

bitLenInt QStabilizer::Compose(QStabilizerPtr toCopy, const bitLenInt start)
{
    // We simply insert the (elsewhere initialized and valid) "toCopy" stabilizers and destabilizers in corresponding
    // position, and we set the new padding to 0. This is immediately a valid state, if the two original QStablizer
    // instances are valid.

    Finish();
    toCopy->Finish();

    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    const bitLenInt length = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + length;
    const bitLenInt secondStart = nQubitCount + start;
    const std::vector<bool> row(length, 0);

    for (bitLenInt i = 0; i < rowCount; i++) {
        x[i].insert(x[i].begin() + start, row.begin(), row.end());
        z[i].insert(z[i].begin() + start, row.begin(), row.end());
    }

    std::vector<std::vector<bool>> xGroup(length, std::vector<bool>(nQubitCount, 0));
    std::vector<std::vector<bool>> zGroup(length, std::vector<bool>(nQubitCount, 0));
    for (bitLenInt i = 0; i < length; i++) {
        std::copy(toCopy->x[i].begin(), toCopy->x[i].end(), xGroup[i].begin() + start);
        std::copy(toCopy->z[i].begin(), toCopy->z[i].end(), zGroup[i].begin() + start);
    }
    x.insert(x.begin() + start, xGroup.begin(), xGroup.end());
    z.insert(z.begin() + start, zGroup.begin(), zGroup.end());
    r.insert(r.begin() + start, toCopy->r.begin(), toCopy->r.begin() + length);

    std::vector<std::vector<bool>> xGroup2(length, std::vector<bool>(nQubitCount, 0));
    std::vector<std::vector<bool>> zGroup2(length, std::vector<bool>(nQubitCount, 0));
    for (bitLenInt i = 0; i < length; i++) {
        bitLenInt j = length + i;
        std::copy(toCopy->x[j].begin(), toCopy->x[j].end(), xGroup2[i].begin() + start);
        std::copy(toCopy->z[j].begin(), toCopy->z[j].end(), zGroup2[i].begin() + start);
    }
    x.insert(x.begin() + secondStart, xGroup2.begin(), xGroup2.end());
    z.insert(z.begin() + secondStart, zGroup2.begin(), zGroup2.end());
    r.insert(r.begin() + secondStart, toCopy->r.begin() + length, toCopy->r.begin() + (length << 1U));

    qubitCount = nQubitCount;

    return start;
}

bool QStabilizer::CanDecomposeDispose(const bitLenInt start, const bitLenInt length)
{
    if (qubitCount == 1U) {
        return true;
    }

    Finish();

    // We want to have the maximum number of 0 cross terms possible.
    // TODO: Determine whether this is the fundamentally ideal form adjustment.
    gaussian();

    const bitLenInt end = start + length;

    for (bitLenInt i = 0; i < start; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = start; j < end; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = end; i < qubitCount; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = start; j < end; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = start; i < end; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = 0; j < start; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
        for (bitLenInt j = end; j < qubitCount; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    return true;
}

void QStabilizer::DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr dest)
{
    if (length == 0) {
        return;
    }

    if (dest) {
        dest->Dump();
    }
    Finish();

    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "dest" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    const bitLenInt end = start + length;
    const bitLenInt nQubitCount = qubitCount - length;
    const bitLenInt secondStart = nQubitCount + start;
    const bitLenInt secondEnd = nQubitCount + end;

    if (dest) {
        for (bitLenInt i = 0; i < length; i++) {
            bitLenInt j = start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[i].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[i].begin());

            j = qubitCount + start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[(i + length)].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[(i + length)].begin());
        }
        bitLenInt j = start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin());
        j = qubitCount + start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin() + length);
    }

    x.erase(x.begin() + start, x.begin() + end);
    z.erase(z.begin() + start, z.begin() + end);
    r.erase(r.begin() + start, r.begin() + end);
    x.erase(x.begin() + secondStart, x.begin() + secondEnd);
    z.erase(z.begin() + secondStart, z.begin() + secondEnd);
    r.erase(r.begin() + secondStart, r.begin() + secondEnd);

    qubitCount = nQubitCount;

    const bitLenInt rowCount = (qubitCount << 1U) + 1U;

    for (bitLenInt i = 0; i < rowCount; i++) {
        x[i].erase(x[i].begin() + start, x[i].begin() + end);
        z[i].erase(z[i].begin() + start, z[i].begin() + end);
    }
}

bool QStabilizer::ApproxCompare(QStabilizerPtr o)
{
    if (qubitCount != o->qubitCount) {
        return false;
    }

    Finish();
    o->Finish();

    const bitLenInt rowCount = (qubitCount << 1U);

    for (bitLenInt i = 0; i < rowCount; i++) {
        if (r[i] != o->r[i]) {
            return false;
        }

        for (bitLenInt j = 0; j < qubitCount; j++) {
            if (x[i][j] != o->x[i][j]) {
                return false;
            }
            if (z[i][j] != o->z[i][j]) {
                return false;
            }
        }
    }

    return true;
}
} // namespace Qrack
