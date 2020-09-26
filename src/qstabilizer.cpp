//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

#include <chrono>

#include "qstabilizer.hpp"

namespace Qrack {

QStabilizer::QStabilizer(const bitLenInt& n, const bitCapInt& perm, const bool& useHardwareRNG, qrack_rand_gen_ptr rgp)
    : qubitCount(n)
    , x((n << 1U) + 1U, std::vector<bool>(n))
    , z((n << 1U) + 1U, std::vector<bool>(n))
    , r((n << 1U) + 1U)
    , rand_distribution(0.0, 1.0)
    , hardware_rand_generator(NULL)
{
#if !ENABLE_RDRAND
    useHardwareRNG = false;
#endif

    if (useHardwareRNG) {
        hardware_rand_generator = std::make_shared<RdRandom>();
#if !ENABLE_RNDFILE
        if (!(hardware_rand_generator->SupportsRDRAND())) {
            hardware_rand_generator = NULL;
        }
#endif
    }

    if ((rgp == NULL) && (hardware_rand_generator == NULL)) {
        rand_generator = std::make_shared<qrack_rand_gen>();
        randomSeed = std::time(0);
        SetRandomSeed(randomSeed);
    } else {
        rand_generator = rgp;
    }

    SetPermutation(perm);
}

void QStabilizer::SetPermutation(const bitCapInt& perm)
{
    bitLenInt j;

    bitLenInt rowCount = (qubitCount << 1U) + 1U;

    std::fill(r.begin(), r.end(), 0);

    for (bitLenInt i = 0; i < rowCount; i++) {
        std::fill(x[i].begin(), x[i].end(), 0);
        std::fill(z[i].begin(), z[i].end(), 0);

        if (i < qubitCount) {
            x[i][i] = true;
        } else if (i < (qubitCount << 1U)) {
            j = i - qubitCount;
            z[i][j] = true;
        }
    }

    if (!perm) {
        return;
    }

    for (j = 0; j < qubitCount; j++) {
        if (perm & pow2(j)) {
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

    for (bitLenInt j = 0; j < qubitCount; j++) {
        x[i][j] = x[k][j];
        z[i][j] = z[k][j];
    }
    r[i] = r[k];
}

/// Swaps row i and row k
void QStabilizer::rowswap(const bitLenInt& i, const bitLenInt& k)
{
    if (i == k) {
        return;
    }

    rowcopy(qubitCount << 1U, k);
    rowcopy(k, i);
    rowcopy(i, qubitCount << 1U);
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

    e = (e + r[i] + r[k]) & 0x3;

    return (uint8_t)((e < 0) ? (e + 4U) : e);
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
    bitLenInt n = qubitCount;
    bitLenInt maxLcv = n << 1U;
    bitLenInt i = n;
    bitLenInt j;
    bitLenInt k, k2;

    for (j = 0; j < n; j++) {

        // Find a generator containing X in jth column
        for (k = i; k < maxLcv; k++) {
            if (x[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (k2 = i + 1U; k2 < maxLcv; k2++) {
                if (x[k2][j]) {
                    // Gaussian elimination step:
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            i++;
        }
    }

    bitLenInt g = i - n;

    for (j = 0; j < n; j++) {

        // Find a generator containing Z in jth column
        for (k = i; k < maxLcv; k++) {
            if (z[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (k2 = i + 1U; k2 < maxLcv; k2++) {
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
    bitLenInt elemCount = qubitCount << 1U;
    int j;
    int f;
    int min = 0;

    // Wipe the scratch space clean
    r[elemCount] = 0;
    std::fill(x[elemCount].begin(), x[elemCount].end(), 0);
    std::fill(z[elemCount].begin(), z[elemCount].end(), 0);

    for (int i = elemCount - 1; i >= qubitCount + g; i--) {
        f = r[i];
        for (j = qubitCount - 1; j >= 0; j--) {
            if (z[i][j]) {
                min = j;
                if (x[elemCount][j]) {
                    f = (f + 2) & 0x3;
                }
            }
        }

        if (f == 2) {
            j = min;
            // Make the seed consistent with the ith equation
            x[elemCount][j] = !x[elemCount][j];
        }
    }
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1& nrm, complex* stateVec)
{
    bitLenInt elemCount = qubitCount << 1U;
    bitLenInt j;
    uint8_t e = r[elemCount];

    for (j = 0; j < qubitCount; j++) {
        // Pauli operator is "Y"
        if (x[elemCount][j] && z[elemCount][j]) {
            e = (e + 1) & 0x3;
        }
    }

    complex amp = nrm;
    if (e & 1) {
        amp *= I_CMPLX;
    }
    if (e & 2) {
        amp *= -ONE_CMPLX;
    }

    bitCapInt perm = 0;
    for (j = 0; j < qubitCount; j++) {
        if (x[elemCount][j]) {
            perm |= pow2(j);
        }
    }

    stateVec[perm] = amp;
}

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(complex* stateVec)
{
    bitCapInt t;
    bitCapInt t2;
    bitLenInt i;

    // log_2 of number of nonzero basis states
    bitLenInt g = gaussian();
    bitCapInt permCount = pow2(g);
    bitCapInt permCountMin1 = permCount - ONE_BCI;
    bitLenInt elemCount = qubitCount << 1U;
    real1 nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    std::fill(stateVec, stateVec + pow2(qubitCount), ZERO_CMPLX);

    setBasisState(nrm, stateVec);
    for (t = 0; t < permCountMin1; t++) {
        t2 = t ^ (t + 1);
        for (i = 0; i < g; i++) {
            if (t2 & pow2(i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, stateVec);
    }
}

/// Apply a CNOT gate with control and target
void QStabilizer::CNOT(const bitLenInt& c, const bitLenInt& t)
{
    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        if (x[i][c]) {
            x[i][t] = !x[i][t];
        }
        if (z[i][t]) {
            z[i][c] = !z[i][c];
        }
        if (x[i][c] && z[i][t] && x[i][t] && z[i][c]) {
            r[i] = (r[i] + 2) & 0x3;
        }
        if (x[i][c] && z[i][t] && x[i][t] && z[i][c]) {
            r[i] = (r[i] + 2) & 0x3;
        }
    }
}

/// Apply a Hadamard gate to target
void QStabilizer::H(const bitLenInt& t)
{
    bool tmp;

    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        tmp = x[i][t];
        x[i][t] = x[i][t] ^ (x[i][t] ^ z[i][t]);
        z[i][t] = z[i][t] ^ (z[i][t] ^ tmp);
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3;
        }
    }
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::S(const bitLenInt& t)
{
    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3;
        }
        z[i][t] = z[i][t] ^ x[i][t];
    }
}

/**
 * Returns "true" if target qubit is a Z basis eigenstate
 */
bool QStabilizer::IsSeparableZ(const bitLenInt& t)
{
    // for brevity
    bitLenInt n = qubitCount;

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
    bool isSeparable = IsSeparableZ(t);
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
    bool isSeparable = IsSeparableZ(t);
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
bool QStabilizer::M(const bitLenInt& t, const bool& doForce, const bool& result)
{
    if (qubitCount == 1U) {
        if (!x[1][0]) {
            return (r[1] & 2U);
        } else {
            bool rand = doForce ? result : Rand();
            SetPermutation(rand ? 1 : 0);
            return rand;
        }
    }

    bitLenInt elemCount = qubitCount << 1U;

    // Is the outcome random?
    bool ran = false;

    // pivot row in stabilizer
    bitLenInt p;
    // pivot row in destabilizer
    bitLenInt m;

    // for brevity
    bitLenInt n = qubitCount;

    // loop over stabilizer generators
    for (p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        ran = x[p + n][t];
        if (ran) {
            break;
        }
    }

    // If outcome is indeterminate
    if (ran) {
        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, t + n);

        // moment of quantum randomness
        r[p + n] = (doForce ? result : Rand()) ? 2 : 0;
        // Now update the Xbar's and Zbar's that don't commute with Z_b
        for (bitLenInt i = 0; i < elemCount; i++) {
            if ((i != p) && x[i][t]) {
                rowmult(i, p);
            }
        }
        return r[p + n];
    }

    // If outcome is determinate

    // Before, we were checking if stabilizer generators commute with Z_b; now, we're checking destabilizer
    // generators
    for (m = 0; m < n; m++) {
        if (x[m][t]) {
            break;
        }
    }

    rowcopy(elemCount, m + n);
    for (bitLenInt i = m + 1U; i < n; i++) {
        if (x[i][t]) {
            rowmult(elemCount, i + n);
        }
    }

    return r[elemCount];
}

bitLenInt QStabilizer::Compose(QStabilizerPtr toCopy, const bitLenInt& start)
{
    // We simply insert the (elsewhere initialized and valid) "toCopy" stabilizers and destabilizers in corresponding
    // position, and we set the new padding to 0. This is immediately a valid state, if the two original QStablizer
    // instances are valid.

    bitLenInt i, j;

    bitLenInt rowCount = (qubitCount << 1U) + 1U;

    bitLenInt length = toCopy->qubitCount;
    bitLenInt nQubitCount = qubitCount + length;
    bitLenInt secondStart = nQubitCount + start;
    const std::vector<bool> row(length, 0);

    for (i = 0; i < rowCount; i++) {
        x[i].insert(x[i].begin() + start, row.begin(), row.end());
        z[i].insert(z[i].begin() + start, row.begin(), row.end());
    }

    i = qubitCount;

    std::vector<std::vector<bool>> xGroup(length, std::vector<bool>(nQubitCount, 0));
    std::vector<std::vector<bool>> zGroup(length, std::vector<bool>(nQubitCount, 0));
    for (i = 0; i < length; i++) {
        std::copy(toCopy->x[i].begin(), toCopy->x[i].end(), xGroup[i].begin() + start);
        std::copy(toCopy->z[i].begin(), toCopy->z[i].end(), zGroup[i].begin() + start);
    }
    x.insert(x.begin() + start, xGroup.begin(), xGroup.end());
    z.insert(z.begin() + start, zGroup.begin(), zGroup.end());
    r.insert(r.begin() + start, toCopy->r.begin(), toCopy->r.begin() + length);

    std::vector<std::vector<bool>> xGroup2(length, std::vector<bool>(nQubitCount, 0));
    std::vector<std::vector<bool>> zGroup2(length, std::vector<bool>(nQubitCount, 0));
    for (i = 0; i < length; i++) {
        j = length + i;
        std::copy(toCopy->x[j].begin(), toCopy->x[j].end(), xGroup2[i].begin() + start);
        std::copy(toCopy->z[j].begin(), toCopy->z[j].end(), zGroup2[i].begin() + start);
    }
    x.insert(x.begin() + secondStart, xGroup2.begin(), xGroup2.end());
    z.insert(z.begin() + secondStart, zGroup2.begin(), zGroup2.end());
    r.insert(r.begin() + secondStart, toCopy->r.begin() + length, toCopy->r.begin() + (length << 1U));

    qubitCount = nQubitCount;

    return start;
}

void QStabilizer::DecomposeDispose(const bitLenInt& start, const bitLenInt& length, QStabilizerPtr dest)
{
    if (length == 0) {
        return;
    }

    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "dest" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    bitLenInt i, j;

    bitLenInt end = start + length;
    bitLenInt nQubitCount = qubitCount - length;
    bitLenInt secondStart = nQubitCount + start;
    bitLenInt secondEnd = nQubitCount + end;

    if (dest) {
        for (i = 0; i < length; i++) {
            j = start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[i].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[i].begin());

            j = qubitCount + start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[i + length].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[i + length].begin());
        }
        j = start;
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

    bitLenInt rowCount = (qubitCount << 1U) + 1U;

    for (i = 0; i < rowCount; i++) {
        x[i].erase(x[i].begin() + start, x[i].begin() + end);
        z[i].erase(z[i].begin() + start, z[i].begin() + end);
    }
}

bool QStabilizer::ApproxCompare(QStabilizerPtr o)
{
    if (qubitCount != o->qubitCount) {
        return false;
    }

    if (r != o->r) {
        return false;
    }

    bitLenInt rowCount = (qubitCount << 1U) + 1U;

    for (bitLenInt i = 0; i < rowCount; i++) {
        if (x[i] != o->x[i]) {
            return false;
        }
        if (z[i] != o->z[i]) {
            return false;
        }
    }

    return true;
}
} // namespace Qrack
