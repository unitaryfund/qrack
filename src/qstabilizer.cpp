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
    , rand_distribution(0.0, 1.0)
{
    if (useHardwareRNG) {
        hardware_rand_generator = std::make_shared<RdRandom>();
#if !ENABLE_RNDFILE
        if (!(hardware_rand_generator->SupportsRDRAND())) {
            hardware_rand_generator = NULL;
        }
#endif
    }

    if (rgp == NULL) {
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
    bitLenInt obc = overBitCap();

    x = std::vector<std::vector<PAULI>>(rowCount);
    z = std::vector<std::vector<PAULI>>(rowCount);
    r = std::vector<uint8_t>(rowCount);

    for (bitLenInt i = 0; i < rowCount; i++) {
        x[i] = std::vector<PAULI>(obc);
        z[i] = std::vector<PAULI>(obc);

        if (i < qubitCount) {
            x[i][genIndex(i)] = modPow2(i);
        } else if (i < (qubitCount << 1U)) {
            j = i - qubitCount;
            z[i][genIndex(j)] = modPow2(j);
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
    bitLenInt obc = overBitCap();
    for (bitLenInt j = 0; j < obc; j++) {
        x[i][j] = x[k][j];
        z[i][j] = z[k][j];
    }
    r[i] = r[k];
}

/// Swaps row i and row k
void QStabilizer::rowswap(const bitLenInt& i, const bitLenInt& k)
{
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
        z[i][genIndex(b)] = modPow2(b);
    } else {
        b -= qubitCount;
        x[i][genIndex(b)] = modPow2(b);
    }
}

/// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
uint8_t QStabilizer::clifford(const bitLenInt& i, const bitLenInt& k)
{
    bitLenInt obc = overBitCap();
    PAULI pw;
    // Power to which i is raised
    bitLenInt e = 0U;

    for (bitLenInt j = 0; j < obc; j++) {
        for (bitLenInt l = 0; l < 32; l++) {
            pw = pow2(l);
            // X
            if ((x[k][j] & pw) && (!(z[k][j] & pw))) {
                // XY=iZ
                e += (x[i][j] & pw) && (z[i][j] & pw);
                // XZ=-iY
                e -= (!(x[i][j] & pw)) && (z[i][j] & pw);
            }
            // Y
            if ((x[k][j] & pw) && (z[k][j] & pw)) {
                // YZ=iX
                e += ((!(x[i][j] & pw)) && (z[i][j] & pw));
                // YX=-iZ
                e -= ((x[i][j] & pw) && (!(z[i][j] & pw)));
            }
            // Z
            if ((!(x[k][j] & pw)) && (z[k][j] & pw)) {
                // ZX=iY
                e += (x[i][j] & pw) && (!(z[i][j] & pw));
                // ZY=-iX
                e -= (x[i][j] & pw) && (z[i][j] & pw);
            }
        }
    }

    e = (e + r[i] + r[k]) & 0x3;

    return (uint8_t)((e < 0) ? (e + 4U) : e);
}

/// Left-multiply row i by row k
void QStabilizer::rowmult(const bitLenInt& i, const bitLenInt& k)
{
    bitLenInt obc = overBitCap();
    r[i] = clifford(i, k);
    for (bitLenInt j = 0; j < obc; j++) {
        x[i][j] ^= x[k][j];
        z[i][j] ^= z[k][j];
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
    bitLenInt j, j5;
    bitLenInt k, k2;
    PAULI pw;

    for (j = 0; j < n; j++) {
        j5 = genIndex(j);
        pw = modPow2(j);

        // Find a generator containing X in jth column
        for (k = i; k < maxLcv; k++) {
            if (x[k][j5] & pw) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (k2 = i + 1U; k2 < maxLcv; k2++) {
                if (x[k2][j5] & pw) {
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
        j5 = genIndex(j);
        pw = modPow2(j);

        // Find a generator containing Z in jth column
        for (k = i; k < maxLcv; k++) {
            if (z[k][j5] & pw) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (k2 = i + 1U; k2 < maxLcv; k2++) {
                if (z[k2][j5] & pw) {
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
    PAULI pw;
    int j, j5;
    int f;
    int min = 0;

    // Wipe the scratch space clean
    r[elemCount] = 0;
    std::fill(x[elemCount].begin(), x[elemCount].end(), 0);
    std::fill(z[elemCount].begin(), z[elemCount].end(), 0);

    for (int i = elemCount - 1; i >= qubitCount + g; i--) {
        f = r[i];
        for (j = qubitCount - 1; j >= 0; j--) {
            j5 = genIndex(j);
            pw = modPow2(j);
            if (z[i][j5] & pw) {
                min = j;
                if (x[elemCount][j5] & pw) {
                    f = (f + 2) & 0x3;
                }
            }
        }

        if (f == 2) {
            j5 = genIndex(min);
            pw = modPow2(min);
            // Make the seed consistent with the ith equation
            x[elemCount][j5] ^= pw;
        }
    }
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1& nrm, complex* stateVec)
{
    bitLenInt elemCount = qubitCount << 1U;
    bitLenInt j;
    bitLenInt j5;
    PAULI pw;
    uint8_t e = r[elemCount];

    for (j = 0; j < qubitCount; j++) {
        j5 = genIndex(j);
        pw = modPow2(j);
        // Pauli operator is "Y"
        if ((x[elemCount][j5] & pw) && (z[elemCount][j5] & pw)) {
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
        j5 = genIndex(j);
        pw = modPow2(j);
        if (x[elemCount][j5] & pw) {
            perm |= pw;
        }
    }

    // TODO: This += probably isn't necessary.
    stateVec[perm] += amp;
}

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
void QStabilizer::CNOT(const bitLenInt& control, const bitLenInt& target)
{
    bitLenInt c5 = genIndex(control);
    bitLenInt t5 = genIndex(target);
    PAULI pwc = modPow2(control);
    PAULI pwt = modPow2(target);
    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        if (x[i][c5] & pwc) {
            x[i][t5] ^= pwt;
        }
        if (z[i][t5] & pwt) {
            z[i][c5] ^= pwc;
        }
        if ((x[i][c5] & pwc) && (z[i][t5] & pwt) && (x[i][t5] & pwt) && (z[i][c5] & pwc)) {
            r[i] = (r[i] + 2) & 0x3;
        }
        if ((x[i][c5] & pwc) && (z[i][t5] & pwt) && !(x[i][t5] & pwt) && !(z[i][c5] & pwc)) {
            r[i] = (r[i] + 2) & 0x3;
        }
    }
}

/// Apply a Hadamard gate to target
void QStabilizer::H(const bitLenInt& target)
{
    unsigned long tmp;

    bitLenInt t5 = genIndex(target);
    PAULI pw = modPow2(target);
    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        tmp = x[i][t5];
        x[i][t5] ^= (x[i][t5] ^ z[i][t5]) & pw;
        z[i][t5] ^= (z[i][t5] ^ tmp) & pw;
        if ((x[i][t5] & pw) && (z[i][t5] & pw)) {
            r[i] = (r[i] + 2) & 0x3;
        }
    }
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::S(const bitLenInt& target)
{
    bitLenInt t5 = genIndex(target);
    PAULI pw = modPow2(target);
    bitLenInt maxLcv = qubitCount << 1U;

    for (bitLenInt i = 0; i < maxLcv; i++) {
        if ((x[i][t5] & pw) && (z[i][t5] & pw)) {
            r[i] = (r[i] + 2) & 0x3;
        }
        z[i][t5] ^= x[i][t5] & pw;
    }
}

/**
 * Returns "true" if target qubit is a Z basis eigenstate
 */
bool QStabilizer::IsSeparableZ(const bitLenInt& target)
{
    bitLenInt t5 = genIndex(target);
    PAULI pw = modPow2(target);

    // for brevity
    bitLenInt n = qubitCount;

    // loop over stabilizer generators
    for (bitLenInt p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t5] & pw) {
            return false;
        }
    }

    return true;
}

/**
 * Returns "true" if target qubit is an X basis eigenstate
 */
bool QStabilizer::IsSeparableX(const bitLenInt& target)
{
    H(target);
    bool isSeparable = IsSeparableZ(target);
    H(target);

    return isSeparable;
}

/**
 * Returns "true" if target qubit is a Y basis eigenstate
 */
bool QStabilizer::IsSeparableY(const bitLenInt& target)
{
    H(target);
    S(target);
    bool isSeparable = IsSeparableZ(target);
    S(target);
    H(target);

    return isSeparable;
}

/**
 * Returns:
 * 0 if target qubit is not separable
 * 1 if target qubit is a Z basis eigenstate
 * 2 if target qubit is an X basis eigenstate
 * 3 if target qubit is a Y basis eigenstate
 */
uint8_t QStabilizer::IsSeparable(const bitLenInt& target)
{
    if (IsSeparableZ(target)) {
        return 1;
    }

    H(target);

    if (IsSeparableZ(target)) {
        H(target);
        return 2;
    }

    S(target);

    if (IsSeparableZ(target)) {
        S(target);
        H(target);
        return 3;
    }

    return 0;
}

/**
 * Measure qubit b
 */
bool QStabilizer::M(const bitLenInt& target, const bool& doForce, const bool& result)
{
    bitLenInt t5 = genIndex(target);
    PAULI pw = modPow2(target);
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
        ran = (x[p + n][t5] & pw);
        if (ran) {
            break;
        }
    }

    // If outcome is indeterminate
    if (ran) {
        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, target + n);

        // moment of quantum randomness
        r[p + n] = doForce ? result : (Rand() ? 2 : 0);
        // Now update the Xbar's and Zbar's that don't commute with Z_b
        for (bitLenInt i = 0; i < elemCount; i++) {
            if ((i != p) && (x[i][t5] & pw)) {
                rowmult(i, p);
            }
        }
        return r[p + n];
    }

    // If outcome is determinate

    // Before, we were checking if stabilizer generators commute with Z_b; now, we're checking destabilizer
    // generators
    for (m = 0; m < n; m++) {
        if (x[m][t5] & pw) {
            break;
        }
    }

    rowcopy(elemCount, m + n);
    for (bitLenInt i = m + 1U; i < n; i++) {
        if (x[i][t5] & pw) {
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

    bitLenInt length = toCopy->qubitCount;
    bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    for (bitLenInt i = 0; i < nQubitCount; i++) {
        if ((i < start) || (i >= (start + length))) {
            x[i].insert(x[i].begin() + qubitCount + start, length, 0);
            z[i].insert(z[i].begin() + qubitCount + start, length, 0);

            x[i].insert(x[i].begin() + start, length, 0);
            z[i].insert(z[i].begin() + start, length, 0);
        } else {
            std::vector<PAULI> nX = toCopy->x[i - start];
            std::vector<PAULI> nZ = toCopy->x[i - start];

            nX.insert(nX.begin() + qubitCount + length, qubitCount, 0);
            nZ.insert(nZ.begin() + qubitCount + length, qubitCount, 0);

            nX.insert(nX.begin() + length, qubitCount, 0);
            nZ.insert(nZ.begin() + length, qubitCount, 0);

            nX.insert(nX.begin() + qubitCount, start, 0);
            nZ.insert(nZ.begin() + qubitCount, start, 0);

            nX.insert(nX.begin(), start, 0);
            nZ.insert(nZ.begin(), start, 0);

            x.insert(x.begin() + start + i, nX);
            z.insert(z.begin() + start + i, nZ);
        }
    }

    r.insert(r.begin() + qubitCount + start, length);
    std::copy(toCopy->r.begin() + length, toCopy->r.begin() + (length << 1U), r.begin() + qubitCount + start);
    r.insert(r.begin() + start, length);
    std::copy(toCopy->r.begin(), toCopy->r.begin() + length, r.begin() + start);

    qubitCount = nQubitCount;

    return start;
}

void QStabilizer::DecomposeDispose(const bitLenInt& start, const bitLenInt& length, QStabilizerPtr toCopy)
{
    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "toCopy" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    bitLenInt end = start + length;
    bitLenInt endQubitCount = qubitCount - (start + length);

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if ((i < start) || (i >= end)) {
            x[i].erase(x[i].begin() + start, x[i].begin() + length);
            z[i].erase(z[i].begin() + start, z[i].begin() + length);
        } else {
            x[i].erase(x[i].begin() + start + length, x[i].begin() + endQubitCount);
            x[i].erase(x[i].begin(), x[i].begin() + start);
            z[i].erase(z[i].begin() + start + length, z[i].begin() + endQubitCount);
            z[i].erase(z[i].begin(), z[i].begin() + start);

            if (toCopy) {
                toCopy->x[i - start] = x[i];
                toCopy->z[i - start] = z[i];
                toCopy->r[i - start] = r[i];
            }
        }

        x.erase(x.begin() + start, x.begin() + length);
        z.erase(z.begin() + start, z.begin() + length);
        r.erase(r.begin(), r.begin() + length);
    }

    qubitCount -= length;
}

bool QStabilizer::ApproxCompare(QStabilizerPtr o)
{
    if (qubitCount != o->qubitCount) {
        return false;
    }

    if (r != o->r) {
        return false;
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
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
