//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbdt_node.hpp"
#include "qfactory.hpp"

#define IS_REAL_1(r) (abs(ONE_CMPLX - r) <= FP_NORM_EPSILON)
#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_CTRLED_CLIFFORD(top, bottom)                                                                                \
    ((IS_REAL_1(std::real(top)) || IS_REAL_1(std::imag(bottom))) && (IS_SAME(top, bottom) || IS_SAME(top, -bottom)))
#define IS_CLIFFORD_PHASE_INVERT(top, bottom)                                                                          \
    (IS_SAME(top, bottom) || IS_SAME(top, -bottom) || IS_SAME(top, I_CMPLX * bottom) || IS_SAME(top, -I_CMPLX * bottom))
#define IS_CLIFFORD(mtrx)                                                                                              \
    ((IS_PHASE(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[0], mtrx[3])) ||                                                 \
        (IS_INVERT(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[1], mtrx[2])) ||                                             \
        ((IS_SAME(mtrx[0U], mtrx[1U]) || IS_SAME(mtrx[0U], -mtrx[1U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) ||      \
             IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                                                                \
            (IS_SAME(mtrx[0U], mtrx[2U]) || IS_SAME(mtrx[0U], -mtrx[2U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                                                             \
            (IS_SAME(mtrx[0U], mtrx[3U]) || IS_SAME(mtrx[0U], -mtrx[3U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U]))))
#define IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

namespace Qrack {

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds, bitLenInt qubitThreshold,
    real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , devID(deviceId)
    , root(NULL)
    , deviceIDs(devIds)
    , engines(eng)
{
    Init();

    SetQubitCount(qBitCount, qBitCount);

    SetPermutation(initState);
}

QBdt::QBdt(QStabilizerPtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , devID(deviceId)
    , root(NULL)
    , deviceIDs(devIds)
    , engines(eng)
{
    Init();

    SetQubitCount(qBitCount, qBitCount);

    LockEngine(enginePtr);
}

void QBdt::Init()
{
#if ENABLE_PTHREAD
    SetConcurrency(std::thread::hardware_concurrency());
#endif

    bdtStride = (GetStride() + 1U) >> 1U;
    if (!bdtStride) {
        bdtStride = 1U;
    }

    bitLenInt engineLevel = 0U;
    if (!engines.size()) {
        engines.push_back(QINTERFACE_OPTIMAL_BASE);
    }
    QInterfaceEngine rootEngine = engines[0U];
    while ((engines.size() < engineLevel) && (rootEngine != QINTERFACE_CPU) && (rootEngine != QINTERFACE_OPENCL) &&
        (rootEngine != QINTERFACE_HYBRID)) {
        ++engineLevel;
        rootEngine = engines[engineLevel];
    }
}

QBdtQStabilizerNodePtr QBdt::MakeQStabilizerNode(complex scale, bitLenInt qbCount, bitCapInt perm)
{
    return std::make_shared<QBdtQStabilizerNode>(scale,
        std::make_shared<QStabilizer>(qbCount, perm, rand_generator, ONE_CMPLX, false, false, false, 0U,
            hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor));
}

void QBdt::par_for_qbdt(const bitCapInt& end, bitLenInt maxQubit, BdtFunc fn)
{
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    Finish();
    root->Branch(maxQubit);

    const bitCapInt Stride = bdtStride;
    unsigned underThreads = (unsigned)(pow2(qubitCount - (maxQubit + 1U)) / Stride);
    if (underThreads == 1U) {
        underThreads = 0U;
    }
    const unsigned nmCrs = (unsigned)(GetConcurrencyLevel() / (underThreads + 1U));
    unsigned threads = (unsigned)(end / Stride);
    if (threads > nmCrs) {
        threads = nmCrs;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            j |= fn(j);
        }
        root->Prune(maxQubit);
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu] = std::async(std::launch::async, [&myMutex, &idx, &end, &Stride, fn]() {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                bitCapInt j;
                for (j = 0U; j < maxJ; ++j) {
                    bitCapInt k = j + l;
                    k |= fn(k);
                    j = k - l;
                    if (j >= maxJ) {
                        std::lock_guard<std::mutex> lock(myMutex);
                        idx |= j / Stride;
                        break;
                    }
                }
            }
        });
    }

    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu].get();
    }
#else
    for (bitCapInt j = 0U; j < end; ++j) {
        j |= fn(j);
    }
#endif
    root->Prune(maxQubit);
}

void QBdt::_par_for(const bitCapInt& end, ParallelFuncBdt fn)
{
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    const bitCapInt Stride = bdtStride;
    const unsigned nmCrs = GetConcurrencyLevel();
    unsigned threads = (unsigned)(end / Stride);
    if (threads > nmCrs) {
        threads = nmCrs;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            fn(j, 0U);
        }
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu] = std::async(std::launch::async, [&myMutex, &idx, &end, &Stride, cpu, fn]() {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                for (bitCapInt j = 0U; j < maxJ; ++j) {
                    fn(j + l, cpu);
                }
            }
        });
    }

    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu].get();
    }
#else
    for (bitCapInt j = 0U; j < end; ++j) {
        fn(j, 0U);
    }
#endif
}

void QBdt::SetPermutation(bitCapInt initState, complex phaseFac)
{
    Dump();

    if (!qubitCount) {
        return;
    }

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * (real1_f)PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = ONE_CMPLX;
        }
    }

    if (!bdtQubitCount) {
        root = MakeQStabilizerNode(phaseFac, attachedQubitCount, initState);

        return;
    }

    const bitLenInt maxQubit = attachedQubitCount ? (bdtQubitCount - 1U) : bdtQubitCount;
    root = std::make_shared<QBdtNode>(phaseFac);
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt qubit = 0U; qubit < maxQubit; ++qubit) {
        const size_t bit = SelectBit(initState, qubit);
        leaf->branches[bit] = std::make_shared<QBdtNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }

    if (attachedQubitCount) {
        const size_t bit = SelectBit(initState, maxQubit);
        leaf->branches[bit] = MakeQStabilizerNode(ONE_CMPLX, attachedQubitCount, initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtQStabilizerNode>();
    }
}

QInterfacePtr QBdt::Clone()
{
    Finish();

    QBdtPtr copyPtr = std::make_shared<QBdt>(engines, 0U, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->SetQubitCount(qubitCount, attachedQubitCount);

    if (!attachedQubitCount) {
        return copyPtr;
    }

    if (!bdtQubitCount) {
        QBdtQStabilizerNodePtr eLeaf = std::dynamic_pointer_cast<QBdtQStabilizerNode>(copyPtr->root);
        if (eLeaf->qReg) {
            eLeaf->qReg = std::dynamic_pointer_cast<QStabilizer>(eLeaf->qReg->Clone());
        }
        return copyPtr;
    }

    std::map<QStabilizerPtr, QStabilizerPtr> qis;

    copyPtr->SetTraversal([&qis](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        QBdtQStabilizerNodePtr qenp = std::dynamic_pointer_cast<QBdtQStabilizerNode>(leaf);
        QStabilizerPtr qi = NODE_TO_QSTABILIZER(qenp);
        if (qis.find(qi) == qis.end()) {
            qis[qi] = std::dynamic_pointer_cast<QStabilizer>(qi->Clone());
        }
        NODE_TO_QSTABILIZER(qenp) = qis[qi];
    });
    copyPtr->root->Prune(bdtQubitCount);

    return copyPtr;
}

template <typename Fn> void QBdt::GetTraversal(Fn getLambda)
{
    Finish();

    for (bitCapInt i = 0U; i < maxQPower; ++i) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
            if (IS_NODE_0(leaf->scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (!IS_NODE_0(leaf->scale) && attachedQubitCount) {
            scale *= NODE_TO_QSTABILIZER(leaf)->GetAmplitude(i >> bdtQubitCount);
        }

        getLambda((bitCapIntOcl)i, scale);
    }
}
template <typename Fn> void QBdt::SetTraversal(Fn setLambda)
{
    Dump();

    root = std::make_shared<QBdtNode>();
    root->Branch(bdtQubitCount);

    _par_for(maxQPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr prevLeaf = root;
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
            prevLeaf = leaf;
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (attachedQubitCount) {
            leaf = MakeQStabilizerNode(ONE_CMPLX, attachedQubitCount, 0U);
            prevLeaf->branches[SelectBit(i, bdtQubitCount - 1U)] = leaf;
        }

        setLambda((bitCapIntOcl)i, leaf);
    });

    root->PopStateVector(bdtQubitCount);
    root->Prune(bdtQubitCount);
}
void QBdt::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapIntOcl i, complex scale) { state[i] = scale; });
}
void QBdt::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapIntOcl i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBdt::SetQuantumState(const complex* state)
{
    if (!bdtQubitCount) {
        NODE_TO_QSTABILIZER(root)->SetQuantumState(state);
        return;
    }

    if (attachedQubitCount) {
        const bitLenInt qbCount = bdtQubitCount;
        SetTraversal([qbCount, state](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
            NODE_TO_QSTABILIZER(leaf)->SetAmplitude(i >> qbCount, state[i]);
        });
    } else {
        SetTraversal([state](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) { leaf->scale = state[i]; });
    }
}
void QBdt::SetQuantumState(QInterfacePtr eng)
{
    eng->Finish();

    if (!bdtQubitCount) {
        NODE_TO_QSTABILIZER(root) = std::dynamic_pointer_cast<QStabilizer>(eng->Clone());
        return;
    }

    if (attachedQubitCount) {
        const bitLenInt qbCount = bdtQubitCount;
        SetTraversal([qbCount, eng](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
            NODE_TO_QSTABILIZER(leaf)->SetAmplitude(i >> qbCount, eng->GetAmplitude(i));
        });
    } else {
        SetTraversal([eng](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) { leaf->scale = eng->GetAmplitude(i); });
    }
}
void QBdt::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapIntOcl i, complex scale) { outputProbs[i] = norm(scale); });
}

void QBdt::SetStateVector()
{
    Finish();

    if (!bdtQubitCount) {
        return;
    }

    QBdtQStabilizerNodePtr nRoot = MakeQStabilizerNode(ONE_R1, qubitCount);
    GetQuantumState(NODE_TO_QSTABILIZER(nRoot));
    root = nRoot;
    SetQubitCount(qubitCount, qubitCount);
}
void QBdt::ResetStateVector(bitLenInt aqb)
{
    if (attachedQubitCount <= aqb) {
        return;
    }

    Finish();

    if (!bdtQubitCount) {
        QBdtQStabilizerNodePtr oRoot = std::dynamic_pointer_cast<QBdtQStabilizerNode>(root);
        SetQubitCount(qubitCount, aqb);
        SetQuantumState(NODE_TO_QSTABILIZER(oRoot));
    }

    const bitLenInt length = attachedQubitCount - aqb;
    const bitLenInt oBdtQubitCount = bdtQubitCount;
    QBdtPtr nQubits = std::make_shared<QBdt>(engines, length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);
    nQubits->SetQubitCount(length, 0U);
    nQubits->SetPermutation(0U);
    root->InsertAtDepth(nQubits->root, oBdtQubitCount, length);
    SetQubitCount(qubitCount + length, attachedQubitCount);
    for (bitLenInt i = 0U; i < length; ++i) {
        Swap(oBdtQubitCount + i, oBdtQubitCount + length + i);
    }
    root->RemoveSeparableAtDepth(qubitCount - length, length);
    SetQubitCount(qubitCount - length, 0U);
}

void QBdt::SetDevice(int64_t dID)
{
    if (devID == dID) {
        return;
    }

    devID = dID;

    if (!attachedQubitCount) {
        return;
    }

    if (!bdtQubitCount) {
        NODE_TO_QSTABILIZER(root)->SetDevice(dID);
        return;
    }

    SetTraversal([dID](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) { NODE_TO_QSTABILIZER(leaf)->SetDevice(dID); });
}

real1_f QBdt::SumSqrDiff(QBdtPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> projectionBuff(new complex[numCores]());

    Finish();
    toCompare->Finish();

    if (randGlobalPhase) {
        real1_f lPhaseArg = FirstNonzeroPhase();
        real1_f rPhaseArg = toCompare->FirstNonzeroPhase();
        root->scale *= std::polar(ONE_R1, (real1)(rPhaseArg - lPhaseArg));
    }

    _par_for(maxQPower, [&](const bitCapInt& i, const unsigned& cpu) {
        projectionBuff[cpu] += conj(toCompare->GetAmplitude(i)) * GetAmplitude(i);
    });

    complex projection = ZERO_CMPLX;
    for (unsigned i = 0U; i < numCores; ++i) {
        projection += projectionBuff[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(projection));
}

complex QBdt::GetAmplitude(bitCapInt perm)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QBdt::GetAmplitude argument out-of-bounds!");
    }

    Finish();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
        if (IS_NODE_0(leaf->scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NODE_0(leaf->scale) && attachedQubitCount) {
        scale *= NODE_TO_QSTABILIZER(leaf)->GetAmplitude(perm >> bdtQubitCount);
    }

    return scale;
}

bitLenInt QBdt::Compose(QBdtPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QBdt::Compose start index is out-of-bounds!");
    }

    if (!toCopy->qubitCount) {
        return start;
    }

    if (bdtQubitCount && (attachedQubitCount || toCopy->attachedQubitCount)) {
        if (start < bdtQubitCount) {
            const bitLenInt offset = bdtQubitCount - start;
            ROR(qubitCount - offset, 0U, qubitCount);
            Compose(toCopy, offset);
            ROL(qubitCount - offset, 0U, qubitCount);

            return start;
        }

        if (start > bdtQubitCount) {
            const bitLenInt offset = start - bdtQubitCount;
            ROR(offset, 0U, qubitCount);
            Compose(toCopy, qubitCount - offset);
            ROL(offset, 0U, qubitCount);

            return start;
        }
    }

    Finish();

    if (!bdtQubitCount && !toCopy->bdtQubitCount) {
        NODE_TO_QSTABILIZER(root)->Compose(NODE_TO_QSTABILIZER(toCopy->root), start);
        SetQubitCount(qubitCount + toCopy->qubitCount, qubitCount + toCopy->qubitCount);

        return start;
    }

    root->InsertAtDepth(toCopy->root->ShallowClone(), start, toCopy->qubitCount);
    SetQubitCount(qubitCount + toCopy->qubitCount, attachedQubitCount + toCopy->attachedQubitCount);

    return start;
}

QInterfacePtr QBdt::Decompose(bitLenInt start, bitLenInt length)
{
    QBdtPtr dest = std::make_shared<QBdt>(engines, bdtQubitCount, length, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    Decompose(start, dest);

    return dest;
}

void QBdt::DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QBdt::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    if (length > bdtQubitCount) {
        if (dest) {
            ExecuteAsStateVector([&](QInterfacePtr eng) {
                dest->SetStateVector();
                eng->Decompose(start, NODE_TO_QSTABILIZER(dest->root));
                SetQubitCount(qubitCount - length);
                dest->ResetStateVector();
            });
        } else {
            ExecuteAsStateVector([&](QInterfacePtr eng) {
                eng->Dispose(start, length);
                SetQubitCount(qubitCount - length);
            });
        }

        return;
    }

    if (start && bdtQubitCount && attachedQubitCount) {
        ROR(start, 0U, qubitCount);
        DecomposeDispose(0U, length, dest);
        ROL(start, 0U, qubitCount);

        return;
    }

    Finish();

    if (dest) {
        dest->root = root->RemoveSeparableAtDepth(start, length)->ShallowClone();
        dest->SetQubitCount(length);
    } else {
        root->RemoveSeparableAtDepth(start, length);
    }
    SetQubitCount(qubitCount - length, attachedQubitCount);
    root->Prune(bdtQubitCount);
}

bitLenInt QBdt::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    Finish();

    QBdtPtr nQubits = std::make_shared<QBdt>(engines, length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);
    nQubits->SetQubitCount(length, 0U);
    nQubits->SetPermutation(0U);
    nQubits->root->InsertAtDepth(root, length, qubitCount);
    root = nQubits->root;
    SetQubitCount(qubitCount + length, attachedQubitCount);
    ROR(length, 0U, start + length);

    return start;
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

    std::map<QStabilizerPtr, real1> qiProbs;

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    Finish();

    _par_for(qPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NODE_0(leaf->scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (IS_NODE_0(leaf->scale)) {
            return;
        }

        if (isKet) {
            // Phase effects don't matter, for probability expectation.
            QStabilizerPtr qi = NODE_TO_QSTABILIZER(leaf);
            if (qiProbs.find(qi) == qiProbs.end()) {
                qiProbs[qi] = sqrt(NODE_TO_QSTABILIZER(leaf)->Prob(qubit - bdtQubitCount));
            }
            oneChanceBuff[cpu] += norm(scale * qiProbs[qi]);

            return;
        }

        oneChanceBuff[cpu] += norm(scale * leaf->branches[1U]->scale);
    });

    real1 oneChance = ZERO_R1;
    for (unsigned i = 0U; i < numCores; ++i) {
        oneChance += oneChanceBuff[i];
    }

    return clampProb((real1_f)oneChance);
}

real1_f QBdt::ProbAll(bitCapInt perm)
{
    Finish();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;

    for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
        if (IS_NODE_0(leaf->scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NODE_0(leaf->scale) && attachedQubitCount) {
        scale *= NODE_TO_QSTABILIZER(leaf)->GetAmplitude(perm >> bdtQubitCount);
    }

    return clampProb((real1_f)norm(scale));
}

bool QBdt::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const real1_f oneChance = Prob(qubit);
    if (oneChance >= ONE_R1) {
        result = true;
    } else if (oneChance <= ZERO_R1) {
        result = false;
    } else if (!doForce) {
        result = (Rand() <= oneChance);
    }

    if (!doApply) {
        return result;
    }

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);
    root->scale = GetNonunitaryPhase();

    _par_for(qPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NODE_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        std::lock_guard<std::mutex> lock(leaf->mtx);

        if (IS_NODE_0(leaf->scale)) {
            return;
        }

        leaf->Branch();

        if (isKet) {
            NODE_TO_QSTABILIZER(leaf)->ForceM(qubit - bdtQubitCount, result, true, true);
            return;
        }

        QBdtNodeInterfacePtr& b0 = leaf->branches[0U];
        QBdtNodeInterfacePtr& b1 = leaf->branches[1U];

        if (result) {
            if (IS_NODE_0(b1->scale)) {
                leaf->SetZero();
            } else {
                b0->SetZero();
                b1->scale /= abs(b1->scale);
            }
        } else {
            if (IS_NODE_0(b0->scale)) {
                leaf->SetZero();
            } else {
                b0->scale /= abs(b0->scale);
                b1->SetZero();
            }
        }
    });

    root->Prune(maxQubit);

    return result;
}

bitCapInt QBdt::MAll()
{
    bitCapInt result = 0U;
    QBdtNodeInterfacePtr leaf = root;

    Finish();

    for (bitLenInt i = 0U; i < bdtQubitCount; ++i) {
        leaf->Branch();
        real1_f oneChance = clampProb((real1_f)norm(leaf->branches[1U]->scale));
        bool bitResult;
        if (oneChance >= ONE_R1) {
            bitResult = true;
        } else if (oneChance <= ZERO_R1) {
            bitResult = false;
        } else {
            bitResult = (Rand() <= oneChance);
        }

        if (bitResult) {
            leaf->branches[0U]->SetZero();
            leaf->branches[1U]->scale = ONE_CMPLX;
            leaf = leaf->branches[1U];
            result |= pow2(i);
        } else {
            leaf->branches[0U]->scale = ONE_CMPLX;
            leaf->branches[1U]->SetZero();
            leaf = leaf->branches[0U];
        }
    }

    if (bdtQubitCount < qubitCount) {
        // Theoretically, there's only 1 copy of this leaf left, so no need to branch.
        result |= NODE_TO_QSTABILIZER(leaf)->MAll() << bdtQubitCount;
    }

    return result;
}

void QBdt::ApplySingle(const complex* mtrx, bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QBdt::ApplySingle target parameter must be within allocated qubit bounds!");
    }

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]) && IS_NORM_0(mtrx[0U] - mtrx[3U]) &&
        (randGlobalPhase || IS_NORM_0(ONE_CMPLX - mtrx[0U]))) {
        return;
    }

    if (!IS_CLIFFORD(mtrx)) {
        root = root->PopSpecial();
        Swap(0U, target);
        ApplySingle(mtrx, 0U);
        Swap(0U, target);

        return;
    }

    if (!bdtQubitCount) {
        NODE_TO_QSTABILIZER(root)->Mtrx(mtrx, target);

        return;
    }

    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
    const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);

    const complex2 mtrxCol1Shuff = mtrxColShuff(mtrxCol1);
    const complex2 mtrxCol2Shuff = mtrxColShuff(mtrxCol2);
#endif

    par_for_qbdt(qPower, maxQubit,
#if ENABLE_COMPLEX_X2
        [this, maxQubit, target, mtrx, &mtrxCol1, &mtrxCol2, &mtrxCol1Shuff, &mtrxCol2Shuff, isKet](
            const bitCapInt& i) {
#else
        [this, maxQubit, target, mtrx, isKet](const bitCapInt& i) {
#endif
            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < maxQubit; ++j) {
                if (IS_NODE_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
            }

            std::lock_guard<std::mutex> lock(leaf->mtx);

            if (IS_NODE_0(leaf->scale)) {
                return (bitCapInt)0U;
            }

            if (isKet) {
                leaf->Branch();
                NODE_TO_QSTABILIZER(leaf)->Mtrx(mtrx, target - bdtQubitCount);
            } else {
#if ENABLE_COMPLEX_X2
                leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, bdtQubitCount - target);
#else
                leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
            }

            return (bitCapInt)0U;
        });
}

void QBdt::ApplyControlledSingle(
    const complex* mtrx, const std::vector<bitLenInt>& controls, bitLenInt target, bool isAnti)
{
    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QBdt::ApplyControlledSingle target parameter must be within allocated qubit bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount,
        "QBdt::ApplyControlledSingle parameter controls array values must be within allocated qubit bounds!");

    if (!bdtQubitCount) {
        if (isAnti) {
            NODE_TO_QSTABILIZER(root)->MACMtrx(controls, mtrx, target);
        } else {
            NODE_TO_QSTABILIZER(root)->MCMtrx(controls, mtrx, target);
        }
        return;
    }

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]) && IS_NORM_0(ONE_CMPLX - mtrx[0U]) &&
        IS_NORM_0(ONE_CMPLX - mtrx[3U])) {
        return;
    }

    std::vector<bitLenInt> controlVec(controls.begin(), controls.end());
    std::sort(controlVec.begin(), controlVec.end());
    const bool isSwapped = (target < controlVec.back()) && (target < bdtQubitCount);
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }

    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);
    std::vector<bitLenInt> ketControlsVec;
    bitCapInt lowControlMask = 0U;
    for (size_t c = 0U; c < controls.size(); ++c) {
        const bitLenInt control = controlVec[c];
        if (control < bdtQubitCount) {
            lowControlMask |= pow2(maxQubit - (control + 1U));
        } else {
            ketControlsVec.push_back(control - bdtQubitCount);
        }
    }
    bitCapInt lowControlPerm = isAnti ? 0U : lowControlMask;

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
    const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);

    const complex2 mtrxCol1Shuff = mtrxColShuff(mtrxCol1);
    const complex2 mtrxCol2Shuff = mtrxColShuff(mtrxCol2);
#endif

    par_for_qbdt(qPower, maxQubit,
#if ENABLE_COMPLEX_X2
        [this, lowControlMask, lowControlPerm, maxQubit, target, mtrx, &mtrxCol1, &mtrxCol2, &mtrxCol1Shuff,
            &mtrxCol2Shuff, isKet, isAnti, ketControlsVec](const bitCapInt& i) {
#else
        [this, lowControlMask, lowControlPerm, maxQubit, target, mtrx, isKet, isAnti, ketControlsVec](const bitCapInt& i) {
#endif
            if ((i & lowControlMask) != lowControlPerm) {
                return (bitCapInt)(lowControlMask - ONE_BCI);
            }

            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < maxQubit; ++j) {
                if (IS_NODE_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
            }

            std::lock_guard<std::mutex> lock(leaf->mtx);

            if (IS_NODE_0(leaf->scale)) {
                return (bitCapInt)0U;
            }

            if (isKet) {
                leaf->Branch();
                QStabilizerPtr qi = NODE_TO_QSTABILIZER(leaf);
                if (isAnti) {
                    qi->MACMtrx(ketControlsVec, mtrx, target - bdtQubitCount);
                } else {
                    qi->MCMtrx(ketControlsVec, mtrx, target - bdtQubitCount);
                }
            } else {
#if ENABLE_COMPLEX_X2
                leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, bdtQubitCount - target);
#else
                leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
            }

            return (bitCapInt)0U;
        });

    // Undo isSwapped.
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }
}

void QBdt::Mtrx(const complex* mtrx, bitLenInt target) { ApplySingle(mtrx, target); }

void QBdt::MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{
    if (!controls.size()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        ApplyControlledSingle(mtrx, controls, target, false);
    }
}

void QBdt::MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{

    if (!controls.size()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        ApplyControlledSingle(mtrx, controls, target, true);
    }
}

void QBdt::MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (!controls.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        RunNonClifford(controls, topLeft, bottomRight, target, false);
        return;
    }

    const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (!IS_NORM_0(ONE_CMPLX - topLeft)) {
        ApplyControlledSingle(mtrx, controls, target, false);
        return;
    }

    if (IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    std::vector<bitLenInt> lControls(controls);
    std::sort(lControls.begin(), lControls.end());

    if (target < lControls[controls.size() - 1U]) {
        std::swap(target, lControls[controls.size() - 1U]);
    }

    ApplyControlledSingle(mtrx, lControls, target, false);
}

void QBdt::MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target)
{
    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        RunNonClifford(controls, topRight, bottomLeft, target, true);
        return;
    }

    const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (!IS_NORM_0(ONE_CMPLX - topRight) || !IS_NORM_0(ONE_CMPLX - bottomLeft)) {
        ApplyControlledSingle(mtrx, controls, target, false);
        return;
    }

    std::vector<bitLenInt> lControls(controls);
    std::sort(lControls.begin(), lControls.end());

    if ((lControls[controls.size() - 1U] < target) || (target >= bdtQubitCount)) {
        ApplyControlledSingle(mtrx, lControls, target, false);
        return;
    }

    H(target);
    MCPhase(lControls, ONE_CMPLX, -ONE_CMPLX, target);
    H(target);
}

void QBdt::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const std::vector<bitLenInt> controls{ qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
        MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));

    const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
    if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
    if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON) {
        IISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubit1, qubit2); });
}

} // namespace Qrack
