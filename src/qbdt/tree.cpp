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

namespace Qrack {

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds, bitLenInt qubitThreshold,
    real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , maxPageQubits(0U)
    , devID(deviceId)
    , root(NULL)
    , deviceIDs(devIds)
    , engines(eng)
{
    Init();

    // "Attached" state vector qubits are broken:
    // SetQubitCount(qBitCount, (maxPageQubits < qBitCount) ? maxPageQubits : qBitCount);
    SetQubitCount(qBitCount, 0U);

    SetPermutation(initState);
}

QBdt::QBdt(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , maxPageQubits(0U)
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

#if ENABLE_OPENCL
    if (rootEngine != QINTERFACE_CPU) {
        // bitLenInt segmentGlobalQb = -1;
        // bitLenInt segmentGlobalQb = 5U;
        // #if ENABLE_ENV_VARS
        //         if (getenv("QRACK_SEGMENT_QBDT_QB")) {
        //             segmentGlobalQb = (bitLenInt)std::stoi(std::string(getenv("QRACK_SEGMENT_QBDT_QB")));
        //         }
        // #endif
        // maxPageQubits = log2(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex));
        const bitLenInt cpuQubits = (GetStride() <= ONE_BCI) ? 0U : (log2(GetStride() - ONE_BCI) + 1U);
        bitLenInt gpuQubits = log2(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;
        if (gpuQubits > cpuQubits) {
            gpuQubits = cpuQubits;
        }
        // maxPageQubits = (segmentGlobalQb < maxPageQubits) ? maxPageQubits - segmentGlobalQb : 0U;
        // if ((maxPageQubits < gpuQubits) && !getenv("QRACK_SEGMENT_QBDT_QB")) {
        maxPageQubits = gpuQubits;
        // }
    }
#endif
}

QBdtQEngineNodePtr QBdt::MakeQEngineNode(complex scale, bitLenInt qbCount, bitCapInt perm)
{
    return std::make_shared<QBdtQEngineNode>(scale,
        std::dynamic_pointer_cast<QEngine>(
            CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX, doNormalize, false, false, devID,
                hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor, deviceIDs)));
}

void QBdt::SetPermutation(bitCapInt initState, complex phaseFac)
{
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
        root = MakeQEngineNode(phaseFac, attachedQubitCount, initState);

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
        leaf->branches[bit] = MakeQEngineNode(ONE_CMPLX, attachedQubitCount, initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtQEngineNode>();
    }
}

QInterfacePtr QBdt::Clone()
{
    QBdtPtr copyPtr = std::make_shared<QBdt>(engines, 0U, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->SetQubitCount(qubitCount, attachedQubitCount);

    if (!attachedQubitCount) {
        return copyPtr;
    }

    if (!bdtQubitCount) {
        QBdtQEngineNodePtr eLeaf = std::dynamic_pointer_cast<QBdtQEngineNode>(copyPtr->root);
        if (eLeaf->qReg) {
            eLeaf->qReg = std::dynamic_pointer_cast<QEngine>(eLeaf->qReg->Clone());
        }
        return copyPtr;
    }

    std::map<QEnginePtr, QEnginePtr> qis;

    copyPtr->SetTraversal([&qis](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        QBdtQEngineNodePtr qenp = std::dynamic_pointer_cast<QBdtQEngineNode>(leaf);
        QEnginePtr qi = NODE_TO_QENGINE(qenp);
        if (qis.find(qi) == qis.end()) {
            qis[qi] = std::dynamic_pointer_cast<QEngine>(qi->Clone());
        }
        NODE_TO_QENGINE(qenp) = qis[qi];
    });
    copyPtr->root->Prune(bdtQubitCount);

    return copyPtr;
}

template <typename Fn> void QBdt::GetTraversal(Fn getLambda)
{
    for (bitCapInt i = 0U; i < maxQPower; ++i) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (!IS_NORM_0(scale) && attachedQubitCount) {
            scale *= NODE_TO_QENGINE(leaf)->GetAmplitude(i >> bdtQubitCount);
        }

        getLambda((bitCapIntOcl)i, scale);
    }
}
template <typename Fn> void QBdt::SetTraversal(Fn setLambda)
{
    root = std::make_shared<QBdtNode>();

    for (bitCapInt i = 0U; i < maxQPower; ++i) {
        QBdtNodeInterfacePtr prevLeaf = root;
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
            leaf->Branch();
            prevLeaf = leaf;
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (bdtQubitCount < qubitCount) {
            leaf = MakeQEngineNode(ONE_CMPLX, attachedQubitCount, 0U);
            prevLeaf->branches[SelectBit(i, bdtQubitCount - 1U)] = leaf;
        }

        setLambda((bitCapIntOcl)i, leaf);
    }

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
void QBdt::SetQuantumState(complex const* state)
{
    if (!bdtQubitCount) {
        NODE_TO_QENGINE(root)->SetQuantumState(state);
        return;
    }

    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, state](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            NODE_TO_QENGINE(leaf)->SetAmplitude(i >> qbCount, state[i]);
        } else {
            leaf->scale = state[i];
        }
    });
}
void QBdt::SetQuantumState(QInterfacePtr eng)
{
    if (!bdtQubitCount) {
        NODE_TO_QENGINE(root) = std::dynamic_pointer_cast<QEngine>(eng->Clone());
        return;
    }

    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, eng](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            NODE_TO_QENGINE(leaf)->SetAmplitude(i >> qbCount, eng->GetAmplitude(i));
        } else {
            leaf->scale = eng->GetAmplitude(i);
        }
    });
}
void QBdt::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapIntOcl i, complex scale) { outputProbs[i] = norm(scale); });
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
        NODE_TO_QENGINE(root)->SetDevice(dID);
        return;
    }

    SetTraversal([dID](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) { NODE_TO_QENGINE(leaf)->SetDevice(dID); });
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

    if (randGlobalPhase) {
        real1_f lPhaseArg = FirstNonzeroPhase();
        real1_f rPhaseArg = toCompare->FirstNonzeroPhase();
        root->scale *= std::polar(ONE_R1, (real1)(rPhaseArg - lPhaseArg));
    }

    complex projection = ZERO_CMPLX;
    for (bitCapInt i = 0U; i < maxQPower; ++i) {
        projection += conj(toCompare->GetAmplitude(i)) * GetAmplitude(i);
    }

    return ONE_R1_F - clampProb((real1_f)norm(projection));
}

complex QBdt::GetAmplitude(bitCapInt perm)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QBdt::GetAmplitude argument out-of-bounds!");
    }

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NORM_0(scale) && attachedQubitCount) {
        scale *= NODE_TO_QENGINE(leaf)->GetAmplitude(perm >> bdtQubitCount);
    }

    return scale;
}

bitLenInt QBdt::Compose(QBdtPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QBdt::Compose start index is out-of-bounds!");
    }

    if (maxPageQubits < (attachedQubitCount + toCopy->attachedQubitCount)) {
        const bitLenInt diff = (attachedQubitCount + toCopy->attachedQubitCount) - maxPageQubits;
        ResetStateVector((diff < qubitCount) ? (qubitCount - diff) : 0U);

        if (maxPageQubits < (attachedQubitCount + toCopy->attachedQubitCount)) {
            const bitLenInt diff = (attachedQubitCount + toCopy->attachedQubitCount) - maxPageQubits;
            if (toCopy->qubitCount < diff) {
                throw std::runtime_error("Too many attached qubits to compose in QBdt::Compose()!");
            }
            toCopy->ResetStateVector(toCopy->qubitCount - diff);
        }
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

    if (!bdtQubitCount && !toCopy->bdtQubitCount) {
        NODE_TO_QENGINE(root)->Compose(NODE_TO_QENGINE(toCopy->root), start);
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

    if (start && bdtQubitCount && attachedQubitCount) {
        ROR(start, 0U, qubitCount);
        DecomposeDispose(0U, length, dest);
        ROL(start, 0U, qubitCount);

        return;
    }

    bitLenInt attachedDiff = 0U;
    if ((start + length) > bdtQubitCount) {
        attachedDiff = (start > bdtQubitCount) ? length : (start + length - bdtQubitCount);
    }

    if (dest) {
        dest->root = root->RemoveSeparableAtDepth(start, length)->ShallowClone();
        dest->SetQubitCount(length, attachedDiff);
    } else {
        root->RemoveSeparableAtDepth(start, length);
    }
    SetQubitCount(qubitCount - length, attachedQubitCount - attachedDiff);
    root->Prune(bdtQubitCount);
}

bitLenInt QBdt::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QBdtPtr nQubits = std::make_shared<QBdt>(engines, length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    return Compose(nQubits, start);
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

    std::map<QEnginePtr, real1> qiProbs;

    real1 oneChance = ZERO_R1;
    for (bitCapInt i = 0U; i < qPower; ++i) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (IS_NORM_0(scale)) {
            continue;
        }

        if (isKet) {
            // Phase effects don't matter, for probability expectation.
            QEnginePtr qi = NODE_TO_QENGINE(leaf);
            if (qiProbs.find(qi) == qiProbs.end()) {
                qiProbs[qi] = sqrt(NODE_TO_QENGINE(leaf)->Prob(qubit - bdtQubitCount));
            }
            oneChance += norm(scale * qiProbs[qi]);

            continue;
        }

        oneChance += norm(scale * leaf->branches[1U]->scale);
    }

    return clampProb((real1_f)oneChance);
}

real1_f QBdt::ProbAll(bitCapInt perm)
{
    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0U; j < bdtQubitCount; ++j) {
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NORM_0(scale) && attachedQubitCount) {
        scale *= NODE_TO_QENGINE(leaf)->GetAmplitude(perm >> bdtQubitCount);
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

    for (bitCapInt i = 0U; i < qPower; ++i) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NORM_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (IS_NORM_0(leaf->scale)) {
            continue;
        }

        leaf->Branch();

        if (isKet) {
            NODE_TO_QENGINE(leaf)->ForceM(qubit - bdtQubitCount, result, true, true);
            continue;
        }

        QBdtNodeInterfacePtr& b0 = leaf->branches[0U];
        QBdtNodeInterfacePtr& b1 = leaf->branches[1U];

        if (result) {
            if (IS_NORM_0(b1->scale)) {
                leaf->SetZero();
            } else {
                b0->SetZero();
                b1->scale /= abs(b1->scale);
            }
        } else {
            if (IS_NORM_0(b0->scale)) {
                leaf->SetZero();
            } else {
                b0->scale /= abs(b0->scale);
                b1->SetZero();
            }
        }
    }

    root->Prune(maxQubit);

    return result;
}

bitCapInt QBdt::MAll()
{
    bitCapInt result = 0U;
    QBdtNodeInterfacePtr leaf = root;
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
        result |= NODE_TO_QENGINE(leaf)->MAll() << bdtQubitCount;
    }

    return result;
}

void QBdt::ApplySingle(complex const* mtrx, bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QBdt::ApplySingle target parameter must be within allocated qubit bounds!");
    }

    if (!bdtQubitCount) {
        NODE_TO_QENGINE(root)->Mtrx(mtrx, target);
        return;
    }

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]) && IS_NORM_0(mtrx[0U] - mtrx[3U]) &&
        (randGlobalPhase || IS_NORM_0(ONE_CMPLX - mtrx[0U]))) {
        return;
    }

    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
    const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);
#endif

    par_for_qbdt(0U, qPower, [&](const bitCapInt& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        // Iterate to qubit depth.
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        if (isKet) {
            leaf->Branch();
            NODE_TO_QENGINE(leaf)->Mtrx(mtrx, target - bdtQubitCount);
        } else {
#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, bdtQubitCount - target);
#else
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
        }

        return (bitCapInt)0U;
    });

    root->Prune(maxQubit);
}

void QBdt::ApplyControlledSingle(
    complex const* mtrx, const std::vector<bitLenInt>& controls, bitLenInt target, bool isAnti)
{
    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QBdt::ApplyControlledSingle target parameter must be within allocated qubit bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount,
        "QBdt::ApplyControlledSingle parameter controls array values must be within allocated qubit bounds!");

    if (!bdtQubitCount) {
        if (isAnti) {
            NODE_TO_QENGINE(root)->MACMtrx(controls, mtrx, target);
        } else {
            NODE_TO_QENGINE(root)->MCMtrx(controls, mtrx, target);
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
#endif

    par_for_qbdt(0U, qPower, [&](const bitCapInt& i, const int& cpu) {
        if ((i & lowControlMask) != lowControlPerm) {
            return (bitCapInt)(lowControlMask - ONE_BCI);
        }

        QBdtNodeInterfacePtr leaf = root;
        // Iterate to qubit depth.
        for (bitLenInt j = 0U; j < maxQubit; ++j) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        if (isKet) {
            leaf->Branch();
            QEnginePtr qi = NODE_TO_QENGINE(leaf);
            if (isAnti) {
                qi->MACMtrx(ketControlsVec, mtrx, target - bdtQubitCount);
            } else {
                qi->MCMtrx(ketControlsVec, mtrx, target - bdtQubitCount);
            }
        } else {
#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, bdtQubitCount - target);
#else
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
        }

        return (bitCapInt)0U;
    });

    root->Prune(maxQubit);
    // Undo isSwapped.
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }
}

void QBdt::Mtrx(complex const* mtrx, bitLenInt target) { ApplySingle(mtrx, target); }

void QBdt::MCMtrx(const std::vector<bitLenInt>& controls, complex const* mtrx, bitLenInt target)
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

void QBdt::MACMtrx(const std::vector<bitLenInt>& controls, complex const* mtrx, bitLenInt target)
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
} // namespace Qrack
