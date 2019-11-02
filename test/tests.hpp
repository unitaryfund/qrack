//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "qfactory.hpp"

/* A quick-and-dirty epsilon for clamping floating point values. */
#define QRACK_TEST_EPSILON 0.5

/*
 * Default engine type to run the tests with. Global because catch doesn't
 * support parameterization.
 */
extern enum Qrack::QInterfaceEngine testEngineType;
extern enum Qrack::QInterfaceEngine testSubEngineType;
extern enum Qrack::QInterfaceEngine testSubSubEngineType;
extern qrack_rand_gen_ptr rng;
extern bool enable_normalization;
extern bool disable_hardware_rng;
extern bool async_time;
extern int device_id;
extern bitLenInt max_qubits;
extern bool single_qubit_run;
extern std::string mOutputFileName;
extern std::ofstream mOutputFile;

/* Declare the stream-to-probability prior to including catch.hpp. */
namespace Qrack {

inline std::ostream& outputPerBitProbs(std::ostream& os, Qrack::QInterfacePtr qftReg);
inline std::ostream& outputProbableResult(std::ostream& os, Qrack::QInterfacePtr qftReg);
inline std::ostream& outputIndependentBits(std::ostream& os, Qrack::QInterfacePtr qftReg);

inline std::ostream& operator<<(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    if (os.flags() & std::ios_base::showpoint) {
        os.unsetf(std::ios_base::showpoint);
        return outputPerBitProbs(os, qftReg);
    }
    if (os.flags() & std::ios_base::showbase) {
        os.unsetf(std::ios_base::showbase);
        return outputIndependentBits(os, qftReg);
    }
    return outputProbableResult(os, qftReg);
}

inline std::ostream& outputPerBitProbs(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    os << "[\n";

    for (int i = qftReg->GetQubitCount() - 1; i >= 0; i--) {
        os << "\t " << std::setw(2) << i << "]: " << qftReg->Prob(i) << std::endl;
    }
    return os;
}

inline std::ostream& outputProbableResult(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    bitCapInt i;

    double maxProb = 0;
    bitCapInt maxProbIdx = 0;
    double totalProb = 0;

    // Iterate through all possible values of the bit array
    for (i = 0; i < qftReg->GetMaxQPower(); i++) {
        double prob = qftReg->ProbAll(i);
        totalProb += prob;
        if (prob > maxProb) {
            maxProb = prob;
            maxProbIdx = i;
        }
        // if (prob > 0.0) {
        //    std::cout<<"(Perm "<<(int)i<<" "<<prob<<std::endl;
        //}
    }

    os << qftReg->GetQubitCount() << "/";

    // Print the resulting maximum probability bit pattern.
    for (i = qftReg->GetMaxQPower() >> 1UL; i > 0; i >>= 1UL) {
        if (i & maxProbIdx) {
            os << "1";
        } else {
            os << "0";
        }
    }

    // And print the probability, for interest.
    os << ":" << maxProb;

    return os;
}

inline std::ostream& outputIndependentBits(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    os << "" << qftReg->GetQubitCount() << "/";

    for (int j = qftReg->GetQubitCount() - 1; j >= 0; j--) {
        os << (int)(qftReg->Prob(j) > QRACK_TEST_EPSILON);
    }

    return os;
}

} // namespace Qrack

#include "catch.hpp"

/*
 * A fixture to create a unique QInterface test, of the appropriate type, for
 * each executing test case.
 */
class QInterfaceTestFixture {
protected:
    Qrack::QInterfacePtr qftReg;

public:
    QInterfaceTestFixture();
};

class ProbPattern : public Catch::MatcherBase<Qrack::QInterfacePtr> {
    bitLenInt start;
    bitLenInt length;
    bitCapInt mask;

public:
    ProbPattern(bitLenInt s, bitLenInt l, bitCapInt m)
        : start(s)
        , length(l)
        , mask(m)
    {
    }

    virtual bool match(Qrack::QInterfacePtr const& qftReg) const override
    {
        if (length == 0) {
            ((ProbPattern*)this)->length = qftReg->GetQubitCount();
        }

        if (length > sizeof(mask) * 8) {
            WARN("requested length " << length << " larger than possible bitmap " << sizeof(mask) * 8);
            return false;
        }

        for (bitCapInt j = 0; j < length; j++) {
            /* Consider anything more than a 50% probability as a '1'. */
            bool bit = (qftReg->Prob(j + start) > QRACK_TEST_EPSILON);
            if (bit != !!(mask & (1U << j))) {
                return false;
            }
        }
        return true;
    }

    virtual std::string describe() const override
    {
        std::ostringstream ss;
        ss << "matches bit pattern [" << (int)start << "," << start + length << "]: " << (int)length << "/";
        for (int j = (length - 1); j >= 0; j--) {
            ss << !!((int)(mask & (1 << j)));
        }
        return ss.str();
    }
};

inline ProbPattern HasProbability(bitLenInt s, bitLenInt l, uint64_t m) { return ProbPattern(s, l, m); }
inline ProbPattern HasProbability(uint64_t m) { return ProbPattern(0, 0, m); }
