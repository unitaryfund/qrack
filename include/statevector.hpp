//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This header defines buffers for Qrack::QFusion.
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/parallel_for.hpp"

#include <algorithm>
#include <mutex>

#ifdef ENABLE_PTHREAD
#include <future>
#endif

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

namespace Qrack {

class StateVectorArray;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
class StateVector : public ParallelFor {
protected:
    bitCapIntOcl capacity;

public:
    StateVector(bitCapIntOcl cap)
        : capacity(cap)
    {
    }
    virtual ~StateVector()
    {
        // Intentionally left blank.
    }

    virtual complex read(const bitCapIntOcl& i) = 0;
#if ENABLE_COMPLEX_X2
    virtual complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2) = 0;
#endif
    virtual void write(const bitCapIntOcl& i, const complex& c) = 0;
    /// Optimized "write" that is only guaranteed to write if either amplitude is nonzero. (Useful for the result of 2x2
    /// tensor slicing.)
    virtual void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2) = 0;
    virtual void clear() = 0;
    virtual void copy_in(const complex* inArray) = 0;
    virtual void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy_in(StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset,
        const bitCapIntOcl length) = 0;
    virtual void copy_out(complex* outArray) = 0;
    virtual void copy_out(complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy(StateVectorPtr toCopy) = 0;
    virtual void shuffle(StateVectorPtr svp) = 0;
    virtual void get_probs(real1* outArray) = 0;
};

class StateVectorArray : public StateVector {
public:
    std::unique_ptr<complex[], void (*)(complex*)> amplitudes;

protected:
#if defined(__APPLE__)
    complex* _aligned_state_vec_alloc(bitCapIntOcl allocSize)
    {
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
        return (complex*)toRet;
    }
#endif

    std::unique_ptr<complex[], void (*)(complex*)> Alloc(bitCapIntOcl elemCount)
    {
#if defined(__ANDROID__)
        return std::unique_ptr<complex[], void (*)(complex*)>(new complex[elemCount], [](complex* c) { delete c; });
#else
        // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
        size_t allocSize = sizeof(complex) * elemCount;
        if (allocSize < QRACK_ALIGN_SIZE) {
            allocSize = QRACK_ALIGN_SIZE;
        }
#if defined(__APPLE__)
        return std::unique_ptr<complex[], void (*)(complex*)>(
            _aligned_state_vec_alloc(allocSize), [](complex* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
        return std::unique_ptr<complex[], void (*)(complex*)>(
            (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE), [](complex* c) { _aligned_free(c); });
#else
        return std::unique_ptr<complex[], void (*)(complex*)>(
            (complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize), [](complex* c) { free(c); });
#endif
#endif
    }

    virtual void Free() { amplitudes = NULL; }

public:
    StateVectorArray(bitCapIntOcl cap)
        : StateVector(cap)
        , amplitudes(Alloc(capacity))
    {
        // Intentionally left blank.
    }

    virtual ~StateVectorArray() { Free(); }

    complex read(const bitCapIntOcl& i) { return amplitudes.get()[i]; };

#if ENABLE_COMPLEX_X2
    complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2)
    {
        return complex2(amplitudes.get()[i1], amplitudes.get()[i2]);
    }
#endif

    void write(const bitCapIntOcl& i, const complex& c) { amplitudes.get()[i] = c; };

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        amplitudes.get()[i1] = c1;
        amplitudes.get()[i2] = c2;
    };

    void clear()
    {
        par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = ZERO_CMPLX; });
    }

    void copy_in(const complex* copyIn)
    {
        if (copyIn) {
            par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = copyIn[lcv]; });
        } else {
            par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = ZERO_CMPLX; });
        }
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (copyIn) {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + offset] = copyIn[lcv]; });
        } else {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + offset] = ZERO_CMPLX; });
        }
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        if (copyInSv) {
            const complex* copyIn = std::dynamic_pointer_cast<StateVectorArray>(copyInSv)->amplitudes.get() + srcOffset;
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + dstOffset] = copyIn[lcv]; });
        } else {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + dstOffset] = ZERO_CMPLX; });
        }
    }

    void copy_out(complex* copyOut)
    {
        par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { copyOut[lcv] = amplitudes[lcv]; });
    }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        par_for(
            0, length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { copyOut[lcv] = amplitudes[lcv + offset]; });
    }

    void copy(StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorArray>(toCopy)); }

    void copy(StateVectorArrayPtr toCopy)
    {
        par_for(0, capacity,
            [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = toCopy->amplitudes[lcv]; });
    }

    void shuffle(StateVectorPtr svp) { shuffle(std::dynamic_pointer_cast<StateVectorArray>(svp)); }

    void shuffle(StateVectorArrayPtr svp)
    {
        const bitCapIntOcl offset = capacity >> 1U;
        par_for(0, offset, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const complex tmp = amplitudes[lcv + offset];
            amplitudes[lcv + offset] = svp->amplitudes[lcv];
            svp->amplitudes[lcv] = tmp;
        });
    }

    void get_probs(real1* outArray)
    {
        par_for(
            0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { outArray[lcv] = norm(amplitudes[lcv]); });
    }
};
} // namespace Qrack
