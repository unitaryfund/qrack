////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// THE FOLLOWING SECTION IS ADAPTED FROM THE "GILGAMESH" PROJECT, ( https://github.com/andy-thomason/gilgamesh ), UNDER
// THE MIT LICENSE. AS PART OF Qrack, THE ADAPTED CODE IS LICENSED UNDER THE GNU GENERAL PUBLIC LICENSE V3.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Copyright (c) 2016 Andy Thomason
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the  Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,  and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <atomic>
#include <future>
#include <thread>

namespace Qrack {
template <class F>
void par_for_all(const bitCapInt begin, const bitCapInt end, Complex16* stateArray, const Complex16 nrm,
    const Complex16* mtrx, const bitCapInt* maskArray, F fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(num_cpus);
    for (int cpu = 0; cpu < num_cpus; cpu++) {
        futures[cpu] = std::async(std::launch::async, [cpu, &idx, end, stateArray, nrm, mtrx, maskArray, &fn]() {
            bitCapInt i;
            for (;;) {
                i = idx++;
                if (i >= end)
                    break;
                fn(i, cpu, stateArray, nrm, mtrx, maskArray);
            }
        });
    }

    for (int cpu = 0; cpu < num_cpus; cpu++) {
        futures[cpu].get();
    }
}

template <class F>
void par_for_copy(const bitCapInt begin, const bitCapInt end, Complex16* stateArray, const bitCapInt* bciArgs,
    Complex16* nStateArray, F fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(num_cpus);
    for (int cpu = 0; cpu < num_cpus; cpu++) {
        futures[cpu] = std::async(std::launch::async, [cpu, &idx, end, stateArray, bciArgs, nStateArray, &fn]() {
            bitCapInt i;
            for (;;) {
                i = idx++;
                if (i >= end)
                    break;
                fn(i, cpu, stateArray, bciArgs, nStateArray);
            }
        });
    }

    for (int cpu = 0; cpu < num_cpus; cpu++) {
        futures[cpu].get();
    }
}

template <class F>
void par_for(const bitCapInt begin, const bitCapInt end, Complex16* stateArray, const Complex16 nrm,
    const Complex16* mtrx, const bitCapInt* maskArray, const bitCapInt offset1, const bitCapInt offset2,
    const bitLenInt bitCount, F fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(num_cpus);
    for (int cpu = 0; cpu != num_cpus; ++cpu) {
        futures[cpu] = std::async(
            std::launch::async, [cpu, &idx, end, stateArray, nrm, mtrx, offset1, offset2, bitCount, maskArray, &fn]() {
                bitCapInt i, iLow, iHigh;
                bitLenInt p;
                for (;;) {
                    iHigh = idx++;
                    i = 0;
                    for (p = 0; p < bitCount; p++) {
                        iLow = iHigh % maskArray[p];
                        i += iLow;
                        iHigh = (iHigh - iLow) << 1;
                    }
                    i += iHigh;
                    if (i >= end)
                        break;
                    fn(i, cpu, stateArray, nrm, mtrx, offset1, offset2);
                }
            });
    }

    for (int cpu = 0; cpu != num_cpus; ++cpu) {
        futures[cpu].get();
    }
}

template <class F>
void par_for_skip(const bitCapInt begin, const bitCapInt end, const bitCapInt skipPower, const Complex16* stateArray,
    const bitCapInt* bciArgs, Complex16* nStateVec, F fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(num_cpus);
    for (int cpu = 0; cpu != num_cpus; ++cpu) {
        futures[cpu] =
            std::async(std::launch::async, [cpu, &idx, end, skipPower, stateArray, bciArgs, nStateVec, &fn]() {
                bitCapInt i, iLow, iHigh;
                for (;;) {
                    iHigh = idx++;
                    i = 0;
                    iLow = iHigh % skipPower;
                    i += iLow;
                    iHigh = (iHigh - iLow) << 1;
                    i += iHigh;
                    if (i >= end)
                        break;
                    fn(i, cpu, stateArray, bciArgs, nStateVec);
                }
            });
    }

    for (int cpu = 0; cpu != num_cpus; ++cpu) {
        futures[cpu].get();
    }
}

double par_norm(const bitCapInt maxQPower, const Complex16* stateArray);

} // namespace Qrack
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// THIS ENDS THE EXCERPTED SECTION FROM "GILGAMESH."
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
