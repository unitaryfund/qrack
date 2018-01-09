//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <stdexcept>
#include <memory>
#include <atomic>
#include <thread>
#include <future>

//#include <complex>
#include "complex16simd.hpp"

//#define Complex16 std::complex<double>
#define Complex16 Complex16Simd
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {
	template <class BidirectionalIterator>
	void reverse (BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
	{
	  while ((first < last) && (first < (last - stride))) {
		last -= stride;
		std::iter_swap (first,last);
		first += stride;
	  }
	}

	template <class BidirectionalIterator>
	void rotate (BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last,  bitCapInt stride)
	{
		reverse(first, middle, stride);
		reverse(middle, last, stride);
		reverse(first, last, stride);
	}
}

#include "par_for.hpp"

namespace Qrack {
	/// The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers		
	/** The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers, including primitive bit logic gates and (abstract) opcodes-like methods. */
	class CoherentUnit {
		public:
			///Initialize a coherent unit with qBitCount number of bits, all to |0> state.
			CoherentUnit(bitLenInt qBitCount) : rand_distribution(0.0, 1.0) {
				if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
					throw std::invalid_argument("Cannot instantiate a register with greater capacity than native types on emulating system.");

				rand_generator.seed(std::time(0));

				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
				stateVec.reset(); 
				stateVec = std::move(sv);
				std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0,0.0));
				stateVec[0] = Complex16(cos(angle), sin(angle));

				//InitOCL();
			}
			///Initialize a coherent unit with qBitCount number pf bits, to initState unsigned integer permutation state
			CoherentUnit(bitLenInt qBitCount, bitCapInt initState) : rand_distribution(0.0, 1.0) {
				rand_generator.seed(std::time(0));

				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec.reset(); 
				stateVec = std::move(sv);
				std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0,0.0));
				stateVec[initState] = Complex16(cos(angle), sin(angle));

				//InitOCL();
			}
			///PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
			CoherentUnit(const CoherentUnit& pqs) : rand_distribution(0.0, 1.0) {
				rand_generator.seed(std::time(0));

				runningNorm = pqs.runningNorm;
				qubitCount = pqs.qubitCount;
				maxQPower = pqs.maxQPower;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
				stateVec.reset(); 
				stateVec = std::move(sv);
				std::copy(&(pqs.stateVec[0]), &(pqs.stateVec[0]) + maxQPower, &(stateVec[0]));

				//InitOCL();
			}

			///Get the count of bits in this register
			int GetQubitCount() {
				return qubitCount;
			}
			///PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
			void CloneRawState(Complex16* output) {
				if (runningNorm != 1.0) NormalizeState();
				std::copy(&(stateVec[0]), &(stateVec[0]) + maxQPower, &(output[0]));
			}
			///Generate a random double from 0 to 1
			double Rand() {
				return rand_distribution(rand_generator);
			}
			///Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
			void SetPermutation(bitCapInt perm) {
				double angle = Rand() * 2.0 * M_PI;

				runningNorm = 1.0;
				std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0,0.0));
				stateVec[perm] = Complex16(cos(angle), sin(angle));
			}
			///Set arbitrary pure quantum state, in unsigned int permutation basis
			void SetQuantumState(Complex16* inputState) {
				std::copy(&(inputState[0]), &(inputState[0]) + maxQPower, &(stateVec[0]));
			}
			///Combine (a copy of) another CoherentUnit with this one, after the last bit index of this one.
			/** Combine (a copy of) another CoherentUnit with this one, after the last bit index of this one. (If the programmer doesn't want to "cheat," it is left up to them to delete the old coherent unit that was added. */
			void Cohere(CoherentUnit &toCopy) {
				if (runningNorm != 1.0) NormalizeState();
				if (toCopy.runningNorm != 1.0) toCopy.NormalizeState();

				bitCapInt i;
				bitCapInt nQubitCount = qubitCount + toCopy.qubitCount;
				bitCapInt nMaxQPower = 1<<nQubitCount;
				bitCapInt startMask = 0;
				bitCapInt endMask = 0;
				for (i = 0; i < qubitCount; i++) {
					startMask += (1<<i);
				}
				for (i = qubitCount; i < nQubitCount; i++) {
					endMask += (1<<i);
				}
				double angle = Rand() * 2.0 * M_PI;
				Complex16 phaseFac(cos(angle), sin(angle));
				std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);
				for (i = 0; i < nMaxQPower; i++) {
					nStateVec[i] = phaseFac * sqrt(norm(stateVec[(i & startMask)]) * norm(toCopy.stateVec[((i & endMask)>>qubitCount)]));
				}
				//queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				stateVec.reset();
				stateVec = std::move(nStateVec);
				qubitCount = nQubitCount;
				maxQPower = 1<<nQubitCount;
				//ReInitOCL();

				UpdateRunningNorm();
			}
			///Minimally decohere a set of contigious bits from the full coherent unit.
			/** Minimally decohere a set of contigious bits from the full coherent unit. The length of this coherent unit is reduced by the length of bits decohered, and the bits removed are output in the destination CoherentUnit pointer. The destination object must be initialized to the correct number of bits, in 0 permutation state. */
			void Decohere(bitLenInt start, bitLenInt length, CoherentUnit& destination) {
				if (runningNorm != 1.0) NormalizeState();
				
				bitLenInt end = start + length;
				bitCapInt mask = 0;
				bitCapInt startMask = 0;
				bitCapInt endMask = 0;
				bitCapInt partPower = 1<<length;
				bitCapInt remainderPower = 1<<(qubitCount - length);
				bitCapInt i;				
				for (i = start; i < end; i++) {
					mask += (1<<i);
				}
				for (i = 0; i < start; i++) {
					startMask += (1<<i);
				}
				for (i = end; i < qubitCount; i++) {
					endMask += (1<<i);
				}
				
				std::unique_ptr<double[]> partStateProb(new double[partPower]());
				std::unique_ptr<double[]> remainderStateProb(new double[remainderPower]());
				double prob;
				for (i = 0; i < maxQPower; i++) {
					prob = norm(stateVec[i]);
					partStateProb[(i & mask)>>start] += prob;
					remainderStateProb[(i & startMask) + ((i & endMask)>>length)] += prob;
				}
				//queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
				stateVec.reset();
				stateVec = std::move(sv);
				qubitCount = qubitCount - length;
				maxQPower = 1<<qubitCount;
				//ReInitOCL();

				double angle = Rand() * 2.0 * M_PI;
				Complex16 phaseFac(cos(angle), sin(angle));
				double totProb = 0.0;
				for (i = 0; i < partPower; i++) {
					totProb += partStateProb[i];
				}
				if (totProb == 0.0) {
					destination.stateVec[0] = phaseFac;
				}
				else {
					for (i = 0; i < partPower; i++) {
						destination.stateVec[i] = sqrt(partStateProb[i] / totProb) * phaseFac;
					}
				}

				angle = Rand() * 2.0 * M_PI;
				phaseFac = Complex16(cos(angle), sin(angle));
				totProb = 0.0;
				for (i = 0; i < remainderPower; i++) {
					totProb += remainderStateProb[i];
				}
				if (totProb == 0.0) {
					stateVec[0] = phaseFac;
				}
				else {
					for (i = 0; i < remainderPower; i++) {
						stateVec[i] = sqrt(remainderStateProb[i] / totProb) * phaseFac;
					}
				}

				UpdateRunningNorm();
				destination.UpdateRunningNorm();
			}

			void Dispose(bitLenInt start, bitLenInt length) {
				if (runningNorm != 1.0) NormalizeState();
				
				bitLenInt end = start + length;
				bitCapInt startMask = 0;
				bitCapInt endMask = 0;
				bitCapInt remainderPower = 1<<(qubitCount - length);
				bitCapInt i;				
				for (i = 0; i < start; i++) {
					startMask += (1<<i);
				}
				for (i = end; i < qubitCount; i++) {
					endMask += (1<<i);
				}
				
				std::unique_ptr<double[]> remainderStateProb(new double[remainderPower]());
				for (i = 0; i < maxQPower; i++) {
					remainderStateProb[(i & startMask) + ((i & endMask)>>length)] += norm(stateVec[i]);
				}
				//queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
				stateVec.reset();
				stateVec = std::move(sv);
				qubitCount = qubitCount - length;
				maxQPower = 1<<qubitCount;
				//ReInitOCL();

				double angle = Rand() * 2.0 * M_PI;
				Complex16 phaseFac(cos(angle), sin(angle));
				double totProb = 0.0;
				for (i = 0; i < remainderPower; i++) {
					totProb += remainderStateProb[i];
				}
				if (totProb == 0.0) {
					stateVec[0] = phaseFac;
				}
				else {
					for (i = 0; i < remainderPower; i++) {
						stateVec[i] = sqrt(remainderStateProb[i] / totProb) * phaseFac;
					}
				}

				UpdateRunningNorm();
			}

			//Logic Gates:
			///"AND" compare two bits in CoherentUnit, and store result in outputBit
			void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
				if (!((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
					if ((inputBit1 == outputBit) || (inputBit2 == outputBit)) {
						CoherentUnit extraBit(1, 0);
						Cohere(extraBit);
						CCNOT(inputBit1, inputBit2, qubitCount - 1);
						Swap(qubitCount - 1, outputBit);
						Dispose(qubitCount - 1, 1);
					}
					else {
						SetBit(outputBit, false);
						if (inputBit1 == inputBit2) {
							CNOT(inputBit1, outputBit);
						}
						else {
							CCNOT(inputBit1, inputBit2, outputBit);
						}
					}
				}
			}
			///"OR" compare two bits in CoherentUnit, and store result in outputBit
			void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
				if (!((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
					if ((inputBit1 == outputBit) || (inputBit2 == outputBit)) {
						CoherentUnit extraBit(1, 1);
						Cohere(extraBit);
						AntiCCNOT(inputBit1, inputBit2, qubitCount - 1);
						Swap(qubitCount - 1, outputBit);
						Dispose(qubitCount - 1, 1);
					}
					else {
						SetBit(outputBit, true);
						if (inputBit1 == inputBit2) {
							AntiCNOT(inputBit1, outputBit);
						}
						else {
							AntiCCNOT(inputBit1, inputBit2, outputBit);
						}
					}
				}
			}
			///"XOR" compare two bits in CoherentUnit, and store result in outputBit
			void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
				if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
					SetBit(outputBit, false);
				}
				else {
					if ((inputBit1 == outputBit) || (inputBit2 == outputBit)) {
						CoherentUnit extraBit(1, 0);
						Cohere(extraBit);
						CNOT(inputBit1, qubitCount - 1);
						CNOT(inputBit2, qubitCount - 1);
						Swap(qubitCount - 1, outputBit);
						Dispose(qubitCount - 1, 1);
					}
					else {
						SetBit(outputBit, false);
						CNOT(inputBit1, outputBit);
						CNOT(inputBit2, outputBit);
					}
				}
			}
			/// Doubly-controlled not
			void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
				//if ((control1 >= qubitCount) || (control2 >= qubitCount))
				//	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
				if (control1 == control2) throw std::invalid_argument("CCNOT control bits cannot be same bit.");
				if (control1 == target || control2 == target)
					throw std::invalid_argument("CCNOT control bits cannot also be target.");

				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};

				bitLenInt powerMask = 0;
				Complex16 qubit[2];
				bitCapInt qPowers[4];
				bitCapInt qPowersSorted[3];
				qPowers[1] = 1 << control1;
				qPowersSorted[0] = qPowers[1];
				qPowers[2] = 1 << control2;
				qPowersSorted[1] = qPowers[2];
				qPowers[3] = 1 << target;
				qPowersSorted[2] = qPowers[3];
				qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
				std::sort(qPowersSorted, qPowersSorted + 3);
				Apply2x2(qPowers[0], qPowers[1] + qPowers[2], pauliX, 3, qPowersSorted, false, false);
			}
			/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
			void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
				//if ((control1 >= qubitCount) || (control2 >= qubitCount))
				//	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
				if (control1 == control2) throw std::invalid_argument("CCNOT control bits cannot be same bit.");
				if (control1 == target || control2 == target)
					throw std::invalid_argument("CCNOT control bits cannot also be target.");

				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};

				bitLenInt powerMask = 0;
				Complex16 qubit[2];
				bitCapInt qPowers[4];
				bitCapInt qPowersSorted[3];
				qPowers[1] = 1 << control1;
				qPowersSorted[0] = qPowers[1];
				qPowers[2] = 1 << control2;
				qPowersSorted[1] = qPowers[2];
				qPowers[3] = 1 << target;
				qPowersSorted[2] = qPowers[3];
				qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
				std::sort(qPowersSorted, qPowersSorted + 3);
				Apply2x2(0, qPowers[3], pauliX, 3, qPowersSorted, false, false);
			}
			///Controlled not
			void CNOT(bitLenInt control, bitLenInt target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CNOT control bit cannot also be target.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliX, false);
			}
			///"Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
			void AntiCNOT(bitLenInt control, bitLenInt target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CNOT control bit cannot also be target.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				ApplyAntiControlled2x2(control, target, pauliX, false);
			}
			///Hadamard gate
			void H(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("H tried to operate on bit index greater than total bits.");
				const Complex16 had[4] = {
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0)
				};
				ApplySingleBit(qubitIndex, had, true);
			}
			///Measurement gate
			bool M(bitLenInt qubitIndex) {
				if (runningNorm != 1.0) NormalizeState();

				bool result;
				double prob = Rand();
				double angle = Rand() * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				bitCapInt qPowers[1];
				qPowers[0] = 1 << qubitIndex;
				double oneChance = Prob(qubitIndex);

				result = (prob < oneChance) && oneChance > 0.0;
				double nrmlzr = 1.0;
				bitCapInt lcv;
				if (result) {
					if (oneChance > 0.0) nrmlzr = oneChance;
					par_for_all (0, maxQPower, &(stateVec[0]), Complex16(cosine, sine) / nrmlzr, NULL, qPowers,
						[](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
							if ((lcv & qPowers[0]) == 0) {
								stateVec[lcv] = Complex16(0.0, 0.0);
							}
							else {
								stateVec[lcv] = nrm * stateVec[lcv];
							}
						}
					);
					for (lcv = 0; lcv < maxQPower; lcv++) {
						
					}
				}
				else {
					if (oneChance < 1.0) nrmlzr = sqrt(1.0 - oneChance);
					par_for_all (0, maxQPower, &(stateVec[0]), Complex16(cosine, sine) / nrmlzr, NULL, qPowers,
						[](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
							if ((lcv & qPowers[0]) == 0) {
								stateVec[lcv] = nrm * stateVec[lcv];
							}
							else {
								stateVec[lcv] = Complex16(0.0, 0.0);
							}
						}
					);
				}

				UpdateRunningNorm();

				return result;
			}
			///PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
			double Prob(bitLenInt qubitIndex) {
				if (runningNorm != 1.0) NormalizeState();

				bitCapInt qPower = 1 << qubitIndex;
				double oneChance = 0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPower) == qPower) {
						oneChance += norm(stateVec[lcv]);
					} 
				}

				return oneChance;
			}
			///PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
			double ProbAll(bitCapInt fullRegister) {
				if (runningNorm != 1.0) NormalizeState();

				return norm(stateVec[fullRegister]);
			}
			///PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
			void ProbArray(double* probArray) {
				if (runningNorm != 1.0) NormalizeState();

				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					probArray[lcv] = norm(stateVec[lcv]); 
				}
			}
			///"Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state 
			void R1(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplySingleBit(qubitIndex, mtrx, true);
			}
			///Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state
			/** Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO. */
			void R1Dyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				R1((M_PI * numerator * 2) / denominator, qubitIndex);
			}
			///x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis 
			void RX(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				ApplySingleBit(qubitIndex, pauliRX, true);
			}
			///Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis
			/** Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO. */
			void RXDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RX((-M_PI * numerator * 2) / denominator, qubitIndex);
			}
			///y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis 
			void RY(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				ApplySingleBit(qubitIndex, pauliRY, true);
			}
			///Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO. */
			void RYDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RY((-M_PI * numerator * 2) / denominator, qubitIndex);
			}
			///z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis 
			void RZ(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplySingleBit(qubitIndex, pauliRZ, true);
			}
			///Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO. */
			void RZDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RZ((-M_PI * numerator * 2) / denominator, qubitIndex);
			}
			///Set individual bit to pure |0> (false) or |1> (true) state
			void SetBit(bitLenInt qubitIndex1, bool value) {
				if (value != M(qubitIndex1)) {
					X(qubitIndex1);
				}
			}
			///Swap values of two bits in register
			void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) {
				//if ((qubitIndex1 >= qubitCount) || (qubitIndex2 >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("Swap bits cannot be the same bit.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};

				bitCapInt qPowers[3];
				bitCapInt qPowersSorted[2];
				qPowers[1] = 1 << qubitIndex1;
				qPowers[2] = 1 << qubitIndex2;
				qPowers[0] = qPowers[1] + qPowers[2];
				if (qubitIndex1 < qubitIndex2) {
					qPowersSorted[0] = qPowers[1];
					qPowersSorted[1] = qPowers[2];
				}
				else {
					qPowersSorted[0] = qPowers[2];
					qPowersSorted[1] = qPowers[1];
				}
				
				Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false, false);
			}
			///NOT gate, which is also Pauli x matrix
			void X(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				ApplySingleBit(qubitIndex, pauliX, false);
			}
			///Apply NOT gate, (which is Pauli x matrix,) to each bit in register
			void XAll() {
				bitLenInt lcv;
				for (lcv = 0; lcv < qubitCount; lcv++) {
					X(lcv);
				}
			}
			///Apply Pauli Y matrix to bit
			void Y(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				ApplySingleBit(qubitIndex, pauliY, false);
			}
			///Apply Pauli Z matrix to bit
			void Z(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				ApplySingleBit(qubitIndex, pauliZ, false);
			}
			///Controlled "phase shift gate"
			/** Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state */
			void CR1(double radians, bitLenInt control, bitLenInt target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CR1 control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(control, target, mtrx, true);
			}
			///Controlled dyadic fraction "phase shift gate"
			/** Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state */
			void CR1Dyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CR1Dyad control bit cannot also be target.");
				CR1((-M_PI * numerator * 2) / denominator, control, target);
			}
			///Controlled x axis rotation
			/** Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis */
			void CRX(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRX control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(control, target, pauliRX, true);
			}
			///Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis
			/** Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRXDyad control bit cannot also be target.");
				CRX((-M_PI * numerator * 2) / denominator, control, target);
			}
			///Controlled y axis rotation
			/** Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis */
			void CRY(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRY control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(control, target, pauliRY, true);
			}
			///Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRYDyad control bit cannot also be target.");
				CRY((-M_PI * numerator * 2) / denominator, control, target);
			}
			///Controlled z axis rotation
			/** Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis */
			void CRZ(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRZ control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(control, target, pauliRZ, true);
			}
			///Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis
			/** Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRZDyad control bit cannot also be target.");
				CRZ((-M_PI * numerator * 2) / denominator, control, target);
			}
			///Apply controlled Pauli Y matrix to bit
			void CY(bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CY control bit cannot also be target.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliY, false);
			}
			///Apply controlled Pauli Z matrix to bit
			void CZ(bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CZ control bit cannot also be target.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliZ, false);
			}

			//Single register instructions:
			///"AND" compare two bit ranges in CoherentUnit, and store result in range starting at output
			void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length) {
				if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
					for (bitLenInt i = 0; i < length; i++) {
						AND(inputStart1 + i, inputStart2 + i, outputStart + i);
					}
				}
			}
			///"OR" compare two bit ranges in CoherentUnit, and store result in range starting at output
			void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length) {
				if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
					for (bitLenInt i = 0; i < length; i++) {
						OR(inputStart1 + i, inputStart2 + i, outputStart + i);
					}
				}
			}
			///"XOR" compare two bit ranges in CoherentUnit, and store result in range starting at output
			void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length) {
				if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
					for (bitLenInt i = 0; i < length; i++) {
						XOR(inputStart1 + i, inputStart2 + i, outputStart + i);
					}
				}
			}
			///Arithmetic shift left, with last 2 bits as sign and carry
			void ASL(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (shift > 0)) {
					int i;
					bitLenInt end = start + length;
					if (shift >= length) {
						for (i = start; i < end; i++) {
							SetBit(i, false);
						}
					}
					else {						
						Swap(end - 1, end - 2);
						Reverse(start, end);
						Reverse(start, start + shift);
						Reverse(start + shift, end);
						Swap(end - 1, end - 2);

						for (i = 0; i < shift; i++) {
							SetBit(i, false);
						}
					}
				}
			}
			///Arithmetic shift right, with last 2 bits as sign and carry
			void ASR(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (shift > 0)) {
					int i;
					bitLenInt end = start + length;
					if (shift >= length) {
						for (i = start; i < end; i++) {
							SetBit(i, false);
						}
					}
					else {	
						Swap(end - 1, end - 2);
						Reverse(start + shift, end);
						Reverse(start, start + shift);
						Reverse(start, end);
						Swap(end - 1, end - 2);

						for (i = start; i < shift; i++) {
							SetBit(end - i - 1, false);
						}
					}
				}
			}
			///Logical shift left, filling the extra bits with |0>
			void LSL(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (shift > 0)) {
					int i;
					bitLenInt end = start + length;
					if (shift >= length) {
						for (i = start; i < end; i++) {
							SetBit(i, false);
						}
					}
					else {	
						ROL(shift, start, length);
						for (i = start; i < shift; i++) {
							SetBit(i, false);
						}
					}
				}
			}
			///Logical shift right, filling the extra bits with |0>
			void LSR(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (shift > 0)) {
					int i;
					bitLenInt end = start + length;
					if (shift >= length) {
						for (i = start; i < end; i++) {
							SetBit(i, false);
						}
					}
					else {	
						ROR(shift, start, length);
						for (i = start; i < shift; i++) {
							SetBit(end - i - 1, false);
						}
					}
				}
			}
			/// "Circular shift left" - shift bits left, and carry last bits.
			void ROL(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if (length > 0) {
					shift = shift % length;
					if (shift > 0) {
						bitLenInt end = start + length;
						Reverse(start, end);
						Reverse(start, start + shift);
						Reverse(start + shift, end);
					}
				}
			}
			/// "Circular shift right" - shift bits right, and carry first bits.
			void ROR(bitLenInt shift, bitLenInt start, bitLenInt length) {
				if (length > 0) {
					shift = shift % length;
					if (shift > 0) {
						bitLenInt end = start + length;
						Reverse(start + shift, end);
						Reverse(start, start + shift);
						Reverse(start, end);
					}
				}
			}
			///Increment quantum integer by classical integer
			void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
				par_for_reg(start, length, qubitCount, toAdd, &(stateVec[0]),
					       [](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower,
						     const bitCapInt lengthPower, const bitCapInt toAdd, Complex16* stateArray) {
							rotate(stateArray + k,
								  stateArray + ((lengthPower - toAdd) * startPower) + k,
								  stateArray + endPower,
								  startPower);
						}
				);
			}
			///Decrement quantum integer by classical integer
			void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
				par_for_reg(start, length, qubitCount, toSub, &(stateVec[0]),
					       [](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower,
						     const bitCapInt lengthPower, const bitCapInt toSub, Complex16* stateArray) {
							rotate(stateArray + k,
								  stateArray + (toSub * startPower) + k,
								  stateArray + endPower,
								  startPower);
						}
				);
			}
			///Add two quantum integers
			/** Add integer of "length" bits in "inClear" to integer of "length" bits in "inOut," and store result in "inOut." Integer in "inClear" is cleared. */
			void ADD(bitLenInt inOut, bitLenInt inClear, bitLenInt length) {
				bitLenInt i, j, loopCount;
				bitLenInt origQubitCount = qubitCount;
				CoherentUnit carry(length, 0);
				Cohere(carry);
				loopCount = 0;
				for (i = 0; i < (length - 2); i+=2) {
					AND(inOut, inClear, origQubitCount, length);
					XOR(inOut, inClear, inOut, length);
					ASL(1, origQubitCount, length);
					AND(inOut, origQubitCount, inClear, length);
					XOR(inOut, origQubitCount, inOut, length);
					ASL(1, inClear, length);
				}
				i+=2;
				if (i == length) {
					AND(inOut, inClear, origQubitCount, length);
					XOR(inOut, inClear, inOut, length);
					ASL(1, origQubitCount, length);
					XOR(inOut, origQubitCount, inOut, length);
				}
				else {
					XOR(inOut, inClear, inOut, length);
				}
				Dispose(origQubitCount, length);
			}
			///Subtract two quantum integers
			/** Subtract integer of "length" bits in "inClear" from integer of "length" bits in "inOut," and store result in "inOut." Integer in "inClear" is cleared. */
			void SUB(bitLenInt inOut, bitLenInt inClear, bitLenInt length) {
				DEC(1, inClear, length);
				for (bitLenInt i = 0; i < length; i++) {
					X(inClear + i);
				}
				ADD(inOut, inClear, length);
			}
			/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
			void QFT(bitLenInt start, bitLenInt length) {
				if (length > 0) {
					bitLenInt end = start + length;
					bitLenInt i, j;
					for (i = start; i < end; i++) {
						H(i);
						for (j = 1; j < (end - i); j++) {
							CR1Dyad(1, 1<<j, i + j, i); 
						}
					}
				}
			}
		private:
			double runningNorm;
			bitLenInt qubitCount;
			bitCapInt maxQPower;
			std::unique_ptr<Complex16[]> stateVec;

			std::default_random_engine rand_generator;
			std::uniform_real_distribution<double> rand_distribution;

			void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx,
					const bitLenInt bitCount, const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm) {
				Complex16 nrm = Complex16(1.0 / runningNorm, 0.0);
				par_for (0, maxQPower, &(stateVec[0]), Complex16(doApplyNorm ? (1.0 / runningNorm) : 1.0, 0.0),
					   mtrx, qPowersSorted, offset1, offset2, bitCount,
					[](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm,
					      const Complex16* mtrx, const bitCapInt offset1, const bitCapInt offset2) {
						Complex16 qubit[2];

						qubit[0] = stateVec[lcv + offset1];
						qubit[1] = stateVec[lcv + offset2];						

						Complex16 Y0 = qubit[0];
						qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
						qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

						stateVec[lcv + offset1] = qubit[0];
						stateVec[lcv + offset2] = qubit[1];
					}
				);

				if (doCalcNorm) {
					UpdateRunningNorm();
				}
				else {
					runningNorm = 1.0;
				}
			}

			void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm) {
				bitCapInt qPowers[1];
				qPowers[0] = 1<<qubitIndex;
				Apply2x2(qPowers[0], 0, mtrx, 1, qPowers, true, doCalcNorm);
			}

			void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm) {
				bitCapInt qPowers[3];
				bitCapInt qPowersSorted[2];
				qPowers[1] = 1 << control;
				qPowers[2] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2];
				if (control < target) {
					qPowersSorted[0] = qPowers[1];
					qPowersSorted[1] = qPowers[2];
				}
				else {
					qPowersSorted[0] = qPowers[2];
					qPowersSorted[1] = qPowers[1];
				}
				Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, false, doCalcNorm);
			}

			void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm) {
				bitCapInt qPowers[3];
				bitCapInt qPowersSorted[2];
				qPowers[1] = 1 << control;
				qPowers[2] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2];
				if (control < target) {
					qPowersSorted[0] = qPowers[1];
					qPowersSorted[1] = qPowers[2];
				}
				else {
					qPowersSorted[0] = qPowers[2];
					qPowersSorted[1] = qPowers[1];
				}
				Apply2x2(0, qPowers[2], mtrx, 2, qPowersSorted, false, doCalcNorm);
			}

			void Carry(bitLenInt integerStart, bitLenInt integerLength, bitLenInt carryBit) {
				if (integerLength > 0) {
					bitLenInt i;
					CoherentUnit extraBit(1, 0);
					Cohere(extraBit);
					for (i = 0; i < integerLength; i+=2) {
						AND(carryBit, integerStart + i, qubitCount - 1);
						XOR(carryBit, integerStart + i, integerStart + i);
						if ((i + 1) < integerLength) { 
							AND(qubitCount - 1, integerStart+ i + 1, carryBit);
							XOR(qubitCount - 1, integerStart + i + 1, integerStart + i + 1);
						}
					}
					if (i != integerLength) {
						Swap(carryBit, qubitCount - 1);
					}
					Dispose(qubitCount - 1, 1);
				}
			}

			void NormalizeState() {
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					stateVec[lcv] /= runningNorm;
				}
				runningNorm = 1.0;
			}

			void Reverse(bitLenInt start, bitLenInt end) {
				if (start + 1 < end) {
					end -= 1;
					bitLenInt i;
					bitLenInt iter = start + (end - start - 1) / 2;
					for (i = start; i <= iter; i++) {
						Swap(i, end - i + start);
					}
				}
			}

			void UpdateRunningNorm() {
				runningNorm = par_norm(maxQPower, &(stateVec[0]));
			}
	};
}
