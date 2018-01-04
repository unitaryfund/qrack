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
#include <random>
#include <stdexcept>
#include <memory>
#include <atomic>
#include <thread>
#include <future>

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

//#include <complex>
#include "complex16simd.hpp"

//#define Complex16 std::complex<double>
#define Complex16 Complex16Simd
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack { class CoherentUnit; }

#include "par_for.hpp"

namespace Qrack {
	/// "Qrack::RegisterDim" is used to dimension a register in "Qrack::CoherentUnit" constructors
	/** "Qrack::RegisterDim" is used to dimension a register in "Qrack::CoherentUnit" constructors. An array is passed in with an array of register dimensions. The registers become indexed by their position in the array, and they can be accessed with a numbered enum. */
	struct RegisterDim {
		bitLenInt length;
		bitLenInt startBit;
	};

	/// "Qrack::OCLSingleton" manages the single OpenCL context
	/** "Qrack::OCLSingleton" manages the single OpenCL context. */
	class OCLSingleton{
		public:
			///Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.) 
			static OCLSingleton* Instance();
			///If this is the first time instantiating the OpenCL context, you may specify platform number and device number.
			static OCLSingleton* Instance(int plat, int dev);
			///Get a pointer to the OpenCL context
			cl::Context* GetContextPtr() {
				return &context;
			}
			///Get a pointer to the OpenCL queue
			cl::CommandQueue* GetQueuePtr() {
				return &queue;
			}
			///Get a pointer to the Apply2x2 function kernel
			cl::Kernel* GetApply2x2Ptr() {
				return &apply2x2;
			}
			///Get a pointer to the UpdateRunningNorm function kernel
			cl::Kernel* GetUpdateRunningNormPtr() {
				return &updateRunningNorm;
			}

		private:
			std::vector<cl::Platform> all_platforms;
			cl::Platform default_platform;
			std::vector<cl::Device> all_devices;
			cl::Device default_device;
			cl::Context context;
			cl::Program program;
			cl::CommandQueue queue;
			cl::Kernel apply2x2;
			cl::Kernel updateRunningNorm;

			OCLSingleton(){
				InitOCL(0, 0);
			}  // Private so that it can  not be called
			OCLSingleton(int plat, int dev){
				InitOCL(plat, dev);
			}  // Private so that it can  not be called
			OCLSingleton(OCLSingleton const&){};             // copy constructor is private
			OCLSingleton& operator=(OCLSingleton const&){};  // assignment operator is private
			static OCLSingleton* m_pInstance;

			void InitOCL(int plat, int dev) {
				// get all platforms (drivers), e.g. NVIDIA
				
				cl::Platform::get(&all_platforms);

				if (all_platforms.size()==0) {
					std::cout<<" No platforms found. Check OpenCL installation!\n";
					exit(1);
				}
				default_platform=all_platforms[plat];
				std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

				// get default device (CPUs, GPUs) of the default platform
				default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
				if(all_devices.size()==0){
					std::cout<<" No devices found. Check OpenCL installation!\n";
					exit(1);
				}

				// use device[1] because that's a GPU; device[0] is the CPU
				default_device=all_devices[dev];
				std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

				// a context is like a "runtime link" to the device and platform;
				// i.e. communication is possible
				context=cl::Context({default_device});

				// create the program that we want to execute on the device
				cl::Program::Sources sources;

				// calculates for each element; C = A + B
				std::string kernel_code=
				"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
				"   inline double2 zmul(const double2 lhs, const double2 rhs) {"
				"	return (lhs * (double2)(rhs.y, -(rhs.y))) + (rhs.x * (double2)(lhs.y, lhs.x));"
				"   }"
				""
				"   void kernel apply2x2(global double2* stateVec, constant double2* cmplxPtr,"
				"			   constant ulong* ulongPtr) {"
				""
				"	ulong ID, Nthreads, lcv;"
				""
				"       ID = get_global_id(0);"
				"       Nthreads = get_global_size(0);"
				"	constant double2* mtrx = cmplxPtr;"	
				"	double2 nrm = cmplxPtr[4];"
				"	ulong bitCount = ulongPtr[0];"
				"	ulong maxI = ulongPtr[1];"
				"	constant ulong* qPowers = (ulongPtr + 2);"
				"	constant ulong* qPowersSorted = (ulongPtr + 4);"
				"	ulong offset1 = qPowers[0];"
				"	ulong offset2 = qPowers[1];"
				""
				"	double2 Y0;"
				"	ulong i, iLow, iHigh;"
				"       double2 qubit[2];"
				"	unsigned char p;"
				"	lcv = ID;"
				"	iHigh = lcv;"
				"	i = 0;"
				"	for (p = 0; p < bitCount; p++) {"
				"		iLow = iHigh % qPowersSorted[p];"
				"		i += iLow;"
				"		iHigh = (iHigh - iLow)<<1;"				
				"	}"
				"	i += iHigh;"
				"	while (i < maxI) {"				
				"		qubit[0] = stateVec[i + offset1];"
				"		qubit[1] = stateVec[i + offset2];"			
				""
				"		Y0 = qubit[0];"
				"		qubit[0] = zmul(nrm, (zmul(mtrx[0], Y0) + zmul(mtrx[1], qubit[1])));"
				"		qubit[1] = zmul(nrm, (zmul(mtrx[2], Y0) + zmul(mtrx[3], qubit[1])));"
				""
				"		stateVec[i + offset1] = qubit[0];"
				"		stateVec[i + offset2] = qubit[1];"
				""
				"		lcv += Nthreads;"
				"		iHigh = lcv;"
				"		i = 0;"
				"		for (p = 0; p < bitCount; p++) {"
				"			iLow = iHigh % qPowersSorted[p];"
				"			i += iLow;"
				"			iHigh = (iHigh - iLow)<<1;"				
				"		}"
				"		i += iHigh;"
				"	}"
				"   }"
				""
				"   void kernel updateRunningNorm(global double2* stateVec, constant ulong* maxPtr, global double* nrm) {"
				"	ulong lcv, maxQPower, ID, Nthreads;"
				"	maxQPower = *maxPtr;"
				"	double sqrNorm = 0.0;"
				"       ID = get_global_id(0);"
				"       Nthreads = get_global_size(0);"
				"	double2 temp;"
				"	for (lcv = ID; lcv < maxQPower; lcv+=Nthreads) {"
				"		temp = stateVec[lcv];"
				"		temp *= temp;"
				"		sqrNorm += temp.x + temp.y;"
				"	}"
				"	nrm[ID] = sqrNorm;"
				"   }";
				sources.push_back({kernel_code.c_str(), kernel_code.length()});

				program = cl::Program(context, sources);
				if (program.build({default_device}) != CL_SUCCESS) {
					std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
					exit(1);
				}

				queue = cl::CommandQueue(context, default_device);
				apply2x2 = cl::Kernel(program, "apply2x2");
				updateRunningNorm = cl::Kernel(program, "updateRunningNorm");
			}
	};

	OCLSingleton* OCLSingleton::m_pInstance = NULL;
	OCLSingleton* OCLSingleton::Instance() {
		if (!m_pInstance) m_pInstance = new OCLSingleton();
		return m_pInstance;
	}
	OCLSingleton* OCLSingleton::Instance(int plat, int dev) {
		if (!m_pInstance) {
			m_pInstance = new OCLSingleton(plat, dev);
		}
		else {
			std::cout<<"Warning: Tried to reinitialize OpenCL environment with platform and device."<<std::endl;
		}
		return m_pInstance;
	}

	/// The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers		
	/** The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers, including primitive bit logic gates and (abstract) opcodes-like methods. */
	class CoherentUnit {
		public:
			///Initialize a coherent unit with qBitCount number of bits, all to |0> state.
			CoherentUnit(bitLenInt qBitCount) : rand_distribution(0.0, 1.0) {
				if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
					throw std::invalid_argument("Cannot instantiate a register with greater capacity than native types on emulating system.");

				bitCapInt lcv;
				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec = std::move(sv);
				InitOCL();
				
				stateVec[0] = Complex16(cos(angle), sin(angle));
				for (lcv = 1; lcv < maxQPower; lcv++) {
					stateVec[lcv] = Complex16(0.0, 0.0);
				}

				std::unique_ptr<RegisterDim[]> rd(new RegisterDim[1]);
				registerDims = std::move(rd);
				registerDims[0].length = qubitCount;
				registerDims[0].startBit = 0;
				registerCount = 1;
			}
			///Initialize a coherent unit with qBitCount number pf bits, to initState unsigned integer permutation state
			CoherentUnit(bitLenInt qBitCount, bitCapInt initState) : rand_distribution(0.0, 1.0) {
				bitCapInt lcv;
				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec = std::move(sv);
				InitOCL();

				for (lcv = 0; lcv < maxQPower; lcv++) {
					if (lcv == initState) {
						stateVec[lcv] = Complex16(cos(angle), sin(angle));
					}	
					else {
						stateVec[lcv] = Complex16(0.0, 0.0);
					}
				}

				std::unique_ptr<RegisterDim[]> rd(new RegisterDim[1]);
				registerDims = std::move(rd);
				registerDims[0].length = qubitCount;
				registerDims[0].startBit = 0;
				registerCount = 1;
			}
			///Initialize a coherent unit with register dimensions
			CoherentUnit(const RegisterDim* regDims, bitLenInt regCount) : rand_distribution(0.0, 1.0) {
				bitCapInt lcv;
				qubitCount = 0;
				for (lcv = 0; lcv < registerCount; lcv++) {
					qubitCount += registerDims[lcv].length;
				}

				if (qubitCount > (sizeof(bitCapInt) * bitsInByte))
					throw std::invalid_argument("Cannot instantiate a register with greater capacity than native types on emulating system.");

				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qubitCount;
				maxQPower = 1<<qubitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec = std::move(sv);
				InitOCL();

				stateVec[0] = Complex16(cos(angle), sin(angle));
				for (lcv = 1; lcv < maxQPower; lcv++) {
					stateVec[lcv] = Complex16(0.0, 0.0);
				}

				registerCount = regCount;
				std::unique_ptr<RegisterDim[]> rd(new RegisterDim[regCount]);
				registerDims = std::move(rd);
				std::copy(&(regDims[0]), &(regDims[regCount]), &(registerDims[0]));
			}

			///Initialize a coherent unit with register dimensions and initial overall permutation state
			CoherentUnit(const RegisterDim* regDims, bitLenInt regCount, bitCapInt initState) : rand_distribution(0.0, 1.0) {
				bitLenInt lcv;
				qubitCount = 0;
				for (lcv = 0; lcv < registerCount; lcv++) {
					qubitCount += registerDims[lcv].length;
				}

				if (qubitCount > (sizeof(bitCapInt) * bitsInByte))
					throw std::invalid_argument("Cannot instantiate a register with greater capacity than native types on emulating system.");
				double angle = Rand() * 2.0 * M_PI;
				runningNorm = 1.0;
				qubitCount = qubitCount;
				maxQPower = 1<<qubitCount;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec = std::move(sv);
				InitOCL();

				for (lcv = 0; lcv < maxQPower; lcv++) {
					if (lcv == initState) {
						stateVec[lcv] = Complex16(cos(angle), sin(angle));
					}	
					else {
						stateVec[lcv] = Complex16(0.0, 0.0);
					}
				}

				registerCount = regCount;
				std::unique_ptr<RegisterDim[]> rd(new RegisterDim[regCount]);
				registerDims = std::move(rd);
				std::copy(&(regDims[0]), &(regDims[regCount]), &(registerDims[0]));
			}
			///PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
			CoherentUnit(const CoherentUnit& pqs) : rand_distribution(0.0, 1.0) {
				runningNorm = pqs.runningNorm;
				qubitCount = pqs.qubitCount;
				maxQPower = pqs.maxQPower;
				std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]); 
				stateVec = std::move(sv);
				InitOCL();
				std::copy(&(pqs.stateVec[0]), &(pqs.stateVec[qubitCount]), &(stateVec[0]));
				registerCount = pqs.registerCount;
				std::unique_ptr<RegisterDim[]> rd(new RegisterDim[registerCount]);
				registerDims = std::move(rd);
				std::copy(&(pqs.registerDims[0]), &(pqs.registerDims[pqs.registerCount]), &(registerDims[0]));
			}

			///Get the count of bits in this register
			int GetQubitCount() {
				return qubitCount;
			}
			///PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
			void CloneRawState(Complex16* output) {
				if (runningNorm != 1.0) NormalizeState();
				std::copy(&(stateVec[0]), &(stateVec[maxQPower]), &(output[0]));
			}
			///Generate a random double from 0 to 1
			double Rand() {
				return rand_distribution(rand_generator);
			}
			///Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
			void SetPermutation(bitCapInt perm) {
				double angle = Rand() * 2.0 * M_PI;

				runningNorm = 1.0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if (lcv == perm) {
						stateVec[lcv] = Complex16(cos(angle), sin(angle));
					}	
					else {
						stateVec[lcv] = Complex16(0.0, 0.0);
					}
				}
			}
			///Set arbitrary pure quantum state, in unsigned int permutation basis
			void SetQuantumState(Complex16* inputState) {
				std::copy(&(inputState[0]), &(inputState[maxQPower]), &(stateVec[0]));
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
				std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);
				for (i = 0; i < nMaxQPower; i++) {
					nStateVec[i] = stateVec[(i & startMask)] * toCopy.stateVec[((i & endMask)>>qubitCount)];
				}
				enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				stateVec.reset();
				stateVec = std::move(nStateVec);
				stateBuffer = cl::Buffer(*(clObj->GetContextPtr()), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * nMaxQPower, &(stateVec[0]));
				qubitCount = nQubitCount;
				maxQPower = nMaxQPower;

				UpdateRunningNorm(false);
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
				enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				stateVec.reset();
				std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
				stateVec = std::move(sv);
				stateBuffer = cl::Buffer(*(clObj->GetContextPtr()), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * remainderPower, &(stateVec[0]));
				cl::enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
				qubitCount = qubitCount - length;
				maxQPower = 1<<qubitCount;

				double angle = Rand() * 2.0 * M_PI;
				Complex16 phaseFac(cos(angle), sin(angle));
				double totProb = 0.0;
				for (i = 0; i < partPower; i++) {
					totProb += partStateProb[i];
					destination.stateVec[i] = sqrt(partStateProb[i]) * phaseFac;
				}
				if (totProb == 0.0) {
					destination.stateVec[0] = phaseFac;
				}

				angle = Rand() * 2.0 * M_PI;
				phaseFac = Complex16(cos(angle), sin(angle));
				totProb = 0.0;
				for (i = 0; i < remainderPower; i++) {
					totProb += remainderStateProb[i];
					stateVec[i] = sqrt(remainderStateProb[i]) * phaseFac;
				}
				if (totProb == 0.0) {
					stateVec[0] = phaseFac;
				}

				UpdateRunningNorm(true);
				destination.UpdateRunningNorm(true);
			}

			//Logic Gates:
			///Classical "AND" compare two bits in register, and store result in first bit
			void AND(bitLenInt resultBit, bitLenInt compareBit) {
				//if ((resultBit >= qubitCount) || (compareBit >= qubitCount))
				//	throw std::invalid_argument("AND tried to operate on bit index greater than total bits.");
				if (resultBit == compareBit) throw std::invalid_argument("AND bits cannot be the same bit.");

				bool result = M(resultBit);
				bool compare = M(compareBit);
				if (result && !compare) {
					X(resultBit);
				}
			}
			///Classical "OR" compare two bits in register, and store result in first bit
			void OR(bitLenInt resultBit, bitLenInt compareBit) {
				//if ((resultBit >= qubitCount) || (compareBit >= qubitCount))
				//	throw std::invalid_argument("OR tried to operate on bit index greater than total bits.");
				if (resultBit == compareBit) throw std::invalid_argument("OR bits cannot be the same bit.");
				
				bool result = M(resultBit);
				bool compare = M(compareBit);
				if (!result && compare) {
					X(resultBit);
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

				bitCapInt qPowers[4];
				qPowers[1] = 1 << control1;
				qPowers[2] = 1 << control2;
				qPowers[3] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
				bitCapInt qPowersSorted[3];
				std::copy(qPowers, qPowers + 4, qPowersSorted);
				std::sort(qPowersSorted, qPowersSorted + 3);

				Apply2x2(qPowers[0], qPowers[1] + qPowers[2], pauliX, 3, qPowersSorted, false);
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
			///Hadamard gate
			void H(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("H tried to operate on bit index greater than total bits.");
				if (runningNorm != 1.0) NormalizeState();

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
					par_for_all (0, maxQPower, &(stateVec[0]), Complex16(cosine / nrmlzr, sine), NULL, qPowers,
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
					par_for_all (0, maxQPower, &(stateVec[0]), Complex16(cosine / nrmlzr, sine), NULL, qPowers,
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

				UpdateRunningNorm(true);

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
				qPowers[1] = 1 << qubitIndex1;
				qPowers[2] = 1 << qubitIndex2;
				qPowers[0] = qPowers[1] + qPowers[2];
				bitCapInt qPowersSorted[2];
				if (qubitIndex1 < qubitIndex2) {
					qPowersSorted[0] = qPowers[1];
					qPowersSorted[1] = qPowers[2];
				}
				else {
					qPowersSorted[0] = qPowers[2];
					qPowersSorted[1] = qPowers[1];
				}
				Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false);
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
			///Arithmetic shift left, of a numbered register, with last 2 bits as sign and carry
			void ASL(bitLenInt shift, bitLenInt regIndex) {
				ASL(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
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
			///Arithmetic shift left, of a numbered register, with last 2 bits as sign and carry
			void ASR(bitLenInt shift, bitLenInt regIndex) {
				ASR(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
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
			///Logical shift left, of a numbered register, filling the extra bits with |0>
			void LSL(bitLenInt shift, bitLenInt regIndex) {
				LSL(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
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
			///Logical shift left, of a numbered register, filling the extra bits with |0>
			void LSR(bitLenInt shift, bitLenInt regIndex) {
				LSR(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
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
			///"Circular shift left" of a numbered register - shift bits left, and carry last bits.
			void ROL(bitLenInt shift, bitLenInt regIndex) {
				ROL(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
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
			///"Circular shift right" of a numbered register - shift bits left, and carry last bits.
			void ROR(bitLenInt shift, bitLenInt regIndex) {
				ROR(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
			}
			///Add integer (without sign)
			void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (toAdd > 0)) {
					bitCapInt i, j;
					bitCapInt startPower = 1<<start;
					bitCapInt endPower = 1<<(start + length);
					bitCapInt maxLCV = maxQPower - endPower - startPower;
					par_for_reg(startPower, endPower, maxLCV, toAdd, &(stateVec[0]), this,
						[](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower, const bitCapInt toAdd, Complex16* stateVec, CoherentUnit* caller) {
							caller->RotateComplex(k, k + endPower, toAdd, true, startPower, stateVec);
						}
					);
				}
			}
			///Add integer, to a numbered register, (without sign)
			void INC(bitLenInt shift, bitLenInt regIndex) {
				INC(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
			}
			///Subtract integer (without sign)
			void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (toSub > 0)) {
					bitCapInt i, j;
					bitCapInt startPower = 1<<start;
					bitCapInt endPower = 1<<(start + length);
					bitCapInt maxLCV = maxQPower - endPower - startPower;
					par_for_reg(startPower, endPower, maxLCV, toSub, &(stateVec[0]), this,
						[](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower, const bitCapInt toSub, Complex16* stateVec, CoherentUnit* caller) {
							caller->RotateComplex(k, k + endPower, toSub, false, startPower, stateVec);
						}
					);
				}
			}
			///Subtract integer, from a numbered register, (without sign)
			void DEC(bitLenInt shift, bitLenInt regIndex) {
				DEC(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
			}
			///Add (with sign, with carry bit, carry overflow to minimum negative)
			void SINC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (toAdd > 0)) {
					bitCapInt i, j;
					bitCapInt end = start + length;
					bitCapInt startPower = 1<<start;
					bitCapInt endPower = 1<<end;
					bitCapInt maxLCV = maxQPower - endPower - startPower;

					Swap(end - 1, end - 2);
					ROL(1, start, length);
					par_for_reg(startPower, endPower, maxLCV, toAdd, &(stateVec[0]), this,
						[](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower, const bitCapInt toAdd, Complex16* stateVec, CoherentUnit* caller) {
							caller->RotateComplex(k + 1, k + endPower + 1, toAdd - 1, true, startPower<<1, stateVec);
							caller->RotateComplex(k, k + endPower, toAdd, true, startPower<<1, stateVec);
						}
					);
					ROR(1, start, length);
					Swap(end - 1, end - 2);
				}			
			}
			///Add integer, to a numbered register, (with sign, with carry bit, carry overflow to minimum negative)
			void SINC(bitLenInt shift, bitLenInt regIndex) {
				SINC(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
			}
			///Subtract (with sign, with carry bit, carry overflow to maximum positive)
			void SDEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
				if ((length > 0) && (toSub > 0)) {
					bitCapInt i, j;
					bitCapInt end = start + length;
					bitCapInt startPower = 1<<start;
					bitCapInt endPower = 1<<end;
					bitCapInt maxLCV = maxQPower - endPower - startPower;

					Swap(end - 1, end - 2);
					ROL(1, start, length);
					par_for_reg(startPower, endPower, maxLCV, toSub, &(stateVec[0]), this,
						[](const bitCapInt k, const int cpu, const bitCapInt startPower, const bitCapInt endPower, const bitCapInt toSub, Complex16* stateVec, CoherentUnit* caller) {
							caller->RotateComplex(k, k + endPower, toSub - 1, false, startPower<<1, stateVec);
							caller->RotateComplex(k + 1, k + endPower + 1, toSub, false, startPower<<1, stateVec);
						}
					);
					ROR(1, start, length);
					Swap(end - 1, end - 2);
				}
			}
			///Add integer, to a numbered register, (with sign, with carry bit, carry overflow to minimum negative)
			void SDEC(bitLenInt shift, bitLenInt regIndex) {
				SDEC(shift, registerDims[regIndex].startBit, registerDims[regIndex].length);
			}
			/// Quantum Fourier Transform - Apply the quantum Fourier transform to a bit segment
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
			///Quantum Fourier Transform - Apply the quantum Fourier transform to a numbered register
			void QFT(bitLenInt regIndex) {
				QFT(registerDims[regIndex].startBit, registerDims[regIndex].length);
			}

		private:
			double runningNorm;
			bitLenInt qubitCount;
			bitCapInt maxQPower;
			bitLenInt registerCount;
			std::unique_ptr<RegisterDim[]> registerDims;
			std::unique_ptr<Complex16[]> stateVec;

			std::default_random_engine rand_generator;
			std::uniform_real_distribution<double> rand_distribution;

			OCLSingleton* clObj;
			cl::CommandQueue queue;
			cl::Buffer stateBuffer;
			cl::Buffer cmplxBuffer;
			cl::Buffer ulongBuffer;
			cl::Buffer nrmBuffer;
			cl::Buffer maxBuffer;

			void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx,
					const bitLenInt bitCount, const bitCapInt* qPowersSorted, bool doCalcNorm) {
				bitCapInt mxI[1] = {maxQPower};
				Complex16 cmplx[5];
				for (int i = 0; i < 4; i++){
					cmplx[i] = mtrx[i];
				}
				cmplx[4] = Complex16(1.0 / runningNorm, 0.0);
				bitCapInt ulong[7] = {bitCount, maxQPower, offset1, offset2, 0, 0, 0};
				for (int i = 0; i < bitCount; i++) {
					ulong[4 + i] = qPowersSorted[i];
				}

				queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				queue.enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(Complex16) * 5, cmplx);
				queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 7, ulong);

				cl::Kernel apply2x2 = *(clObj->GetApply2x2Ptr());
				queue.finish();
				apply2x2.setArg(0, stateBuffer);
				apply2x2.setArg(1, cmplxBuffer);
				apply2x2.setArg(2, ulongBuffer);
				queue.enqueueNDRangeKernel(apply2x2, cl::NullRange,  // kernel, offset
            				cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
					cl::NDRange(1)); // local number (per group)

				if (doCalcNorm) {
					UpdateRunningNorm(false);
				}
				else {
					runningNorm = 1.0;
				}
			}

			void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm) {
				bitCapInt qPowers[1];
				qPowers[0] = 1 << qubitIndex;
				Apply2x2(qPowers[0], 0, mtrx, 1, qPowers, doCalcNorm);
			}

			void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm) {
				bitCapInt qPowers[3];
				qPowers[1] = 1 << control;
				qPowers[2] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2];
				bitCapInt qPowersSorted[2];
				if (control < target) {
					qPowersSorted[0] = qPowers[1];
					qPowersSorted[1] = qPowers[2];
				}
				else {
					qPowersSorted[0] = qPowers[2];
					qPowersSorted[1] = qPowers[1];
				}

				Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
			}

			void InitOCL() {
				clObj = OCLSingleton::Instance();

				queue = *(clObj->GetQueuePtr());
				cl::Context context = *(clObj->GetContextPtr());

				// create buffers on device (allocate space on GPU)
				stateBuffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(stateVec[0]));
				cmplxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Complex16) * 5);
				ulongBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Complex16) * 7);
				nrmBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
				maxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt));

				queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
			}

			void NormalizeState() {
				Complex16* sortedStateVec = new Complex16[maxQPower];
				std::copy(&(stateVec[0]), &(stateVec[0]) + maxQPower, sortedStateVec);
				runningNorm = 1/par_norm(maxQPower, sortedStateVec);
				delete [] sortedStateVec;

				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					stateVec[lcv] *= runningNorm;
				}
				runningNorm = 1.0;
			}

			void UpdateRunningNorm(bool isMapped) {
				if (isMapped) {
					queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
				}
				double nrmParts[CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE] = {0};
				queue.enqueueWriteBuffer(maxBuffer, CL_FALSE, 0, sizeof(bitCapInt), &(maxQPower));
				queue.enqueueWriteBuffer(nrmBuffer, CL_FALSE, 0, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);				

				cl::Kernel updateRunningNorm = *(clObj->GetUpdateRunningNormPtr());
				queue.finish();
				updateRunningNorm.setArg(0, stateBuffer);
				updateRunningNorm.setArg(1, maxBuffer);
				updateRunningNorm.setArg(2, nrmBuffer);
				queue.enqueueNDRangeKernel(updateRunningNorm, cl::NullRange,  // kernel, offset
            				cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
					cl::NDRange(1)); // local number (per group)

				// read result from GPU to here
				runningNorm = 0;
				queue.enqueueReadBuffer(nrmBuffer, CL_TRUE, 0, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
				for (int i = 0; i < CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE; i++) {
					runningNorm += nrmParts[i];
				}
				runningNorm = sqrt(runningNorm);

				// read result from GPU to here
				queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
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

			void ReverseComplex(bitCapInt start, bitCapInt end, bitCapInt stride, Complex16* vec) {
				if (start + 1 < end) {
					end -= 1;
					Complex16 temp;
					bitCapInt i;
					bitCapInt iter = start + (end - start - 1) / 2;
					for (i = start; i <= iter; i+=stride) {
						temp = vec[i];
						vec[i] = vec[end - i + start];
						vec[end - i + start] = temp;
					}
				}
			}

			void RotateComplex(bitCapInt start, bitCapInt end, bitCapInt shift, bool leftRot, bitCapInt stride, Complex16* vec) {
				shift *= stride;
				if (shift > 0) {
					if (leftRot) {
						ReverseComplex(start, end, stride, vec);
						ReverseComplex(start, start + shift - stride, stride, vec);
						ReverseComplex(start + shift, end, stride, vec);
					}
					else {
						ReverseComplex(start + shift, end, stride, vec);
						ReverseComplex(start, start + shift - stride, stride, vec);
						ReverseComplex(start, end, stride, vec);
					}
				}
			}
	};
}
