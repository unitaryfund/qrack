#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "qrack.hpp"

int main() {
	char testKey;

	const int planckTimes = 65500;
        const int mpPowerOfTwo = 16;
	const int maxTrials = 1000;

	int i, j;

	Qrack::Register qftReg(8, 1);

	double qftProbs[8];
	std::cout<<"LSL Test:"<<std::endl;
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			std::cout<<qftReg.Prob(j);
		}
		std::cout<<"->";
		qftReg.LSL();
		for (j = 0; j < 8; j++) {
			std::cout<<qftReg.Prob(j);
		}
		std::cout<<std::endl;	
	}
	std::cout<<"LSR Test:"<<std::endl;
	qftReg.SetPermutation(128);
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			std::cout<<qftReg.Prob(j);
		}
		std::cout<<"->";
		qftReg.LSR();
		for (j = 0; j < 8; j++) {
			std::cout<<qftReg.Prob(j);
		}
		std::cout<<std::endl;	
	}

	qftReg.SetPermutation(0);
	for (i = 0; i < 8; i++) {
		qftReg.H(i);
	}

	std::cout<<"M Test:"<<std::endl;
	std::cout<<"Initial:"<<std::endl;
	for (i = 0; i < 8; i++) {
		qftProbs[i] = qftReg.Prob(i);
		std::cout<<"Bit "<<i<<", Chance of 1:"<<qftProbs[i]<<std::endl;
	}

	qftReg.M(0);
	std::cout<<"Final:"<<std::endl;
	for (i = 0; i < 8; i++) {
		qftProbs[i] = qftReg.Prob(i);
		std::cout<<"Bit "<<i<<", Chance of 1:"<<qftProbs[i]<<std::endl;
	}

	qftReg.SetPermutation(85);

	std::cout<<"Quantum Fourier transform of 85 (1+4+16+64), with 1 bits first passed through Hadamard gates:"<<std::endl;

	for (i = 0; i < 8; i+=2) {
		qftReg.H(i);
	}	

	std::cout<<"Initial:"<<std::endl;
	for (i = 0; i < 8; i++) {
		qftProbs[i] = qftReg.Prob(i);
		std::cout<<"Bit "<<i<<", Chance of 1:"<<qftProbs[i]<<std::endl;
	}

	qftReg.QFT();

	std::cout<<"Final:"<<std::endl;
	for (i = 0; i < 8; i++) {
		qftProbs[i] = qftReg.Prob(i);
		std::cout<<"Bit "<<i<<", Chance of 1:"<<qftProbs[i]<<std::endl;
	}

	std::cout<<"Next step might take a while..."<<std::endl;

	Qrack::Register qReg(mpPowerOfTwo, 0);

	//50/50 Superposition between "step" and "don't step" at each discrete time step
	//is equivalent to Pi/4 rotation around the y-axis of spin, each time: 
	double angle = -M_PI / 4.0;
	//This is our starting distance from the destination point (plus one).
	unsigned int power = 1 << mpPowerOfTwo;

	//We will store our ultimate position in this variable and return it from the operation:
	unsigned int toReturn = 0;

	double* zeroProbs = new double[mpPowerOfTwo];

	//This isn't exactly the same as a classical unidirectional discrete random walk.
	//At each step, we superpose our current distance from the destination with a distance
	//that is one unit shorter. This is equivalent to a rotation around the y-axis,
	//"Ry(Pi/4, qubit[0])", where qubit[0] is the least significant qubit of our distance.
	//Four successive steps of superposition then give a rotation of Pi.
	//A rotation of Pi about the y-axis MUST turn a pure state of |0> into a pure state of |1>
	//and vice versa.
	// Hence, we already know a maximum amount of time steps this could take, "power * 4".
	// We can just return zero if our time step count is greater than or equal to this.
	if (planckTimes / 4 < power) {
		//If we haven't exceeded the known maximum time, we carry out the algorithm.
		//We grab enough qubits and set them to the initial state.
		//Weirdly, we use |0> to represent 1 and |1> to represent 0,
		//just so we can avoid many unnecessary "not" gates, "X(...)" operations.
		
		//double testProb[power];
		//double totalProb;
		unsigned int workingPower = 1;
		for (i = 1; i <= planckTimes; i++) {
			//For each time step, first increment superposition in the least significant bit:
			qReg.RY(angle, 0);
			//At 2 steps, we could have a change in the second least significant bit.
			//At 4 steps, we could have a change in the third least significant bit AND the second least.
			//At 8 steps, we could have a change in the fourth least, third least, and second least.
			//(...Etc.)
			workingPower = 1;
			for (j = 1; j < mpPowerOfTwo; j++) {
				workingPower = workingPower << 1;
				if (i % workingPower == 0) {
					//"CNOT" is a quantum "controlled not" gate.
					//If the control bit is in a 50/50 superposition, for example,
					// the other input bit ends up in 50/50 superposition of being reversed, or "not-ed".
					// The two input bits can become entangled in this process! If the control bit were next
					// immediately measured in the |1> state, we'd know the other input qubit was flipped.
					// If the control bit was next immediately measured in the |0> state, we'd know the other input
					// qubit was not flipped.

					//(Here's where we avoid two unnecessary "not" or "X(...)" gates by flipping our 0/1 convention:)
					qReg.CNOT(j - 1, j);
				}
			}

			//qReg.ProbArray(testProb);
			//totalProb = 0.0;
			//for (j = 0; j < power; j++) {
			//	totalProb += testProb[j];
			//}
			//if (totalProb < 0.999 || totalProb > 1.001) {
			//	for (j = 0; j < power; j++) {
			//		std::cout<<j<<" Prob:"<<testProb[j]<<std::endl;
			//	}
			//	std::cout<<"Total Prob is"<<totalProb<<" at iteration "<<i<<"."<<std::endl;
			//	std::cin >> testKey;
			//}
		}

		//The qubits are now in their fully simulated, superposed and entangled end state.
		// Ultimately, we have to measure a final state in the |0>/|1> basis for each bit, breaking the
		// superpositions and entanglements. Quantum measurement is nondeterministic and introduces randomness,
		// so we repeat the simulation many times in the driver code and average the results.
		
		for (j = 0; j < mpPowerOfTwo; j++) {
			zeroProbs[j] = 1.0 - qReg.Prob(j);
			std::cout<<"Bit "<<j<<", Chance of 0:"<<zeroProbs[j]<<std::endl;
		}
	}

	unsigned int outcome;
	unsigned int* masses = new unsigned int[1000];
	double totalMass = 0;
	for (i = 0; i < maxTrials; i++) {
		outcome = 0;
		for (j = 0; j < mpPowerOfTwo; j++) {
			if (qReg.Rand() < zeroProbs[j]) {
				outcome += 1 << j;
			}
		}
		masses[i] = outcome;
		totalMass += outcome;
	}

	delete [] zeroProbs;

	
	double averageMass = totalMass / maxTrials;
	double sqrDiff = 0.0;
	double diff;
	//Calculate the standard deviation of the simulation trials:
	for (int trial = 0; trial < maxTrials; trial++)
	{
		diff = masses[trial] - averageMass;
		sqrDiff += diff * diff;
	}
	double stdDev = sqrt(sqrDiff / (maxTrials - 1));

	std::cout<<"Trials:"<<maxTrials<<std::endl;
	std::cout<<"Starting Point:"<<((1<<mpPowerOfTwo) - 1)<<std::endl;
	std::cout<<"Time units passed:"<<planckTimes<<std::endl;
	std::cout<<"Average distance left:"<<averageMass<<std::endl;
	std::cout<<"Distance left std. dev.:"<<stdDev<<std::endl;

	//("Hello, Universe!")
}
