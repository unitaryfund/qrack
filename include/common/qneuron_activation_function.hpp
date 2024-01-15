//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

namespace Qrack {
/**
 * Enumerated list of activation functions
 */
enum QNeuronActivationFn {
    /// Default
    Sigmoid = 0,
    /// Rectified linear
    ReLU = 1,
    /// Gaussian linear
    GeLU = 2,
    /// Version of (default) "Sigmoid" with tunable sharpness
    Generalized_Logistic = 3,
    /// Leaky rectified linear
    Leaky_ReLU = 4
};
} // namespace Qrack
