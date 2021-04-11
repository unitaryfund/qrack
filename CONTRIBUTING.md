# Contributing

**Qrack wants to get you hooked! All contributions are welcome. The core team is just figuring out quantum physics for ourselves.**

"...I might think about it a little bit and if I can't figure it out, then I go on to something else, but I don't have to know an answer, I don't feel frightened by not knowing things, by being lost in a mysterious universe without having any purpose, which is the way it really is so far as I can tell. It doesn't frighten me."
—Richard P. Feynman

## "The Mission"

[bennbollay](https://github.com/bennbollay) and [WrathfulSpatula](https://github.com/WrathfulSpatula) started Qrack with a hope of providing the basis of an open source framework for a practical software interface between quantum computers and the "classical" ones we have today, among other goals. A quantum simulator could serve many other functions: it could serve to simulate realistically "noisy" quantum computers to aid development of genuinely quantum hardware, or it could be an excellent teaching tool that is more concerned with communicating the concepts and methodology of quantum programming than with burning out graphics cards for raw computing power. We hope Qrack can maybe also be a utility for research and teaching, but our primary intent is for it to be the open source simulator you'd reach for when you need an interface between between your classical personal computer and a true "QPU." We hope that, as quantum logic is adopted into new standards, Qrack also gives you an efficient abstraction layer on top of arbitrary classical parallelism, to get the full benefit of technologies like OpenCL running on your GPU in a few dozen lines of code, for simple tasks like a lookup table search, integer factoring, or a discrete Fourier transform.

We're flexible in where **you** choose to take Qrack, though. Benn and Daniel feel strongly that quantum computing technologies are too potentially disruptive, even to day-to-day life, to not have strong open source resources available for everyone. The resources are only as good as their use to you, though, in your day-to-day life. We also have trouble anticipating exactly where the best positive impact can even be made, with Qrack. So, if something you want to contribute feels like it's "tangential" or "other" with regards to Qrack, please open a pull request or an issue anyway, because this software is for you, wherever you need it.

## How to get started

```
   $ git clone https://github.com/vm6502q/qrack.git
   $ cd qrack
   $ mkdir _build && cd _build
   $ cmake ..
   $ make all
   $ ./unittest
```

...and you've just run your first set of Qrack unit tests. (Please tell us, if they break!) Qrack will run without OpenCL installed, but poorly. Even if you intend to run on a CPU, we suggest that you use the device selection parameters and methods in OCLEngine and QEngineOCL to just select your CPU as the OpenCL device, from the indexed list of devices Qrack automatically prints when running with OpenCL.

More information is availabe in the [README](https://github.com/vm6502q/qrack/blob/master/README.md), [the official documentation and API reference](https://qrack.readthedocs.io/en/latest/), and from Benn and Dan on Discord, at the [Advanced Computing Topics](https://discordapp.com/invite/Gj3CHDy) server. We're a friendly bunch. We're happy to help you write your "Hello Quantum World!"

## Good First Issues

This section and the next will be updated, as needs change. Even if you've never written a quantum program before, there are a few things you could do with the code base that would really help us out:

- **Examples and tutorials**: As someone new to Qrack, you have an advantage over the core team members: your early "scratch work" (and documentation) in figuring out how to use Qrack might be more instructive to other new users than what Benn and Dan can easily produce as teaching tools. If you play with Qrack at all, and you get it do anything cool, chances are that it'd be cool and instructive to other new users, as well. In particular, see [VM6502Q and it's dummy.ram examples](https://github.com/vm6502q/vm6502q/blob/master/dummy.ram) as well as the [examples repo](https://github.com/vm6502q/examples), for help getting started.

- **Documentation proofreading and editing**: If something is grammatically incorrect in the Doxygen or [https://qrack.readthedocs.io](https://qrack.readthedocs.io/en/latest/), or just plain confusing, (Dan has a tendency to get carried away,) it'd help us greatly for you to suggest changes on them, and we could always benefit from fresh eyes that aren't set in a particular way of seeing things.

- **Normalizing method signatures**: Qrack is based on the "gate model" of quantum computation, but it provides single bit gates, length-wise parallel gates, and gates with arbitrary controls bits, each with one method signature. These signature templates were chosen as the need for them became apparent to us, and they're fairly consistent, but not 100% consistent and template-able. If you could fit all the QInterface API method signatures to a normal template, we'd be grateful, and we could review your convention. You might want to open a Github issue first, proposing your planned template, and Benn and Dan could help with suggestions and tips.

- **Qubit mapper layer**: By default, Qrack's qubit addressing is a like a 1 dimensional array of qubits, starting from index 0. This is simple, and it's proved fairly versatile. It would be good to have a layer on top of "QEngine" types that acts as an arbitrary dictionary for qubits, or perhaps a multidimensional array. See the implementation of "QUnit," and its "shards," for an idea of how that mechanism might be turned into a general and stand-alone layer.

- **Windows and Mac build testing**: The core developers intend to support all of Linux, Windows, and Mac, and have built on all of the above. We primarily develop and test on Linux machines, and cmake builds might develop bugs on other platforms as we go. If you could test on your system, particularly Windows or Mac, and figure out how to get the build working on your machine, we would be eternally grateful, and include you in the changelog notes when you open a pull request.

## Advanced Issues

- **Embedding and wrapping in C#, F#, or Python**: Qrack tries to keep to the reasonably "low level" standard of C++ for portability and performance, but there's no reason not to give the Python and .NET developers some love. [We've looked at ProjectQ for Python wrapping and a quantum compiler](https://github.com/WrathfulSpatula/ProjectQ/tree/qrack_simulator), but this is still new territory for us.

- **Backport to C**: Be the legendary hero, and have Linus Torvalds personally fall at your feet with gratitude for bringing the quantum revolution to the Linux kernel, or something like that. Yeah, maybe that's grandiose, but a "straight C" language backport **might** open up such wild possibilities, with potentially better performance, all wrapped up on top with a Python-based quantum compiler. This would honestly be pretty godly.

- **Practical Multiprocessor Support**: Qrack has two experimental engine types for distributing work to multiple processors, the "QPager" type, which is more QEngine (and QHiPSTER) -like, and QUnitOCLMulti, which is QUnit-like and distributes separable subsystems across different processors. Limited multiprocessor use cases of either have achieved practical return, but if you can make either of these or a Qrack-based design of your own outperform QEngineOCL and/or QUnit on a PC, as the case may be, you've just earned yourself what could very well be world's fastest cluster-distributed quantum computer simulation, (at least with the help of a transparent OpenCL clustering framework like VirtualCL/SnuCL).

## Further Reading on Quantum Theory

For quantum computation, Nielson and Chuang's **Quantum Computation and Quantum Information** has come to be considered a standard and complete text. It's divided into sections intended to be useful to both, and respectively, those coming from physics and those coming from computer science, to bring everyone onto the same page by the end of the book. Also, come visit us at the [Advanced Computing Topics](https://discordapp.com/invite/Gj3CHDy) server, and we might have more timely and targeted suggestions for you.
