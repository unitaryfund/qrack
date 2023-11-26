-include ./Makefile

clean-cmake:
	rm -f -r CMakeFiles
	rm -f CMakeCache.txt
	rm -f *.cmake

clean:
	rm -f -r CMakeFiles
	rm -f CMakeCache.txt
	rm -f *.cmake
	rm -f Makefile
	cp -r * ../
	rm -f makefile
	cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib -DPACK_DEBIAN=ON -DCPP_STD=14 -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_OPENCL=ON -DQBCAPPOW=12 ..
