cd ../../ && scons -j123
cd test/descriptors
scons -j123
rm -r dbs
mkdir dbs
./main

