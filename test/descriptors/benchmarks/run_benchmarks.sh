cd ../../../ && scons -j123
cd test/descriptors/benchmarks

scons -j123
rm -r dbs
mkdir dbs

./search_comparison
