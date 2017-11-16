cd ../../../ && scons -j123
cd test/benchmarks/descriptors

scons -j123
rm -r dbs
mkdir dbs

./search_comparison
