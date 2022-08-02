
sudo apt update
sudo apt install -y libcgal-dev

mkdir -p evaluation/result


cd evaluation/tf_ops/approxmatch
bash tf_approxmatch_compile.sh
cd ../nn_distance
bash tf_nndistance_compile.sh
cd ../../..



cd evaluation/evaluation_code
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp ./evaluation ../../

cd ../..

mkdir -p result

