cd bayesian_eit/cpp
mkdir -p build
cd build
cmake ..
make
cd ../../python
python setup.py develop
cd ../..

cd bayesian_pdes
python setup.py develop
cd ..

cd mcmc
python setup.py develop
