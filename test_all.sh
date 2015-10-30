#! /bin/bash

clear

echo "Run unit tests..."
echo
echo "Building tests..."

make clean
make test_eafe
make test_faspfenics

echo
echo "Running unit tests..."

./test_eafe
./test_faspfenics

echo "Done testing"
