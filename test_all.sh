#! /bin/bash

### Bash script that put the code throught all the tests in tests/
### run it as "./test_all.sh" to test normally or "./test_all.sh DEBUG" to test with DEBUG=true


clear

echo "Run unit tests..."
echo
echo "Building tests..."

make test_eafe
make test_faspfenics
make test_faspsolver
make test_bc
make test_add_matrix
make test_replace_matrix
make test_lin_pnp

echo
echo "Running unit tests..."

if [ "$1"=="DEBUG" ]; then
	./test_eafe $1
	./test_faspfenics $1
	./test_faspsolver $1
	./test_bc $1
	./test_add_matrix $1
	./test_replace_matrix $1
	./test_lin_pnp $1

else
	./test_eafe
	./test_faspfenics
	./test_faspsolver
	./test_bc
	./test_add_matrix
	./test_replace_matrix
	./test_lin_pnp
fi


echo "Done testing"
