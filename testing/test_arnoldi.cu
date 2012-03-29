/*
 * test_arnoldi.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: Squillace Filippo
 */

//#define CUSP_USE_TEXTURE_MEMORY


#include <iostream>
#include <sstream>
#include <string.h>


#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <string.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>



#include <cusp/krylov/arnoldi.h>
#include <cusp/detail/matrix_base.h>
//#include "../../cusp/krylov/arnoldi.h"


#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestSuite.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

void checkStatus(culaStatus status)
{
	if(!status)
		return;
	if(status == culaArgumentError)
		printf("Invalid value for parameter %d\n", culaGetErrorInfo());
	else if(status == culaDataError)
		printf("Data error (%d)\n", culaGetErrorInfo());
	else if(status == culaBlasError)
		printf("Blas error (%d)\n", culaGetErrorInfo());
	else if(status == culaRuntimeError)
		printf("Runtime error (%d)\n", culaGetErrorInfo());
	else
		printf("%s\n", culaGetStatusString(status));

	culaShutdown();
	exit(EXIT_FAILURE);
}


class ArnoldiTestCase : public CppUnit::TestFixture {

	CPPUNIT_TEST_SUITE (ArnoldiTestCase);
	CPPUNIT_TEST (test_host_arnoldi);
	CPPUNIT_TEST(test_device_arnoldi);
	CPPUNIT_TEST(test_host_qr_shift);
	CPPUNIT_TEST(test_device_qr_shift);
	CPPUNIT_TEST(test_host_iram);
	CPPUNIT_TEST(test_device_iram);
	CPPUNIT_TEST_SUITE_END ();

	typedef int    IndexType;
	typedef float ValueType;
	typedef cusp::array2d<float,cusp::device_memory, cusp::column_major> DeviceMatrix_array2d;
	typedef cusp::array2d<float, cusp::host_memory, cusp::column_major>   HostMatrix_array2d;

	typedef cusp::array1d<float,cusp::device_memory> DeviceVector_array1d;
	typedef cusp::array1d<float, cusp::host_memory>   HostVector_array1d;

	typedef cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix_csr;
	typedef cusp::csr_matrix<IndexType, float, cusp::host_memory>   HostMatrix_csr;

private:

	std::vector<std::string> path_def_pos;
	std::vector<DeviceMatrix_csr> dev_mat_def_pos;
	std::vector<HostMatrix_csr> host_mat_def_pos;


public:

	void setUp()
	{

		culaStatus status;
		status = culaInitialize();
		checkStatus(status);


		// ################################ POSITIVE DEFINITE #####################
		path_def_pos = std::vector<std::string>(1);
//		path_def_pos[0] = "data/positive-definite/lehmer10.mtx";
//		path_def_pos[1] = "data/positive-definite/lehmer20.mtx";
//		path_def_pos[0] = "data/positive-definite/lehmer50.mtx";
//		path_def_pos[0] = "data/positive-definite/lehmer100.mtx";
		path_def_pos[0] = "data/positive-definite/lehmer200.mtx";
//		path_def_pos[0] = "data/positive-definite/moler200.mtx";
//		path_def_pos[0] = "data/L11_4_ringhals.mtx";

		host_mat_def_pos = std::vector<HostMatrix_csr>(path_def_pos.size());
		dev_mat_def_pos = std::vector<DeviceMatrix_csr>(path_def_pos.size());
		for(size_t i=0; i<path_def_pos.size(); i++){
			cusp::io::read_matrix_market_file(host_mat_def_pos[i], path_def_pos[i]);
			dev_mat_def_pos[i] = DeviceMatrix_csr(host_mat_def_pos[i]);
		}


	}

	void tearDown()
	{
		culaShutdown();
	}

	void test_host_arnoldi()
	{

		for(size_t i=0; i<path_def_pos.size(); i++){

			size_t m = 10;
			HostMatrix_array2d H(m, m);
			HostMatrix_array2d V(host_mat_def_pos[i].num_rows, m);
			HostVector_array1d f(host_mat_def_pos[i].num_rows, ValueType(0));

			cusp::krylov::arnoldi(host_mat_def_pos[i], H, V, f, 0, 3);
			cusp::krylov::arnoldi(host_mat_def_pos[i], H, V, f, 2, 5);
			cusp::krylov::arnoldi(host_mat_def_pos[i], H, V, f, 4, m);

			HostMatrix_array2d A2d;
			HostMatrix_array2d V2;
			HostMatrix_array2d H2;

			HostMatrix_array2d C;
			HostMatrix_array2d C2;


			size_t N = host_mat_def_pos[i].num_rows;

			cusp::convert(host_mat_def_pos[i], A2d);


			// create submatrix V2
			cusp::copy(V, V2);
			V2.resize(N,m);

			// create submatrix H2
			H2.resize(m,m);
			size_t l = H.num_rows;
			for(size_t j=0; j<m; j++)
				thrust::copy(H.values.begin()+ l*j, H.values.begin()+ l*j +m, H2.values.begin()+ m*j);

			cusp::multiply(A2d, V2, C);

			cusp::multiply(V2, H2, C2);

			cusp::blas::axpy(f.begin() , f.end(), C2.values.begin()+(m-1)*N, ValueType(1));


			ValueType errRel = nrmVector("host_arnoldi: "+path_def_pos[i], C.values, C2.values);
			CPPUNIT_ASSERT( errRel < 1.0e-5 );

		}
	}



	void test_device_arnoldi()
	{


		for(size_t i=0; i<path_def_pos.size(); i++){

			size_t m = 10;
			DeviceMatrix_array2d H(m, m);
			DeviceMatrix_array2d V(dev_mat_def_pos[i].num_rows, m);
			DeviceVector_array1d f(dev_mat_def_pos[i].num_rows, ValueType(0));

			//		  DeviceMatrix_csr dev_mat;
			//		  cusp::convert(dev_mat_def_pos[i], dev_mat);
			cusp::krylov::arnoldi(dev_mat_def_pos[i], H, V, f, 0, 3);
			cusp::krylov::arnoldi(dev_mat_def_pos[i], H, V, f, 2, 5);
			cusp::krylov::arnoldi(dev_mat_def_pos[i], H, V, f, 4, m);

			HostMatrix_array2d A2d;
			HostMatrix_array2d V2;
			HostMatrix_array2d H2;

			HostMatrix_array2d C;
			HostMatrix_array2d C2;
			HostVector_array1d f_host;
			cusp::convert(f,f_host);

			size_t N = host_mat_def_pos[i].num_rows;

			cusp::convert(host_mat_def_pos[i], A2d);


			// create submatrix V2
			cusp::convert(V, V2);
			V2.resize(N,m);

			// create submatrix H2
			H2.resize(m,m);
			size_t l = H.num_rows;
			for(size_t j=0; j<m; j++)
				thrust::copy(H.values.begin()+ l*j, H.values.begin()+ l*j +m, H2.values.begin()+ m*j);

			cusp::multiply(A2d, V2, C);

			cusp::multiply(V2, H2, C2);

			cusp::blas::axpy(f_host.begin() , f_host.end(), C2.values.begin()+(m-1)*N, float(1));


			ValueType errRel = nrmVector("device_arnoldi: "+path_def_pos[i], C.values, C2.values);
			CPPUNIT_ASSERT( errRel < 1.0e-6 );
		}
	}

	void test_host_qr_shift()
	{
		for(size_t i=0; i<path_def_pos.size(); i++){

			size_t N = host_mat_def_pos[i].num_rows;
			size_t k = 5;
			size_t m = 2*k+1;


			HostMatrix_array2d H(m, m);
			HostMatrix_array2d V(N, m);
			HostVector_array1d f(N, ValueType(0));
			HostMatrix_array2d H_(m,m);
			cusp::array1d<int, cusp::host_memory> perm(m);
			HostVector_array1d	eigvals(m, ValueType(0));
			HostMatrix_array2d eigvects_H(m,m);


			cusp::krylov::arnoldi(host_mat_def_pos[i], H, V, f, 0, k);
			cusp::krylov::arnoldi(host_mat_def_pos[i], H, V, f, k-1, m);

			cusp::copy(H, H_);
			cuspla::geev(H_, eigvals, eigvects_H);

			// Selects the last p elements
			// init perm
			for(size_t i=0;i<m; i++) perm[i]=i;
			thrust::sort_by_key(eigvals.begin(), eigvals.end(), perm.begin());


			ValueType sigma;
			cusp::krylov::shifted_qr(H, V, eigvals, m-k, sigma);

			//f(k-1) = V(:, k)*beta + f(m-1)*sigma(k-1)
			ValueType beta = H(k, k-1);


			cusp::blas::axpby(V.values.begin()+N*(k), \
					V.values.begin()+N*(k+1), f.begin(), \
					f.begin(), beta, sigma);




			// ****** BEGIN TEST ******

			HostMatrix_array2d A2d;
			HostMatrix_array2d V2(N,k);
			HostMatrix_array2d H2(k,k);

			HostMatrix_array2d C(N,k);
			HostMatrix_array2d C2(N,k);


			cusp::convert(host_mat_def_pos[i], A2d);


			// create submatrix V2
			for(size_t j=0; j<k; j++)
				thrust::copy(V.values.begin()+ N*j, V.values.begin()+ N*j +N, V2.values.begin()+ N*j);

			// create submatrix H2
			for(size_t j=0; j<k; j++)
				thrust::copy(H.values.begin()+ m*j, H.values.begin()+ m*j +k, H2.values.begin()+ k*j);


			cusp::multiply(A2d, V2, C);

			cusp::multiply(V2, H2, C2);

			cusp::blas::axpy(f.begin() , f.end(), C2.values.begin()+(k-1)*N, ValueType(1));


			ValueType errRel = nrmVector("test_host_qr_shift: "+path_def_pos[i], C.values, C2.values);
			CPPUNIT_ASSERT( errRel < 1.0e-4 );

		}
	}


	void test_device_qr_shift()
	{
		for(size_t i=0; i<path_def_pos.size(); i++){

			size_t N = dev_mat_def_pos[i].num_rows;
			size_t k = 5;
			size_t m = 2*k+1;


			DeviceMatrix_array2d H(m, m);
			DeviceMatrix_array2d V(N, m);
			DeviceVector_array1d f(N, ValueType(0));
			DeviceMatrix_array2d H_(m,m);
			cusp::array1d<int, cusp::device_memory> perm(m);
			DeviceVector_array1d	eigvals(m, ValueType(0));
			DeviceMatrix_array2d eigvects_H(m,m);


			cusp::krylov::arnoldi(dev_mat_def_pos[i], H, V, f, 0, k);
			cusp::krylov::arnoldi(dev_mat_def_pos[i], H, V, f, k-1, m);

			cusp::copy(H, H_);
			cuspla::geev(H_, eigvals, eigvects_H);

			// Selects the last p elements
			// init perm
			for(size_t i=0;i<m; i++) perm[i]=i;
			thrust::sort_by_key(eigvals.begin(), eigvals.end(), perm.begin());


			ValueType sigma;
			cusp::krylov::shifted_qr(H, V, eigvals, m-k, sigma);

			//f(k-1) = V(:, k)*beta + f(m-1)*sigma(k-1)
			ValueType beta = H(k, k-1);


			cusp::blas::axpby(V.values.begin()+N*(k), \
					V.values.begin()+N*(k+1), f.begin(), \
					f.begin(), beta, sigma);




			// ****** BEGIN TEST ******

			HostMatrix_array2d A2d;
			HostMatrix_array2d V2(N,k);
			HostMatrix_array2d H2(k,k);

			HostMatrix_array2d C(N,k);
			HostMatrix_array2d C2(N,k);

			HostVector_array1d f_host(N);
			cusp::copy(f, f_host);

			cusp::convert(host_mat_def_pos[i], A2d);


			// create submatrix V2
			for(size_t j=0; j<k; j++)
				thrust::copy(V.values.begin()+ N*j, V.values.begin()+ N*j +N, V2.values.begin()+ N*j);

			// create submatrix H2
			for(size_t j=0; j<k; j++)
				thrust::copy(H.values.begin()+ m*j, H.values.begin()+ m*j +k, H2.values.begin()+ k*j);


			cusp::multiply(A2d, V2, C);

			cusp::multiply(V2, H2, C2);

			cusp::blas::axpy(f_host.begin() , f_host.end(), C2.values.begin()+(k-1)*N, ValueType(1));


			ValueType errRel = nrmVector("test_device_qr_shift: "+path_def_pos[i], C.values, C2.values);
			CPPUNIT_ASSERT( errRel < 1.0e-4 );

//			CPPUNIT_ASSERT( 0 == 0 );


		}
	}

	void test_host_iram(){
		for(size_t i=0; i<path_def_pos.size(); i++){
			size_t k = 4;

			size_t n = host_mat_def_pos[i].num_rows;
			size_t m = host_mat_def_pos[i].num_cols;
			HostMatrix_array2d eigvects;
			HostMatrix_array2d A2d;
			HostVector_array1d eigvals;
			HostVector_array1d y1, eigvec(m);

			cusp::krylov::implicitly_restarted_arnoldi(host_mat_def_pos[i],\
					eigvals, eigvects, k, 0);


			cusp::convert(host_mat_def_pos[i], A2d);
			for(size_t j=0; j<eigvals.size(); j++){
				thrust::copy(eigvects.values.begin()+ j*n, eigvects.values.begin()+ (j+1)*n,eigvec.begin());
				cuspla::gemv(A2d, eigvec, y1, false);
				cusp::blas::scal(eigvec, (ValueType)eigvals[j]);

				std::stringstream j_str, eigval_str;
				j_str << j;
				eigval_str << eigvals[j];

				ValueType errRel = nrmVector("host_iram eigval["+j_str.str()+"]:"+eigval_str.str()+" "+path_def_pos[i], y1, eigvec);
				CPPUNIT_ASSERT( errRel < 1.0e-2 );

			}
		}
	}


	void test_device_iram(){
		for(size_t i=0; i<path_def_pos.size(); i++){
			size_t k = 4;

			size_t n = host_mat_def_pos[i].num_rows;
			size_t m = host_mat_def_pos[i].num_cols;
			DeviceMatrix_array2d eigvects;
			HostMatrix_array2d A2d;
			DeviceVector_array1d eigvals;
			HostVector_array1d y1, eigvec(m);

			cusp::krylov::implicitly_restarted_arnoldi(dev_mat_def_pos[i],\
					eigvals, eigvects, k, 0);


			cusp::convert(dev_mat_def_pos[i], A2d);
			for(size_t j=0; j<eigvals.size(); j++){
				thrust::copy(eigvects.values.begin()+ j*n, eigvects.values.begin()+ (j+1)*n,eigvec.begin());
				cuspla::gemv(A2d, eigvec, y1, false);
				cusp::blas::scal(eigvec, (ValueType)eigvals[j]);

				std::stringstream j_str, eigval_str;
				j_str << j;
				eigval_str << eigvals[j];

				ValueType errRel = nrmVector("host_iram eigval["+j_str.str()+"]:"+eigval_str.str()+" "+path_def_pos[i], y1, eigvec);
				CPPUNIT_ASSERT( errRel < 1.0e-2 );

			}
		}
	}




	template <typename Array1d>
	ValueType nrmVector(std::string title, Array1d& A, Array1d& A2){
		ValueType nrmA = cusp::blas::nrm2(A);
		ValueType nrmA2 = cusp::blas::nrm2(A2);
		// Calculates the difference and overwrite the matrix C
		cusp::blas::axpy(A, A2, ValueType(-1));
		ValueType nrmDiff = cusp::blas::nrm2(A2);



		ValueType errRel = ValueType(0);
		if(nrmA==ValueType(0))
			errRel = ValueType(1.0e-30);
		else
			errRel = nrmDiff/nrmA;

#ifdef VERBOSE
#ifndef VVERBOSE
		if(errRel != errRel || errRel >= 1.0e-2){ // Checks if error is nan
#endif VVERBOSE

			std::cout << title << ": AbsoluteErr=" << nrmDiff <<\
					" RelativeErr=" << errRel << "\n" << std::endl;
#ifndef VVERBOSE
		}
#endif VVERBOSE
#endif


		return errRel;
	}



};





CPPUNIT_TEST_SUITE_REGISTRATION( ArnoldiTestCase );

int main(int argc, char** argv)
{

	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest( registry.makeTest() );
	runner.run();
	return 0;

}



