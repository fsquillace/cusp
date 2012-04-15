/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/detail/random.h>
#include <cusp/print.h>


#include <cuspla.cu>


#include "../../performance/timer.h"


namespace cusp
{
namespace krylov
{

template <typename Matrix, typename Array2d>
void lanczos(const Matrix& A, Array2d& H, size_t k = 10)
{
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_cols;
	size_t maxiter = std::min(N, k);

    // allocate workspace
	cusp::array1d<ValueType,MemorySpace> v0(N);
    cusp::array1d<ValueType,MemorySpace> v1(N);
	cusp::array1d<ValueType,MemorySpace> w(N);
    
    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::detail::random_reals<ValueType>(N), v1);

	cusp::blas::scal(v1, ValueType(1) / cusp::blas::nrm2(v1));

	Array2d H_(maxiter + 1, maxiter, 0);

	ValueType alpha = 0.0, beta = 0.0;

	size_t j;

	for(j = 0; j < maxiter; j++)
	{
		cusp::multiply(A, v1, w);

		if(j >= 1)
		{
			H_(j - 1, j) = beta;
			cusp::blas::axpy(v0, w, -beta);
		}

		alpha = cusp::blas::dot(w, v1);
		H_(j,j) = alpha;

		cusp::blas::axpy(v1, w, -alpha);

		beta = cusp::blas::nrm2(w);
		H_(j + 1, j) = beta;

		if(beta < 1e-10) break;

		cusp::blas::scal(w, ValueType(1) / beta);				

        // [v0 v1  w] - > [v1  w v0]
        v0.swap(v1);
        v1.swap(w);
	}

	H.resize(j,j);
	for(size_t row = 0; row < j; row++)
		for(size_t col = 0; col < j; col++)
			H(row,col) = H_(row,col);
}

template <typename Matrix, typename Array2d>
void arnoldi(const Matrix& A, Array2d& H, size_t k = 10)
{
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_rows;

	size_t maxiter = std::min(N, k);

	Array2d H_(maxiter + 1, maxiter, 0);

    // allocate workspace of k + 1 vectors
    std::vector< cusp::array1d<ValueType,MemorySpace> > V(maxiter + 1);
    for (size_t i = 0; i < maxiter + 1; i++)
        V[i].resize(N);
	
    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::detail::random_reals<ValueType>(N), V[0]);

    // normalize v0
	cusp::blas::scal(V[0], ValueType(1) / cusp::blas::nrm2(V[0]));	

	size_t j;

	for(j = 0; j < maxiter; j++)
	{
		cusp::multiply(A, V[j], V[j + 1]);

		for(size_t i = 0; i <= j; i++)
		{
			H_(i,j) = cusp::blas::dot(V[i], V[j + 1]);

			cusp::blas::axpy(V[i], V[j + 1], -H_(i,j));
		}

		H_(j+1,j) = cusp::blas::nrm2(V[j + 1]);

		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V[j + 1], ValueType(1) / H_(j+1,j));
	}

	H.resize(j,j);
	for( size_t row = 0; row < j; row++ )
		for( size_t col = 0; col < j; col++ )
			H(row,col) = H_(row,col);
}



template <typename Matrix, typename Array1d, typename Array2d>
void arnoldi(Matrix& A, Array2d& H, Array2d& V, Array1d& f, const size_t start, const size_t m)
{
    /*
     *
     * A (input)  (NxM) sparse matrix
     *
     * H (output) (mxm) Hessenberg matrix
     *
     * V (input/output) (Nxm) Subspace base vectors
     *
     * f (input/output) (N)
     *
     * start (input)
     *
     * stop (input)
     *
     * m (input) subspace dimension
     */

    typedef typename Matrix::value_type   ValueType;

    size_t N = A.num_rows;


    // allocate workspace of m vectors
    std::vector< Array1d > V_(m);
    for (size_t i = 0; i < m; i++)
        V_[i].resize(N);

    // allocate the vector f
    f.resize(N);

    if(start==0){
        // initialize starting vector to random values in [0,1)
        cusp::copy(cusp::detail::random_reals<ValueType>(N), V_[0]);
        // normalize v0
        ValueType nrm = ValueType(1) / cusp::blas::nrm2(V_[0]);
        cusp::blas::scal(V_[0], nrm);

        // Remember that w = f the residual
        // w = A*V(:, 0)
        cusp::multiply(A, V_[0], f);


        // H(0, 0) = V(:, 0)*w
        H(0,0) = cusp::blas::dot(V_[0], f);
        //f = w-V(:,0)*H(0,0)
        cusp::blas::axpy(V_[0], f, -H(0,0));


    }
    else{
        // copy the first j=0...k-1 columns of V
        for(size_t j=0; j<=start; j++){
            thrust::copy(V.values.begin()+N*j, V.values.begin()+N*(j+1), V_[j].begin());
        }

    }

    size_t j;

    for(j = start; j < m-1; j++)
    {

        //beta = ||f||
        H(j+1,j) = cusp::blas::nrm2(f);

        if(H(j+1,j) < 1e-10){
            break;
        }

        // V(:,j+1) = f/beta
        cusp::copy(f, V_[j + 1]);
        cusp::blas::scal(V_[j + 1], ValueType(1) / H(j+1,j));


        cusp::multiply(A, V_[j+1], f);



        for(size_t i = 0; i <= j+1; i++)
        {
            H(i,j+1) = cusp::blas::dot(V_[i], f);
        }

        for(size_t i = 0; i <= j+1; i++)
        {
            cusp::blas::axpy(V_[i], f, -H(i,j+1));
        }


    }

//    H.resize(j+1,j+1);

    V.resize(N, j+1);
    for(size_t l = start; l < j+1; l++)
        thrust::copy(V_[l].begin(), V_[l].end(), V.values.begin()+l*N);


}


template< typename Array2d, typename Array1d>
void shifted_qr(Array2d& H, Array2d& V, Array1d& mu, size_t p, typename Array2d::value_type& sigma){
// TODO optimize shift QR by using directly the Householder reflectors instead of QR
	typedef typename Array2d::value_type   ValueType;

	Array2d Mtmp;
	Array1d Vtmp;


	size_t m = H.num_rows;
	size_t n = H.num_cols;

	// q = e_{m} <- canonical vector
	Array1d q(m, ValueType(0));
	q[m-1] = ValueType(1);

	Array2d Q(m, m);
	Array2d R(std::min(m,n), n);

	for(size_t j=0; j < p; j++){

		// H_ = H - uj*I
		cusp::copy(H, Mtmp);

	    thrust::counting_iterator<int> stencil (0);
	    thrust::transform_if(Mtmp.values.begin(), Mtmp.values.end(), \
	        stencil, \
	        Mtmp.values.begin(), \
	        cuspla::plus_const<ValueType>(-mu[j]), \
	        cuspla::in_diagonal(n,m));


		// Computes QR(H_)
		cuspla::geqrf(Mtmp, Q, R, false); // we don't need R

		// H=Q'*H*Q
		cuspla::gemm(Q,H,Mtmp, ValueType(1),ValueType(0),true,false);
		// all the following products are wrong!!
		cuspla::gemm(Mtmp,Q,H, ValueType(1),ValueType(0),false,false);

		// V = V*Q
		cuspla::gemm(V, Q, Mtmp, ValueType(1));
		cusp::copy(Mtmp, V);

		// q = q'*Q => Q'*q

		cuspla::gemv(Q, q, Vtmp, true);
		cusp::copy(Vtmp, q);

	}

	// s = Q(m-1, k-1)
	sigma = Q(m-1, m-p-1);

}


template <typename Matrix, typename Array2d, typename Array1d>
void iram(Matrix& A, Array1d& eigvals,\
		Array2d& eigvects, size_t k, size_t m, cusp::column_major){
    /*
     * Compute the Implicitly Restarted Arnoldi Method
     * It works only if array2d is defined column_major
     *
     */

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    // TODO Solve double problem in iram
    // TODO Solve round-off error

    // Calculate the machine precision
    ValueType machEps = 1.0f;
    do {
       machEps /= 2.0f;
       // If next epsilon yields 1, then break, because current
       // epsilon is the machine epsilon.
    }
    while ((float)(1.0 + (machEps/2.0)) != 1.0);


    size_t N = A.num_rows;
    size_t p = m-k; // to default p = k+1

    size_t maxiter = 100;
    size_t iter = 0;



    Array2d H(m, m, ValueType(0));
    Array2d V(N, m);
    Array1d f(N, ValueType(0));
    Array2d H_(m,m);

    cusp::array1d<int, MemorySpace> perm(m);
    eigvals.resize(m);
    Array2d eigvects_H(m,m);

    arnoldi(A, H, V, f, 0, k);

    // number converged
    size_t nc;
    timer t;
    do{

        arnoldi(A, H, V, f, k-1, m);

        cusp::copy(H, H_);
        cuspla::geev(H_, eigvals, eigvects_H);

        // Selects the last p elements
        // sort the elements of perm to 0, 1, 2, ...
        thrust::sequence(perm.begin(), perm.end());
        thrust::sort_by_key(eigvals.begin(), eigvals.end(), perm.begin());


        // We need to normalize among the columns of eigvects_H
        for(size_t j=0; j<m; j++){
        	//thrust::copy(eigvects_H.values.begin()+ j*N, eigvects_H.values.begin()+ (j+1)*N, evect.begin());
        	ValueType alpha = ValueType(1) / cusp::blas::nrm2(eigvects_H.values.begin()+ j*m,\
        			eigvects_H.values.begin()+ (j+1)*m);

            thrust::for_each(eigvects_H.values.begin()+ j*m, eigvects_H.values.begin()+ (j+1)*m,\
            		cuspla::mul_const<ValueType>(alpha));

//            printf("norm(eigvect[%d])=%f\n", j,cusp::blas::nrm2(eigvects_H.values.begin()+ j*m,\
//        			eigvects_H.values.begin()+ (j+1)*m));
        }


        nc = nconv(eigvals, eigvects_H, perm, f, k, ValueType(1.0e-6), machEps*cusp::blas::nrm2(H.values));
        if(nc == k || iter >= maxiter)
            break;

        // Apply increased k
//        nc = std::min(nc, p/2);
//        if(p-nc==ValueType(0))
//            break;

        ValueType sigma;
        shifted_qr(H, V, eigvals, p-nc, sigma);

        //f(k-1) = V(:, k)*beta + f(m-1)*sigma(k-1)
        ValueType beta = H(k, k-1);


        cusp::blas::axpby(V.values.begin()+N*(k), \
                V.values.begin()+N*(k+1), f.begin(), \
                f.begin(), beta, sigma);


        iter++;

    }while(true);

    cudaThreadSynchronize();
    float t_iter = t.seconds_elapsed();
    printf("t_iter=%f\n", t_iter/iter);

    // Shrinks eigvals to the first k elements
    thrust::copy(eigvals.begin()+p, eigvals.end(), eigvals.begin());
    eigvals.resize(k);


    // This below takes a lot of time!
    Array2d eigvects_tmp(m, k);
    for(size_t j=0; j<k; j++){
        thrust::copy(eigvects_H.values.begin()+m*perm[m-k +j],\
        		eigvects_H.values.begin()+m*perm[m-k +j]+m, eigvects_tmp.values.begin()+m*j);
    }
    cuspla::gemm(V, eigvects_tmp, eigvects, ValueType(1));
    eigvects.resize(N, k);

}

template <typename Array2d, typename Array1d, typename Array1dInt, typename ValueType>
size_t nconv(Array1d& eigvals, Array2d& eigvects, Array1dInt& perm, Array1d& f, size_t k, ValueType tol, ValueType eps){
    // the number of eigvals is equal to the number k (but not m) of required eigvals.
    // eigvals is already ordered, eigvects needs the perm vector
    // eigvals (mx1)
    // eigvect (mxm)
    // perm (mx1)

    size_t nconv = 0;


    size_t m = eigvects.num_rows;
    float err_tot = 0.0;


    for(size_t i=m-1; i>=m-k; i--){
        const ValueType err = cusp::blas::nrm2(f)*std::abs(eigvects(m-1, perm[i]));
        err_tot = err_tot + err;

        if(err < std::max(eps, tol*std::abs(eigvals[i])))
            nconv++;
    }

#ifdef VERBOSE
    printf("nconv=%d err_avg=%f\n",nconv, err_tot/k);
#endif

    return nconv;
}


// Entry point
template <typename Matrix, typename Array2d, typename Array1d>
void iram(Matrix& A, Array1d& eigvals, Array2d& eigvects, size_t k, size_t m){
	/*
     * If m=0 the value will be changed in 2*k+1
	 */

    // Set the default value
    if(m==0) m = std::min(2*k +1, A.num_rows);

	return iram(A, eigvals, eigvects, k, m, typename Array2d::orientation());
}


} // end namespace krylov
} // end namespace cusp

