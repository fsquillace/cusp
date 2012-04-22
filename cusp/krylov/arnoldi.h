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
void arnoldi(Matrix& A, Array2d& H, Array2d& V,\
		Array1d& f, const size_t start, const size_t m)
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
        ValueType beta = ValueType(1) / cusp::blas::nrm2(V_[0]);
        cusp::blas::scal(V_[0], beta);

        // Remember that w = f the residual
        // w = A*V(:, 0)
        cusp::multiply(A, V_[0], f);

        beta = ValueType(1) / cusp::blas::nrm2(V_[0]);
        cusp::blas::scal(V_[0], beta);

        // H(0, 0) = V(:, 0)*w
        H(0,0) = cusp::blas::dot(V_[0], f);
        //f = w-V(:,0)*H(0,0)
        cusp::blas::axpy(V_[0], f, -H(0,0));

        // Perform one step of iterative refinement
        // to correct any orthogonality problems
        ValueType alpha = cusp::blas::dot(V_[0], f);
        cusp::blas::axpy(V_[0], f, -alpha);

        H(0,0) = H(0,0) + alpha;


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
void shifted_qr(Array2d& H, Array2d& V, Array1d& mu, size_t p,\
		typename Array2d::value_type& sigma){
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
		Array2d& eigvects, size_t k, size_t m,\
		double tol, size_t maxit,
		cusp::column_major){

    /*
     * Compute the Implicitly Restarted Arnoldi Method
     * It works only if array2d is defined column_major
     *
     */

    typedef typename Array2d::memory_space MemorySpace;
    typedef typename Array2d::value_type   ValueType;

    size_t N = A.num_rows;
    size_t iter = 0;
    size_t kconv; // number converged
    timer t;

    // Trick to get faster convergence.  The tail end (closest to cut off
    // of the sort) will not converge as fast as the leading end of the
    // "wanted" Ritz values.
    // In this way we can also modify k
    // according to the strategy used
    size_t ksave = k;
    size_t psave = m-k;
    k = std::min(N-1,k+3);
    size_t p = m-k; // to default p = k+1


    // Calculate the machine precision
    ValueType machEps = 1.0f;
    do {
       machEps /= 2.0f;
       // If next epsilon yields 1, then break, because current
       // epsilon is the machine epsilon.
    }
    while ((float)(1.0 + (machEps/2.0)) != 1.0);


    Array2d H(m, m, ValueType(0));
    Array2d V(N, m);
    Array1d f(N, ValueType(0));
    Array2d H_(m,m);

    cusp::array1d<int, MemorySpace> perm(m);
    eigvals.resize(m);
    Array2d eigvects_H(m,m);

    arnoldi(A, H, V, f, 0, k);



    do{

        arnoldi(A, H, V, f, k-1, m);

        cusp::copy(H, H_);
        cuspla::geev(H_, eigvals, eigvects_H);

        // Sort in ascending order
        // eigvals and perm will be modified
        thrust::sequence(perm.begin(), perm.end());
        thrust::sort_by_key(eigvals.begin(), eigvals.end(), perm.begin());

        // We need to normalize among the columns of eigvects_H
        for(size_t j=0; j<m; j++){
        	// alpha=1/eigvects(:,j)
        	ValueType alpha = ValueType(1) / cusp::blas::nrm2(eigvects_H.values.begin()+ j*m,\
        			eigvects_H.values.begin()+ (j+1)*m);

            thrust::for_each(eigvects_H.values.begin()+ j*m, eigvects_H.values.begin()+ (j+1)*m,\
            		cuspla::mul_const<ValueType>(alpha));

//            printf("norm(eigvect[%d])=%f\n", j,cusp::blas::nrm2(eigvects_H.values.begin()+ j*m,\
//        			eigvects_H.values.begin()+ (j+1)*m));
        }


        kconv = nconv(eigvals, eigvects_H, perm, f, k, tol, machEps*cusp::blas::nrm2(H.values));

        if(kconv >= ksave || iter >= maxit){
            break;
        }


        // If some ritz values have converged then
        // adjust k and p to move the "boundary"
        // of the filter cutoff.
        if(kconv > 0){
             size_t kk = ksave + 3 + kconv;
             p = std::max(size_t(psave/3), m-kk);
             k = m - p;
        }



        ValueType sigma;
        shifted_qr(H, V, eigvals, p-kconv, sigma);

        //f(k-1) = V(:, k)*beta + f(m-1)*sigma(k-1)
        ValueType beta = H(k, k-1);
        cusp::blas::axpby(V.values.begin()+N*(k), \
                V.values.begin()+N*(k+1), f.begin(), \
                f.begin(), beta, sigma);


        iter++;

    }while(true);

    float t_iter = t.seconds_elapsed();
    printf("t_iter=%f\n", t_iter/iter);

    // Shrinks eigvals to the first k elements
    thrust::copy(eigvals.begin()+psave, eigvals.end(), eigvals.begin());
    eigvals.resize(ksave);


    // This below takes a lot of time!
    Array2d eigvects_tmp(m, ksave);
    for(size_t j=0; j<ksave; j++){
        thrust::copy(eigvects_H.values.begin()+m*perm[m-ksave +j],\
        		eigvects_H.values.begin()+m*perm[m-ksave +j]+m,\
        		eigvects_tmp.values.begin()+m*j);
    }

    cuspla::gemm(V, eigvects_tmp, eigvects, ValueType(1));
    eigvects.resize(N, ksave);

    // Normalize eigvects
    for(size_t j=0; j<ksave; j++){
    	// alpha=1/eigvects(:,j)
    	ValueType alpha = ValueType(1) / cusp::blas::nrm2(eigvects.values.begin()+ j*N,\
    			eigvects.values.begin()+ (j+1)*N);

        thrust::for_each(eigvects.values.begin()+ j*N, eigvects.values.begin()+ (j+1)*N,\
        		cuspla::mul_const<ValueType>(alpha));
    }

}

template <typename Array2d, typename Array1d, \
typename Array1dInt, typename ValueType>
size_t nconv(Array1d& eigvals, Array2d& eigvects,\
		Array1dInt& perm, Array1d& f, size_t k, \
		double tol, ValueType eps){
    // the number of eigvals is equal to the number k (but not m) of required eigvals.
    // eigvals is already ordered, eigvects needs the perm vector
    // eigvals (mx1)
    // eigvect (mxm)
    // perm (mx1)

    size_t nconv = 0;


    size_t m = eigvects.num_rows;
    float err_tot = 0.0;

    //The while loop counts the number of converged ritz values.
    for(size_t i=m-1; i>=m-k; i--){
        const ValueType err=\
        		cusp::blas::nrm2(f)*std::abs(eigvects(m-1, perm[i]));
        err_tot = err_tot + err;

        if(err <= std::max(eps, ValueType(tol)*std::abs(eigvals[i])))
            nconv++;
        else
        	break;
    }


    // TODO include the ritzests values such as in eigs.m and the other stop criterion

#ifdef VERBOSE
    printf("nconv=%d err_avg=%f\n",nconv, err_tot/k);
#endif

    return nconv;
}


// Entry point
template <typename Matrix, typename Array2d, typename Array1d>
void iram(Matrix& A, Array1d& eigvals, Array2d& eigvects,\
		size_t k=0, size_t m=0, double tol=1e-6,\
		size_t maxit=0){
    /*
     * Input:
     * k - Number of desired eigenvalues.
     *     If k==0 or not defined to default k=min(n, 6).
     *
     * m - Krylov basis dimension.
     *     If m==0 or not defined to default
     *     m=min(max(2*k+1, 20), n).
     *
     * maxit - Maximum number of iterations.
     *         If maxit==0 or not defined to default
     *         maxit=max(300, 2*n/p).
    */

	typedef typename Array2d::value_type ValueType;

    if(A.num_rows!=A.num_cols){
        printf("Error: Matrix A must be squared\n");
        return;
    }


    if(k==0)
        k=std::min(A.num_rows, size_t(6));

    if(m==0){
    	m = std::min(std::max(2*k+1,size_t(20)),A.num_rows);
    }

    if(maxit==0)
    	maxit = std::max(size_t(300),2*A.num_rows/m);

    // A is the matrix of all zeros
    if(A.num_entries==0){
        eigvals.resize(k);
        eigvects.resize(A.num_rows, k);
        thrust::fill(eigvals.begin(), eigvals.end(), ValueType(0));
        // Adds 1 of each element of the diagonal
        thrust::counting_iterator<int> stencil (0);
        thrust::transform_if(\
        		eigvects.values.begin(), \
        		eigvects.values.end(), \
                stencil, \
                eigvects.values.begin(), \
                cuspla::assigns<ValueType>(ValueType(1)), \
                cuspla::in_diagonal(\
                		eigvects.num_rows,\
                		eigvects.num_cols));
    }

    if(k<0 || k > A.num_rows-1){
        printf("Error: k must be an integer between 0 and n.\n");
        return;
    }
    else if(m > A.num_rows-1){
        printf("Error: m must be lesser or equals than n.\n");
        return;
    }

	return iram(A, eigvals, eigvects, k, m,\
			tol, maxit,\
			typename Array2d::orientation());
}


} // end namespace krylov
} // end namespace cusp

