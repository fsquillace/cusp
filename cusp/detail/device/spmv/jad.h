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

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/device_ptr.h>

namespace cusp
{
namespace detail
{
namespace device
{

////////////////////////////////////////////////////////////////////////
// JAD SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_jad_device
//   Straightforward translation of standard JAD SpMV to CUDA
//   where each thread computes y[i] = A[i,:] * x
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_jad_tex_device
//   Same as spmv_jad_device, except x is accessed via texture cache.
//

template <bool UseCache, unsigned int BLOCK_SIZE,
          typename IndexType,
          typename ValueType>
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void
spmv_jad_scalar_kernel(const IndexType num_rows,
                       const IndexType num_cols,
                       const IndexType num_jagged_diagonals,
                       const IndexType * __restrict__ Ad,
                       const IndexType * __restrict__ Aj,
                       const ValueType * __restrict__ Ax,
                       const ValueType * __restrict__ x,
                             ValueType * __restrict__ y)
{

    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    extern __shared__ IndexType array[];
    volatile IndexType* Ad_s = (IndexType*)array;



    for(int i=threadIdx.x; i<= num_jagged_diagonals; i+=BLOCK_SIZE)
        Ad_s[i] = Ad[i];

    __syncthreads();



    for(register IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        ValueType sum = 0;


        IndexType dia_start = Ad_s[0];
        IndexType dia_stop = Ad_s[1];
        IndexType dia_lenght = dia_stop - dia_start;
        register IndexType dia = 0;
        while((dia_lenght >row) && dia < num_jagged_diagonals){


            sum += Ax[dia_start + row] * fetch_x<UseCache>(Aj[dia_start + row], x);


            dia++;
            if(dia<num_jagged_diagonals){
            dia_start = dia_stop;
            dia_stop = Ad_s[dia+1];
            dia_lenght = dia_stop - dia_start;
            }
        }

        y[row] = sum;
    }
}

template <bool UseCache, unsigned int BLOCK_SIZE,
          typename IndexType,
          typename ValueType>
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void
spmv_jad_scalar_kernel_no_shared(const IndexType num_rows,
                       const IndexType num_cols,
                       const IndexType num_jagged_diagonals,
                       const IndexType * __restrict__ Ad,
                       const IndexType * __restrict__ Aj,
                       const ValueType * __restrict__ Ax,
                       const ValueType * __restrict__ x,
                             ValueType * __restrict__ y)
{

    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;


    for(register IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        ValueType sum = 0;


        IndexType dia_start = Ad[0];
        IndexType dia_stop = Ad[1];
        IndexType dia_lenght = dia_stop - dia_start;
        register IndexType dia = 0;
        while((dia_lenght >row) && dia < num_jagged_diagonals){


            sum += Ax[dia_start + row] * fetch_x<UseCache>(Aj[dia_start + row], x);


            dia++;
            if(dia<num_jagged_diagonals){
            dia_start = dia_stop;
            dia_stop = Ad[dia+1];
            dia_lenght = dia_stop - dia_start;
            }
        }

        y[row] = sum;
    }

//    if(thread_id==0){
//        for(int i=0; i<num_cols; i++)
//            CUPRINTF("x[%d]:%f\n",i,x[i]);
//
//        for(int i=0; i<num_rows; i++)
//            CUPRINTF("y[%d]:%f\n",i,y[i]);
//
//        for(int i=0; i<num_jagged_diagonals+1; i++)
//            CUPRINTF("Ad[%d]:%d\n",i,Ad_s[i]);
//
////        for(int i=0; i<num_entries; i++){
////            CUPRINTF("Aj[%d]:%d\n",i,Aj[i]);
////            CUPRINTF("Ax[%d]:%f\n",i,Ax[i]);
////        }
//    }

}

//template <typename IndexType,
//          typename ValueType>
//__global__ void
//spmv_jad_permute_kernel(const IndexType num_rows,
//                       const IndexType num_cols,
//                       const IndexType num_jagged_diagonals,
//                       const IndexType * permutations,
//                             ValueType * y)
//{
//
//    IndexType i = 0;
//    IndexType counter=0;
//    ValueType tmp = y[i], tmp2;
//
//    while(counter<num_rows){
//        const IndexType index = permutations[i];
//        tmp2 = y[index];
//        y[index] = tmp;
//        tmp = tmp2;
//        i = index;
//
//
//        counter++;
//    }
//
//    for(int i=0; i<num_rows; i++)
//        CUPRINTF("permutations[%d]:%d\n",i,permutations[i]);
//    for(int i=0; i<num_rows; i++)
//        CUPRINTF("y_new[%d]:%f\n",i,y[i]);
//
//
//}

template <typename IndexType,
          typename ValueType>
__global__ void
spmv_jad_permute_kernel(
                        const IndexType num_rows,
                       const IndexType num_cols,
                       const IndexType num_jagged_diagonals,
                       const IndexType * permutations,
                       const ValueType * y,
                             ValueType * y_new)
{


    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        y_new[permutations[row]] = y[row];
    }

//    if(thread_id==0){
//        for(int i=0; i<num_rows; i++)
//            CUPRINTF("permutations[%d]:%d\n",i,permutations[i]);
//        for(int i=0; i<num_rows; i++)
//            CUPRINTF("y_new[%d]:%f\n",i,y_new[i]);
//    }

}
template <bool UseCache,
          typename Matrix,
          typename ValueType>
void __spmv_jad(const Matrix&    A,
                       const ValueType* x,
                             ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    const size_t BLOCK_SIZE = 256;
    const size_t dia_offsets_byte = (A.num_jagged_diagonals+1)*sizeof(IndexType);
//    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(spmv_jad_scalar_kernel<UseCache, BLOCK_SIZE, IndexType, ValueType>, BLOCK_SIZE, dia_offsets_byte);
    const size_t NUM_BLOCKS = DIVIDE_INTO(A.num_rows, BLOCK_SIZE); //std::min(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));
//    printf("NUM_BLOCKS:%d MAX_BLOCKS:%d NEEDED_BLOCKS:%d\n",NUM_BLOCKS, MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));
    cusp::array1d<ValueType, cusp::device_memory> y_new(A.num_rows);
    const size_t SHARED_MEM = 16384;

    const bool use_shared = (dia_offsets_byte<=SHARED_MEM)?true:false;
//    printf("num_jagged_diagonals:%d num_rows:%d num_cols:%d\n", A.num_jagged_diagonals, A.num_rows, A.num_cols);


    if (UseCache)
        bind_x(x);


    if(use_shared)
        spmv_jad_scalar_kernel<UseCache,BLOCK_SIZE,IndexType,ValueType> <<<NUM_BLOCKS, BLOCK_SIZE,
            dia_offsets_byte>>>
            (A.num_rows,
             A.num_cols,
             A.num_jagged_diagonals,
             thrust::raw_pointer_cast(&A.diagonal_offsets[0]),
             thrust::raw_pointer_cast(&A.column_indices[0]),
             thrust::raw_pointer_cast(&A.values[0]),
             x,
             thrust::raw_pointer_cast(&y_new[0]));
    else
        spmv_jad_scalar_kernel_no_shared<UseCache,BLOCK_SIZE,IndexType,ValueType> <<<NUM_BLOCKS, BLOCK_SIZE>>>
            (A.num_rows,
             A.num_cols,
             A.num_jagged_diagonals,
             thrust::raw_pointer_cast(&A.diagonal_offsets[0]),
             thrust::raw_pointer_cast(&A.column_indices[0]),
             thrust::raw_pointer_cast(&A.values[0]),
             x,
             thrust::raw_pointer_cast(&y_new[0]));



    spmv_jad_permute_kernel<IndexType, ValueType><<<NUM_BLOCKS,BLOCK_SIZE>>>
        (A.num_rows,
         A.num_cols,
         A.num_jagged_diagonals,
         thrust::raw_pointer_cast(&A.permutations[0]),
         thrust::raw_pointer_cast(&y_new[0]),
         y);


    if (UseCache)
        unbind_x(x);


}

template <typename Matrix,
          typename ValueType>
void spmv_jad(const Matrix&    A,
                     const ValueType* x,
                           ValueType* y)
{
    __spmv_jad<false>(A, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_jad_tex(const Matrix&    A,
                         const ValueType* x,
                               ValueType* y)
{
    __spmv_jad<true>(A, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

