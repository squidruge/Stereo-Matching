#pragma once

template<typename T>
__global__ void buildList(T** arrs, size_t size, size_t tot_list) {
	size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (Idx >= tot_list) return;

	arrs[Idx] = (T*)malloc(sizeof(T) * size);
	memset(arrs[Idx], 0, sizeof(T) * size);
}

template<typename T>
__global__ void copyList(T** dsts, const T* const* srcs, size_t size, size_t tot_list) {
	size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (Idx >= tot_list) return;

	memcpy(dsts[Idx], srcs[Idx], sizeof(T) * size);
}

template<typename T>
__global__ void clearList(T** arrs, size_t tot_list) {
	size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (Idx >= tot_list) return;

	free(arrs[Idx]);
}