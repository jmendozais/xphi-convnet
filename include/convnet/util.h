/*
 * util.h
 *
 *  Created on: Jun 15, 2014
 *      Author: mac
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <assert.h>
#include <algorithm>
#include "omp.h"

#define IDX2(i, j, ni, nj) ((i)*(nj) + (j))
//#define IDX(i,j,k, ni, nj, nk) (i*(nj*nk) + j*nk + k)
#define IDX4(i,j,k,p, ni, nj, nk, np) ((i)*(nj)*(nk)*(np) + (j)*(nk)*(np) + (k)*(np) + (p))

inline int idx(int i,int j,int k,int p,int ni,int nj,int nk,int np) {
	return ((i)*(nj)*(nk)*(np) + (j)*(nk)*(np) + (k)*(np) + (p));
}
/**
 * images : holds an minibatch of images. shape ( minibsize, cols, rows, channels )
 * nimages, imagerows, imagecols, nchannels : image params
 * filters : holds a set of filters in a arrays of shape(fside, fside, num of featuremaps or channels)
 * output : holds the output array shape(imgRows-fside+1/stride,imgCols/stride,channels)
 */
void conv(float *images, const int numImgs, const int numChannels,
		const int imgRows, const int imgCols, float* filters,
		const int numFilters, const int filterSize, float *output, const int outRows,
		const int outCols, const int stride) {
	assert(filterSize % 2 == 1);
	assert((imgCols - filterSize)%stride == 0);
	assert((imgRows - filterSize)%stride == 0);
	assert((imgCols - filterSize)/stride + 1 == outCols);
	assert((imgRows - filterSize)/stride + 1 == outRows);
	int xp, yp;
	float res;
	int i, j, x, y, ch, fi, fj, idx1;
	// TODO: replace this by omp collapse
	#pragma omp parallel for private(i, j, xp, yp, x, y, ch, fi, fj, res, idx1)
	for (idx1= 0; idx1 < numImgs * numFilters; ++idx1) {
		i = idx1/numFilters;
		j = idx1%numFilters;
			/* Convolution */
			for(x = 0; x < outRows; ++ x)
				for(y = 0; y < outCols; ++ y) {
					xp = x * stride;
					yp = y * stride;
					res = 0;
					for (ch = 0; ch < numChannels; ++ch)
						for (fi = 0; fi < filterSize; ++fi)
							for (fj = 0; fj < filterSize; ++fj) {
								res += filters[IDX4(j, ch, fi, fj, numFilters,
										numChannels, filterSize, filterSize)]
										* images[IDX4(i, ch, xp+fi, yp+fj, numImgs,
												numChannels, imgRows, imgCols)];
							}
					output[idx(i, j, x, y, numImgs, numFilters, outRows, outCols)] = res;
				}
	}
}

class MaxPooler {
public:
	inline float operator()(float a, float b) {
		return std::max(a, b);
	}
	inline float output(float total, float filterSize) {
		return total;
	}
	inline float base() {
		return -1e100;
	}
};

// TODO: Evaluate if is better set the output units in the outer loop ( see cuda conv net implementation )
template<class Pooler>
void batchPool(float *images, int numImgs, int numChannels, int imgRows, int imgCols, int stride, int filterSize, float *output, int outRows, int outCols, Pooler pooler) {
	assert((imgCols - filterSize)%stride == 0);
	assert((imgRows - filterSize)%stride == 0);
	assert((imgCols - filterSize)/stride + 1 == outCols);
	assert((imgRows - filterSize)/stride + 1 == outRows);

	int filterPixels = filterSize * filterSize;
	int xp, yp, i, j, x, y, m, n, idx1;
	float res;
	#pragma omp parallel for private(i, j, x, y, m, n, res, xp, yp, idx1)
	for (idx1= 0; idx1 < numImgs * numChannels; ++idx1) {
			i = idx1/numChannels;
			j = idx1%numChannels;
			for(x = 0; x < outRows; ++ x)
				for(y = 0; y < outCols; ++ y) {
					xp = x * stride;
					yp = y * stride;
					res = pooler.base();
					for(m= 0; m < filterSize; ++ m)
						for(n = 0; n < filterSize; ++n)
							res = pooler(res, images[IDX4(i, j, xp + m, yp + n, numImgs, numChannels, imgRows, imgCols)]);
					output[IDX4(i, j, x, y, numImgs, numChannels, outRows, outCols)] = pooler.output(res, filterPixels);
				}
	}
}
void fullyConnected(float *images, int numImgs, int numPixels, float *weights, float *outputs, int outSize) {
	float res;
	int i, j, k, idx1;
	#pragma omp parallel for private(i, j, k, res, idx1)
	for (idx1= 0; idx1 < numImgs * outSize; ++idx1) {
			i = idx1/outSize;
			j = idx1%outSize;
			res = 0;
			for(k = 0; k < numPixels; ++ k)
				res += weights[IDX2(j, k, outSize, numPixels)] * images[IDX2(i, k, numImgs, numPixels)];
			outputs[IDX2(i, j, numImgs, outSize)] = res;
	}
}
#endif /* UTIL_H_ */
