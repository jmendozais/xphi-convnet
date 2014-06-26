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

#define IDX2(i, j, ni, nj) (i*nj + j)
//#define IDX(i,j,k, ni, nj, nk) (i*(nj*nk) + j*nk + k)
#define IDX4(i,j,k,p, ni, nj, nk, np) (i*(nj*nk*np) + j*(nk*np) + k*np + p)

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
	for (int i = 0; i < numImgs; ++i)
		for (int j = 0; j < numFilters; ++j)
			/* Convolution */
			for(int x = 0; x < outRows; ++ x)
				for(int y = 0; y < outCols; ++ y) {
					xp = x * stride;
					yp = y * stride;
					res = 0;
					for (int ch = 0; ch < numChannels; ++ch)
						for (int fi = 0; fi < filterSize; ++fi)
							for (int fj = 0; fj < filterSize; ++fj) {
								res += filters[IDX4(j, ch, fi, fj, numFilters,
										numChannels, filterSize, filterSize)]
										* images[IDX4(i, ch, xp+fi, yp+fj, numImgs,
												numChannels, imgRows, imgCols)];
							}
					output[IDX4(i, j, x, y, numImgs, numFilters, outRows, outCols)] = res;
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
	int xp, yp;
	float res;
	for(int i = 0; i < numImgs; ++ i)
		for(int j = 0; j < numChannels; ++ j) {
			for(int x = 0; x < outRows; ++ x)
				for(int y = 0; y < outCols; ++ y) {
					xp = x * stride;
					yp = y * stride;
					res = pooler.base();
					for(int m = 0; m < filterSize; ++ m)
						for(int n = 0; n < filterSize; ++n)
							res = pooler(res, images[IDX4(i, j, xp + m, yp + n, numImgs, numChannels, imgRows, imgCols)]);
					output[IDX4(i, j, x, y, numImgs, numChannels, outRows, outCols)] = pooler.output(res, filterPixels);
				}
		}
}
void fullyConnected(float *images, int numImgs, int numPixels, float *weights, float *outputs, int outSize) {
	float res;
	for(int i = 0; i < numImgs; ++ i)
		for(int j = 0; j < numPixels; ++ j)
			for(int k = 0; k < outSize; ++ k) {
				res = 0;
				for(int x = 0; x < numPixels; ++ x)
					res += weights[IDX2(k, j, outSize, numPixels)] * images[IDX2(i, j, numImgs, numPixels)];
				outputs[IDX2(i, j, numImgs, numPixels)] = res;
			}
}
#endif /* UTIL_H_ */
