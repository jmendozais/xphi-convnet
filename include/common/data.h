/*
 * data.h
 *
 *  Created on: Jun 25, 2014
 *      Author: mac
 */

#ifndef DATA_H_
#define DATA_H_

#include <assert.h>
#include <string.h>
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "matrix.h"

#define IDX4(i,j,k,p, ni, nj, nk, np) ((i)*(nj)*(nk)*(np) + (j)*(nk)*(np) + (k)*(np) + (p))
#define two(x) (x)*(x)
#define CIFAR10_LINE 3073
#define IMG_SIZE 32
#define CHANNELS 3
#define PADDING 2
class DataProvider {
private:
	std::vector<char*> buffers;
	std::vector<std::string> _filePahts;
	std::vector<int> _sizes;
	int _size;
	int _numEntries;
	int _miniBatchSize;
public:
	DataProvider(int miniBatchSize) {
		_size = 0;
		_miniBatchSize = miniBatchSize;
	}
	void addData(std::string filename) {
		struct stat results;
		std::cout << filename << std::endl;
		assert(stat(filename.c_str(), &results) == 0);

		std::ifstream fin(filename.c_str(), std::ios::in | std::ios::binary);
		char *buffer = new char[results.st_size];
		fin.read(buffer, results.st_size);
		buffers.push_back(buffer);
		_sizes.push_back(results.st_size);
		_size += results.st_size;
		_numEntries = _size / CIFAR10_LINE;
	}
	std::vector<Matrix*> getMiniBatch(int miniBatch) {
		int numMinibatches = _size / (CIFAR10_LINE * _miniBatchSize);
		assert(miniBatch < numMinibatches);
		int imageSize = IMG_SIZE;
		int paddedImageSize = imageSize + 2 * PADDING;
		int padding = PADDING;
		int channels = CHANNELS;

		char* buffer = buffers[0], *curBuffer;

		std::vector<Matrix*> res(2);
		res[0] = new Matrix(_miniBatchSize, channels * paddedImageSize * paddedImageSize);
		memset(res[0]->getData(), 0, sizeof(float) * _miniBatchSize * channels * paddedImageSize
						* paddedImageSize);
		res[1] = new Matrix(_miniBatchSize, 1);

		for (int i = 0; i < _miniBatchSize; ++i) {

			curBuffer = buffer + (miniBatch*_miniBatchSize + i)*CIFAR10_LINE;
			res[1]->getData()[i] = (float)curBuffer[0];
			for (int j = 0; j < channels; ++j)
				for (int x = 0; x < imageSize; ++x)
					for (int y = 0; y < imageSize; ++y)
						res[0]->getData()[IDX4(i, j, x + padding, y + padding, _miniBatchSize,
								channels, paddedImageSize, paddedImageSize)] = curBuffer[IDX4(i, j, x, y, 1,
										channels, imageSize, imageSize)];
		}
		return res;
	}
	~DataProvider() {
		for (unsigned int i = 0; i < buffers.size(); ++i)
			delete[] buffers[i];
	}

	const std::vector<std::string>& getFilePahts() const {
		return _filePahts;
	}

	void setFilePahts(const std::vector<std::string>& filePahts) {
		_filePahts = filePahts;
	}

	int getMiniBatchSize() const {
		return _miniBatchSize;
	}

	void setMiniBatchSize(int miniBatchSize) {
		_miniBatchSize = miniBatchSize;
	}

	int getSize() const {
		return _size;
	}

	void setSize(int size) {
		_size = size;
	}

	const std::vector<int>& getSizes() const {
		return _sizes;
	}

	void setSizes(const std::vector<int>& sizes) {
		_sizes = sizes;
	}

	const std::vector<char*>& getBuffers() const {
		return buffers;
	}

	void setBuffers(const std::vector<char*>& buffers) {
		this->buffers = buffers;
	}

	int getNumEntries() const {
		return _numEntries;
	}

	void setNumEntries(int numEntries) {
		_numEntries = numEntries;
	}
};

#endif /* DATA_H_ */
