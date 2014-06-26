/*
 * layer.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#include "assert.h"
#include "layer.h"
#include <iostream>
#include "../common/common.h"
#include "util.h"
#include "convnet.h"
Layer::Layer() {
	// TODO Auto-generated constructor stub

}

Layer::Layer(ConvNet* convNet, std::map<std::string, std::string> params) {
	_convNet = convNet;
	_channels = str2int(params["channels"]);
	_rows = str2int(params["rows"]);
	_cols = str2int(params["cols"]);
	_outputs = new Matrix(_convNet->getMiniBatchSize(), _channels * _rows * _cols);
}

Layer::~Layer() {
	delete _outputs;
}

void Layer::addPrev(Layer* prev) {
	_prevs.push_back(prev);
}
void Layer::addNext(Layer* next) {
	_nexts.push_back(next);
}
std::vector<Layer*>& Layer::getPrevs() {
	return _prevs;
}

int Layer::getChannels() const {
	return _channels;
}

void Layer::setChannels(int channels) {
	_channels = channels;
}

int Layer::getCols() const {
	return _cols;
}

void Layer::setCols(int cols) {
	_cols = cols;
}

int Layer::getRows() const {
	return _rows;
}

void Layer::setRows(int rows) {
	_rows = rows;
}

std::vector<Layer*>& Layer::getNexts() {
	return _nexts;
}

ConvLayer::ConvLayer(ConvNet* convNet, std::map<std::string, std::string> params) : WeightLayer(convNet, params) {
	_numFilters = str2int(params["channels"]);
	_filterSize = str2int(params["filterSize"]);
	_stride = str2int(params["stride"]);
}

void ConvLayer::postInit() {
	assert(_prevs.size() > 0);
	Layer * prev = _prevs[0];
	Matrix* weigths = new Matrix(_numFilters, prev->getChannels() * _filterSize * _filterSize);
	_weightsList.push_back(weigths);
}

PoolLayer::PoolLayer(ConvNet* convNet, std::map<std::string, std::string> params) : Layer(convNet, params) {
	_stride = str2int(params["stride"]);
	_filterSize = str2int(params["sizeX"]);
}

FCLayer::FCLayer(ConvNet* convNet, std::map<std::string, std::string> params) : WeightLayer(convNet, params) {
	_outputSize = str2int(params["outputs"]);
}

void FCLayer::postInit() {
	assert(_prevs.size() > 0);
	Layer * prev = _prevs[0];
	Matrix* weigths = new Matrix(_outputSize, prev->getChannels() * prev->getRows() * prev->getCols());
	_weightsList.push_back(weigths);
}

DataLayer::DataLayer(ConvNet* convNet, std::map<std::string, std::string> params) : Layer(convNet, params) {
}

SoftMaxLayer::SoftMaxLayer(ConvNet* convNet, std::map<std::string, std::string> params) : Layer(convNet, params) {
}

LogRegCostLayer::LogRegCostLayer(ConvNet* convNet, std::map<std::string, std::string> params) : Layer(convNet, params) {
}

WeightLayer::WeightLayer() {
}

WeightLayer::WeightLayer(ConvNet* convNet, std::map<std::string, std::string> params) : Layer(convNet, params) {
}

void DataLayer::fprop(Matrix* input) {
	assert(input != NULL);
	//delete _outputs;
	_outputs = input;
	fpropNext();
}

void Layer::fpropNext() {
	for(int i = 0; i < _nexts.size(); ++i)
		_nexts[i]->fprop(_outputs);
}

void ConvLayer::fprop(Matrix* input) {
	assert(input != NULL);
	assert(getPrevs().size() > 0);

	Layer* prev = getPrevs()[0];
	conv(input->getData(), _convNet->getMiniBatchSize(), prev->getChannels(), prev->getRows(), prev->getCols(), _weightsList[0]->getData(), _numFilters, _filterSize, _outputs->getData(), _rows, _cols, _stride);
	fpropNext();
}

void PoolLayer::fprop(Matrix* input) {
	assert(input != NULL);
	assert(getPrevs().size() > 0);
	Layer* prev = getPrevs()[0];

	batchPool(input->getData(), _convNet->getMiniBatchSize(), prev->getChannels(), prev->getRows(), prev->getCols(), _stride, _filterSize, _outputs->getData(), _rows, _cols, MaxPooler());
	fpropNext();

}

void FCLayer::fprop(Matrix* input) {
	assert(input != NULL);
	assert(getPrevs().size() > 0);
	Layer* prev = getPrevs()[0];
	fullyConnected(input->getData(), _convNet->getMiniBatchSize(), prev->getChannels() * prev->getCols() * prev->getRows(), _weightsList[0]->getData(), _outputs->getData(), _outputSize);
	fpropNext();
}
