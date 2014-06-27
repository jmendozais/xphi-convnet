/*
 * convnet.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#include "convnet.h"
#include "../common/common.h"
#include <iostream>

ConvNet::ConvNet() {
	// TODO Auto-generated constructor stub

}

ConvNet::~ConvNet() {
	// TODO Auto-generated destructor stub
}

ConvNet::ConvNet(std::vector<std::map<std::string, std::string> > params, int miniBatchSize) {
	_miniBatchSize = miniBatchSize;
	int numLayers = params.size();
	std::map<std::string, int> indexByName;
	std::string name, type;
	for(int i = 0; i < numLayers; ++i) {
		name = params[i]["name"];
		type = params[i]["type"];
		Layer *newLayer = NULL;
		DataLayer *newDataLayer = NULL;
		if( type == "data" ) {
			newDataLayer = new DataLayer(this, params[i]);
			newLayer = newDataLayer;
			_dataLayers.push_back(newDataLayer);
		} else if( type == "conv")
			newLayer = new ConvLayer(this, params[i]);
		else if( type == "pool")
			newLayer = new PoolLayer(this, params[i]);
		else if( type == "fc")
			newLayer = new FCLayer(this, params[i]);
		else if( type == "softmax" )
			newLayer = new SoftMaxLayer(this, params[i]);
		else if( type == "cost.logreg")
			newLayer = new LogRegCostLayer(this, params[i]);

		if(newLayer != NULL) {
			indexByName[name] = _layers.size();
			_layers.push_back(newLayer);
		} else {
			std::cout << "[WARN]: Unknown layer type: " << type << std::endl;
		}
	}
	/* adding backward links */
	std::map<std::string, std::string>::const_iterator it;
	std::map<std::string, int>::const_iterator it2;
	std::vector<std::string> inputs;
	for(int i = 0; i < numLayers; ++i) {
		it = params[i].find("inputs");
		if(it != params[i].end()) {
			inputs = split((*it).second, ",");
			for(int j = 0; j < inputs.size(); ++ j) {
				it2 = indexByName.find(inputs[j]);
				assert(it2 != indexByName.end());
				_layers[i]->addPrev(_layers[(*it2).second]);
			}
		} else {
			assert(params[i]["type"] == "data");
		}
	}
	/* adding forward links */
	for(int i = 0; i < numLayers; ++i) {
		std::vector<Layer*>& prevs = _layers[i]->getPrevs();
		for(int j = 0; j < prevs.size(); ++ j)
			prevs[j]->addNext(_layers[i]);
	}
	for(int i = 0; i < numLayers; ++i) {
		_layers[i]->postInit();
	}
	_checker = 0;
}

Layer ConvNet::initLayer(std::string layerType,
		std::map<std::string, int> params) {
}

void ConvNet::fprop(std::vector<Matrix*> input) {
	assert(_dataLayers.size() == input.size());
	_dataLayers[0]->fprop(input[0]);/*
	for(int i = 0; i < _dataLayers.size(); ++ i)
		_dataLayers[i]->fprop(input[i]);*/
}

int ConvNet::getMiniBatchSize() const {
	return _miniBatchSize;
}

void ConvNet::setMiniBatchSize(int miniBatchSize) {
	_miniBatchSize = miniBatchSize;
}
