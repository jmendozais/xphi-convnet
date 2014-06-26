/*
 * convnet.h
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#ifndef CONVNET_H_
#define CONVNET_H_
#include <map>
#include <string>
#include "assert.h"
#include "layer.h"
class ConvNet {

private:
	std::vector<Layer*> _layers;
	std::vector<DataLayer*> _dataLayers;
	int _miniBatchSize;
public:
	ConvNet();
	/*
	 *
	 */
	ConvNet(std::vector<std::map<std::string, std::string> > params, int miniBatchSize);
	virtual ~ConvNet();
	Layer initLayer(std::string layerType, std::map<std::string, int> params);
	void fprop(std::vector<Matrix *> input);
	int getMiniBatchSize() const;
	void setMiniBatchSize(int miniBatchSize);
};

#endif /* CONVNET_H_ */
