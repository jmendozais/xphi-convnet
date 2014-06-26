/*
 * layer.h
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <map>
#include <string>
#include <vector>

#include "../common/matrix.h"
class ConvNet;

class Layer {
protected:
	ConvNet* _convNet;
	std::vector<Layer*> _prevs;
	std::vector<Layer*> _nexts;
	std::vector<Matrix*> inputs;
	int _rows;
	int _cols;
	int _channels;
	Matrix* _outputs;
public:
	Layer();
	Layer(ConvNet* convNet, std::map<std::string, std::string> params);
	virtual ~Layer();
	virtual void fprop(Matrix* input) {}
	void fpropNext();
	void addPrev(Layer* prev);
	void addNext(Layer* next);
	std::vector<Layer*>& getPrevs();
	std::vector<Layer*>& getNexts();

	virtual void postInit() {}

	int getChannels() const;
	void setChannels(int channels);
	int getCols() const;
	void setCols(int cols);
	int getRows() const;
	void setRows(int rows);
};

class WeightLayer : public Layer {
protected:
	std::vector<Matrix*> _weightsList;
	std::vector<Matrix*> _biasesList;
public:
	WeightLayer();
	WeightLayer(ConvNet* convNet, std::map<std::string, std::string> params);
	virtual void fprop(Matrix* input) {}
	virtual void postInit() {}
};

class ConvLayer: public WeightLayer {
private:
	int _numFilters;
	int _filterSize;
	int _stride;

public:
	ConvLayer(ConvNet* convNet, std::map<std::string, std::string> params);
	void fprop(Matrix* input);
	void postInit();
};

class PoolLayer: public Layer {
	int _filterSize;
	int _stride;
public:
	PoolLayer(ConvNet* convNet, std::map<std::string, std::string> params);
	void fprop(Matrix* input);
};

class FCLayer: public WeightLayer {
public:
	int _outputSize;
	FCLayer(ConvNet* convNet, std::map<std::string, std::string> params);
	void fprop(Matrix* input);
	void postInit();
};

class DataLayer: public Layer {
public:
	DataLayer(ConvNet* convNet, std::map<std::string, std::string> params);
	void fprop(Matrix* input);
};

class SoftMaxLayer: public Layer {
public:
	SoftMaxLayer(ConvNet* convNet, std::map<std::string, std::string> params);
};

class LogRegCostLayer: public Layer {
public:
	LogRegCostLayer(ConvNet* convNet, std::map<std::string, std::string> params);
};
#endif /* LAYER_H_ */
