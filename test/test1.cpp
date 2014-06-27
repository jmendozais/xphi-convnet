/*
 * test1.cpp
 *
 *  Created on: Jun 15, 2014
 *      Author: mac
 */

//#include <iostream>
using namespace std;

#include "../include/convnet/convnet.h"
#include "../include/common/data.h"
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "omp.h"

std::vector<std::map<std::string, std::string> > parseInput(std::string file) {
	std::vector<std::map<std::string, std::string> > params;
	std::ifstream fin(file.c_str(), std::ifstream::in);
	std::string line;
	assert(!fin.eof());
	fin >> line;
	do {
		assert(std::string::npos != line.find('['));
		std::size_t pos = line.find(']');
		assert( std::string::npos != pos);
		std::map<std::string, std::string> layerParams;
		std::string name = line.substr(1, pos-1), key, value;
		layerParams["name"] = name;
		while(fin >> line) {
			assert(line.size() > 0);
			if(line[0] == '[')
				break;
			pos = line.find('=');
			assert(pos != std::string::npos);
			key = line.substr(0, pos);
			value = line.substr(pos + 1, line.size() - pos - 1);
			layerParams[key] = value;
		}
		params.push_back(layerParams);
		if(fin.eof())
			break;
	} while(true);
	return params;
}

void testConvNet() {
	std::vector<std::map<std::string, std::string> > params = parseInput("test/19confignopadding");
	ConvNet convNet(params, 1);
	DataProvider dp(1);
	dp.addData("test/cifar-10-batches-bin/data_batch_1.bin");
	int numMinibatches = dp.getSize()/dp.getMiniBatchSize();
	for(int i = 0; i < 1; ++i) {
		std::vector<Matrix*> inputs = dp.getMiniBatch(2);
		for(int j = 0; j < 3*36*36; ++ j)
			std::cout << inputs[0]->getData()[j] << ", ";
		delete inputs[0];
		delete inputs[1];
	}
}
const int MINI_BATCH_SIZE = 16;
double testConvNetDataDividedBy(std::string inputName, int divisor) {
	std::vector<std::map<std::string, std::string> > params = parseInput("test/19confignopadding");
	ConvNet convNet(params, MINI_BATCH_SIZE);
	DataProvider dp(MINI_BATCH_SIZE);
	dp.addData(inputName);
	int numMinibatches = dp.getNumEntries()/dp.getMiniBatchSize();

	double start, end, duration;
	start = omp_get_wtime();
	int checker = 0;
	for(int i = 0; i < (numMinibatches/divisor); ++i) {
		std::vector<Matrix*> inputs = dp.getMiniBatch(i);
		convNet.fprop(inputs);
		checker += convNet._checker;
		checker %= 37;/*
		if(i + 1 == (numMinibatches/divisor)) {
			delete inputs[0];
			delete inputs[1];
		}*/
	}
	//std::cout << "checksum: " << checker << std::endl;
	end = omp_get_wtime();
	return end - start;
}

double testConvNet2(std::string inputName) {
	return testConvNetDataDividedBy(inputName, 100);
}
// 2 cores, 1 thread for all 2.44
// 2 cores, 2 thread for all 1.57
// 2 cores, 16 : 1.22
int testBenchmark() {
	std::string inputName;
	std::cin >> inputName;
	omp_set_num_threads(1);
	double stime = testConvNet2(inputName);
	omp_set_num_threads(64);
	double ptime = testConvNet2(inputName);
	std::cout << stime << std::endl;
	std::cout << ptime << std::endl;
	std::cout << "Speedup: " << (stime / ptime) << std::endl;
	return 0;
}
const int SCALE = 19;
int speedupByDifferentProblemSize(int maxThreads, int problemSizes) {
	std::string inputName;
	double stime, ptime;
	std::cin >> inputName;
	for(int i = 0; i < problemSizes; ++i) {
		omp_set_num_threads(1);
		stime = testConvNetDataDividedBy(inputName, (1<<i) * SCALE);
		std::cout << (i?", ":"") <<  "[0, 1";
		for(int numThreads = 2; numThreads <= maxThreads; numThreads <<= 1) {
			omp_set_num_threads(numThreads);
			ptime = testConvNetDataDividedBy(inputName, (1<<i) * SCALE);
			std::cout << ", " << (stime / ptime);
		}
		std::cout << "]";
	}
	std::cout << endl;
	return 0;
}
// TODO: analize why with scale = 19 sometimes causes 'Abort :6' message
int main() {
	//return testBenchmark();
	return speedupByDifferentProblemSize(256, 3);
}

