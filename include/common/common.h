/*
 * common.h
 *
 *  Created on: Jun 24, 2014
 *      Author: mac
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <sstream>
#include <vector>

inline int str2int(std::string number) {
	std::stringstream ss;
	int res;
	ss << number;
	ss >> res;
	return res;
}
inline std::vector<std::string> split(std::string line, std::string delimiters) {
	for(int i = 0; i < line.size(); ++ i)
		if(line[i] == delimiters[0])
			line[i] = ' ';
	std::vector<std::string> res;
	std::stringstream ss;
	std::string tok;
	ss << line;
	while ( ss >> tok )
		res.push_back(tok);
	return res;
}

#endif /* COMMON_H_ */
