//
//Data Structure:  CSR with 4 arrays, we should use structure of array instead of array of structure
//row offsets
//column indices
//values(label of edge)
//flags(label of vertice)

#ifndef _GRAPH_GRAPH_H
#define _GRAPH_GRAPH_H

#include "../util/Util.h"

class Vertex {
	public:
	LABEL label;
	std::vector<VID> neighbor;
	Vertex() {
		label = -1;
	}
	Vertex(LABEL lb):label(lb){}
};
class Graph {
	public:
	std::vector<Vertex> vertices;
	int vertex_num;
	int vlabel_num;
	void addVertex(LABEL _vlb);
	void addEdge(VID _from, VID _to);
	void printGraph();
};

#endif