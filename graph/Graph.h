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
	VID id;
	LABEL label;
	std::vector<Vertex> neighbors;
	Vertex() {
		id = -1;
		label = -1;
	}
	Vertex(VID id, LABEL lb):id(id), label(lb){}
	bool operator < (const Vertex& v) {
		if (this->label == v.label) {
			return this->id < v.id;
		}
		else {
			return this->label < v.label;
		}
	}
};
class Graph {
	public:
	unsigned vertex_num;
	unsigned vlabel_num;
	unsigned undir_edge_num;
	//for csr
	unsigned* row_offset;
	std::pair<uint,uint>* col_offset;
	//for signature
	unsigned* sig_table;
	std::vector<Vertex> vertices;
	
	void addVertex(VID id, LABEL _vlb);
	void addEdge(VID _from, VID _to);
	
	void buildCSR();
	void buildSignature(bool col_oriented);
	void printCSR();
	void printGraph();
	void printSig();
};

#endif