#include "Graph.h"

using namespace std;

void 
Graph::addVertex(LABEL _vlb)
{
	this->vertices.push_back(Vertex(_vlb));
}
void 
Graph::addEdge(VID _from, VID _to)
{
	this->vertices[_from].neighbor.push_back(_to);
    this->vertices[_to].neighbor.push_back(_from);
}

void
Graph::printGraph(){
    cout<<"Print Graph Info"<<endl;
    for(int i = 0; i < this->vertex_num; i++) {
        cout<<i<<" label:"<<this->vertices[i].label<<endl;
        cout<<"Neighbor: ";
        for (auto neig_vid : this->vertices[i].neighbor)
        cout<<neig_vid<<" ";
        cout<<endl;
        cout<<"-------------------------"<<endl;
    }
}