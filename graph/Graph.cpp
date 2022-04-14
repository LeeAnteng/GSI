#include "Graph.h"

using namespace std;

uint32_t myhash(const void * key, int len, uint32_t seed) 
{
    return Util::MurmurHash2(key, len, seed);
}

void 
Graph::addVertex(VID id,LABEL _vlb)
{
	this->vertices.push_back(Vertex(id, _vlb));
}
void 
Graph::addEdge(VID _from, VID _to)
{
	this->vertices[_from].neighbors.push_back(Vertex(_to, vertices[_to].label));
    this->vertices[_to].neighbors.push_back(Vertex(_from, vertices[_from].label));
}
void 
Graph::buildCSR() {
    cout<<"begin build csr"<<endl;
    row_offset = new unsigned[vertex_num + 1];
    memset(row_offset, 0, sizeof(unsigned) * (vertex_num + 1));
    for (int i = 1; i <= vertex_num; i++) {
        row_offset[i] = row_offset[i - 1] + vertices[i - 1].neighbors.size();
    }
    undir_edge_num = row_offset[vertex_num];
    col_offset = new pair<uint,uint>[undir_edge_num];
    int j = 0;
    for (int i = 0; i < vertex_num; i++) {
        sort(vertices[i].neighbors.begin(), vertices[i].neighbors.end());
        for (const auto& neig: vertices[i].neighbors) {
            col_offset[j].first = neig.label;
            col_offset[j].second = neig.id;
            j++;
        }
    }
    cout<<"finished build csr"<<endl;
}

//签名总长8个unsigned，1个存label，
//3个：12 bucket * 8 bit的邻居哈希，
//2个：64 bucket * 1 bit的三角哈希
//2个：64 bucket * 1 bit的路径哈希
void
Graph::buildSignature(bool col_oriented) {
    cout<<"build signature for a new graph"<<endl;
    unsigned tablen = this->vertex_num * SIGNUM;
    sig_table = new unsigned[tablen];
    memset(sig_table, 0, sizeof(unsigned) * tablen);
    bool* isVisited = new bool[this->vertex_num];
    for (int i = 0; i < this->vertex_num; i++) {
        //结构哈希
        queue<pair<unsigned, unsigned>> q;
        
        memset(isVisited, 0, sizeof(bool) * this->vertex_num);

        Vertex& v = this->vertices[i];
        int basei = SIGNUM * i;
        sig_table[basei] = v.label;
        int pos;
        for (const auto& nei: v.neighbors) {

            q.push({nei.id, nei.label});
            isVisited[nei.id] = true;

            LABEL nei_label = nei.label;
            pos = myhash(&nei_label, 4, HASHSEED) % 12;
            int a = pos / 4, b = pos % 4;
            unsigned t = sig_table[basei + 1 + a];
            unsigned c = 255 << (8 * b);
            c = c & t;
            c = c >> (8 * b);
            switch (c)
            {
            case 0:
                c = 1;
                break;
            case 1:
                c = 3;
                break;
            case 3:
                c = 7;
                break;
            case 7:
                c = 15;
                break;
            case 15:
                c = 31;
                break;
            case 31:
                c = 63;
                break;
            case 63:
                c = 127;
                break;
            case 127:
                c = 255;
                break;
            default:
                c = 255;
                break;
            }
            c = c << (8 * b);
            t = t | c;
            sig_table[basei + 1 + a] = t;
        }
        
        while(!q.empty()) {
            unsigned idx = q.front().first;
            unsigned label = q.front().second;
            q.pop();
            for (const auto& nei : vertices[idx].neighbors) {
                unsigned nei_id = nei.id, nei_label = nei.label;
                if (nei_id == i) continue;
                if (isVisited[nei_id]) //三角哈希
                {
                    unsigned s[2];
                    s[0] = min(label, nei_label);
                    s[1] = max(label, nei_label);
                    pos = myhash(s, 8, HASHSEED) % 64;
                    int a = pos / 32, b = pos % 32;
                    unsigned t = sig_table[basei + 4 + a];
                    unsigned c = 1 << b;
                    t = t | c;
                    sig_table[basei + 4 + a] = t;
                }
                else {//路径哈希
                    unsigned s[2];
                    s[0] = label;
                    s[1] = nei_label;
                    pos = myhash(s, 8, HASHSEED) % 64;
                    int a = pos / 32, b = pos % 32;
                    unsigned t = sig_table[basei + 6 + a];
                    unsigned c = 1 << b;
                    t = t | c;
                    sig_table[basei + 6 + a] = t;
                }
            }
        }
    }
    delete[] isVisited;
    if(col_oriented) {
        unsigned* new_table  = new unsigned[tablen];
        unsigned base = 0;
        for (int k = 0; k < SIGNUM; ++k) {
            for (int i = 0; i < this->vertex_num; i++) {
                new_table[base++] = sig_table[i * SIGNUM + k];
            }
        }
        delete[] this->sig_table;
        this->sig_table = new_table;
    }
    

}

void
Graph::printSig() {
    bitset<32> s;
    for (int i = 0; i < this->vertex_num; i++) {
        printf("print %d 's signature:\n", i);
        for (int j = 0; j < SIGNUM; j++) {
            s = sig_table[SIGNUM*i + j];
            cout<<j<<":"<<s<<endl;
        }
        cout<<"--------------------------"<<endl;
    }
}

void
Graph::printGraph(){
    for (int i = 0; i < vertex_num; i++) {
        cout<<"id: "<<vertices[i].id<<" "<<"label: "<<vertices[i].label<<endl;
        for (auto& nei : vertices[i].neighbors) {
            cout<<nei.id<<" ";
        }
        cout<<endl;
        
    }
    }
void
Graph::printCSR() {
    cout<<"row_offset:"<<endl;
    for (int i = 0; i <= vertex_num; i++) {
        cout<<row_offset[i]<<" ";
    }
    cout<<endl;
    cout<<"col_offset:"<<endl;
    for (int i = 0; i < undir_edge_num; i++) {
        cout<<col_offset[i].first<<" ";
    }
    cout<<endl;
    for (int i = 0; i < undir_edge_num; i++) {
        cout<<col_offset[i].second<<" ";
    }
}
