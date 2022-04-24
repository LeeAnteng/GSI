#include <bits/stdc++.h>
using namespace std;

typedef pair<int,int> PII;

int main(int argc, const char * argv[]) {
    // snap.txt label_count
    if( argc > 3 || argc < 2) {
        cerr<<"invalid arguments!"<<endl;
		return -1;
    }
    string row_data = argv[1];
    unsigned label_count = 1;
    if (argc == 3) {
        label_count = atoi(argv[2]);
    }
    FILE* dfp = fopen(row_data.c_str(), "r");
    if (dfp == NULL) {
        cerr<<"input open error!"<<endl;
		return -1;
    }
    set<PII> edges_set;
    set<int> vertex_set;
    char c1,c2;
    int id0, id1, id2, id3;
    while (fscanf(dfp,"%c",&c1) != EOF) 
    {
        if (c1 == '#')
        fscanf(dfp, "%*[^\n]%*c");
        else break;
    }
    fseek(dfp, -1, SEEK_CUR);
    while(fscanf(dfp, "%d\t%d\n",&id0, &id1) != EOF) {
        vertex_set.insert(id0);
        vertex_set.insert(id1);
        id2 = min(id0, id1);
        id3 = max(id0, id1);
        // printf("%d %d\n", id2, id3);
        //防止出现自回边和多重边
        if (id2 != id3 && edges_set.find({id2, id3}) == edges_set.end() && edges_set.find({id3,id2}) == edges_set.end()) {
            edges_set.insert({id2, id3});
        }
    }
    int vertex_count = vertex_set.size();
    unordered_map<int, int> old_to_new;
    vector<int> degree(vertex_count, 0);
    vector<PII> new_edge;
    int i = 0;
    for (auto& oldvid:vertex_set) {
        old_to_new[oldvid] = i++;
    }
    for (auto& edge: edges_set) {
        id0 = old_to_new[edge.first];
        id1 = old_to_new[edge.second];
        new_edge.push_back({id0, id1});
        degree[id0]++;
        degree[id1]++;
    }
    int edge_count = new_edge.size();
    int maxdegree = *max_element(degree.begin(), degree.end());
    for (int i = 0; i < vertex_count; i++) {
        // if (degree[i] >= 1000)
        printf("vid = %d, degree = %d\n",i, degree[i]);

    }
    string output ="../data/" + row_data.substr(17) + ".g";
    cout<<output;
    FILE* ofp = fopen(output.c_str(), "w+");
    fprintf(ofp, "# node_count = %d, edges_count = %d, maxdegree = %d\n", vertex_count, edge_count, maxdegree);
    fprintf(ofp, "t # 0\n");
    fprintf(ofp, "%d %d %d 1\n", vertex_count, edge_count, label_count);
    srand((unsigned int)(time(NULL)));
    for (int j = 0; j < vertex_count; j++) {
        
        fprintf(ofp, "v %d %d\n",j,rand()%label_count + 1);
    }
    for (auto& edge: new_edge) {
        fprintf(ofp, "e %d %d 1\n", edge.first, edge.second);
    }
    fprintf(ofp, "t # -1\n");
    fclose(ofp);


}