/*=============================================================================
# Filename: run.cpp
# Author: bookug
# Mail: bookug@qq.com
# Last Modified: 2017-12-21 16:47
# Description: 
how to time the program?
https://blog.csdn.net/litdaguang/article/details/50520549
warmup GPU and the timing should be the average of 10 runs
=============================================================================*/

#include "../util/Util.h"
#include "../io/IO.h"
#include "../graph/Graph.h"
#include "../match/Match.h"

using namespace std;

//NOTICE:a pattern occurs in a graph, then support++(not the matching num in a graph), support/N >= minsup
vector<Graph*> query_list;

int
main(int argc, const char * argv[])
{
	int i;
	uint2 a;
	string output = "ans.txt";
	if(argc > 5 || argc < 3)
	{
		cerr<<"invalid arguments!"<<endl;
		return -1;
	}
	string data = argv[1];
	string query = argv[2];
	if(argc >= 4)
	{
		output = argv[3];
	}
	int dev = 0;
	if(argc == 5)
	{
		dev = atoi(argv[4]);
	}

	//set the GPU and warmup
	// Match::initGPU(dev);

	long t1 = Util::get_cur_time();

	IO io = IO(query, data, output);
	
	cerr<<"input ok!"<<endl;
	long t2 = Util::get_cur_time();

	unsigned* final_result = NULL;
	int* id_map= NULL;
	unsigned result_row_num = 0, result_col_num = 0;
	Graph* data_graph = NULL;
	Graph* query_graph = NULL;
	io.input(data_graph, io.dfp);
	io.input(query_graph, io.qfp);
	query_graph->buildCSR();
	data_graph->buildCSR();
	query_graph->buildSignature(false);
	data_graph->printCSR();
	// data_graph->buildSignature(false);
	// query_graph->printSig();
	// data_graph->printSig();
	data_graph->buildSignature(true);
	//data_graph->printSig();

	//begin match
	Match m(query_graph,data_graph);
	m.match(io, final_result, result_row_num, result_col_num, id_map);
	io.output(final_result, result_row_num, result_col_num, id_map);
	io.flush();
	delete[] final_result;
	return 0;
}

