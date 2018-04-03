/*
 * This program reads a factor graph file (.fg) and runs Loopy Belief Propagation
 * to infer the marginals for each node thus inferring the category of that node.
 *
 *
 * author: Kaiyu Zheng
 */

#include <dai/alldai.h>
#include <dai/bp.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace dai;

int main(int argc, char **argv) {

  // This program doesn't take evidence file.

  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <filename.fg>" << endl << endl;
    cout << "Reads factor graph <filename.fg> and runs" << endl;
    cout << "Belief Propagation to produce Marginal result" << endl << endl;
    return 1;
  }

  // Read factor graph from file.
  FactorGraph TopologicalMap;
  TopologicalMap.ReadFromFile(argv[1]);

  // Inference engine using BP
  PropertySet infprops;
  infprops.set("verbose", (size_t)1);
  infprops.set("updates", string("SEQRND"));
  infprops.set("tol", 1e-9);
  infprops.set("inference", string("SUMPROD"));
  infprops.set("damping", string("0.0"));
  infprops.set("logdomain", true);

  InfAlg* inf = newInfAlg( "BP", TopologicalMap, infprops );
  inf->init();
  inf->run();
  for( size_t i = 0; i < TopologicalMap.nrVars(); i++ ) {// iterate over all variables in fg
    cout <<  inf->belief(TopologicalMap.var(i)) << endl;
  }
}
