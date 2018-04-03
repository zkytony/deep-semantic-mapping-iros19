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
  // The factor graph for a topological map should be like this. There is a factor
  // of two variables between each pair of connected nodes. The table of this factor
  // basically iterates through possible values for the semantic attributes that the
  // two nodes can have.
  //
  // If, one wants to do local classification correction, the factor graph file should
  // contain some additional information: Each node is connected to an additional factor
  // for one variable (that node), and has N possible values where N is the number of
  // semantic categories, and each value is the likelihood that this node belongs to
  // its corresponding category.

  if (argc != 4 && argc != 5) {
    cout << "Usage: " << argv[0] << " <filename.fg> <filename.tab> <inference_type> [outfile]" << endl << endl;
    cout << "Reads factor graph <filename.fg> and runs" << endl;
    cout << "Belief Propagation to infer the missing variable states using <inference_type>" << endl;
    cout << " with evidence provided from <filename.tab>. MAP states are stored to `outfile` in JSON format." << endl;
    cout << " <inference type> is a flag, can be '-mpe' or '-marginal'" << endl << endl;
    
    return 1;
  }

  const char *outfile = "./Inference_result.json";
  if (argc == 5) {
    outfile = argv[4];
  }

  // Read factor graph from file.
  FactorGraph TopologicalMap;
  TopologicalMap.ReadFromFile(argv[1]);

  // Evidence
  Evidence e;
  ifstream estream(argv[2]);
  e.addEvidenceTabFile(estream, TopologicalMap);
  cout << "Number of samples: " << e.nrSamples() << endl;

  // Inference type
  string inf_type(argv[3]);

  // Inference engine using BP
  PropertySet infprops;
  infprops.set("verbose", (size_t)1);
  infprops.set("updates", string("SEQRND"));
  infprops.set("tol", 1e-9);
  infprops.set("inference", string("SUMPROD"));
  infprops.set("damping", string("0.0"));
  infprops.set("logdomain", true);

  // Set (clamp) variable states to observation
  for( Evidence::const_iterator ei = e.begin(); ei != e.end(); ++ei ) {
    InfAlg* inf = newInfAlg( "BP", TopologicalMap, infprops );
    for (Evidence::Observation::const_iterator i = ei->begin(); i != ei->end(); ++i) {
      inf->clamp(TopologicalMap.findVar(i->first), i->second);
    }
    inf->init();
    inf->run();

    if (inf_type == "marginal") {
      // Print a json file for the marginal inference results
      ofstream of;
      of.open(outfile);
      of << "{" << endl;
      for( size_t i = 0; i < TopologicalMap.nrVars(); i++ ) {// iterate over all variables in fg
	Factor b = inf->belief(TopologicalMap.var(i));
	for (std::vector<Var>::const_iterator vi = b.vars().begin();
	     vi != b.vars().end(); vi++) {
	  // There should be only one variable
	  Var v = *vi;
	  string p = toString(b.p());
	  replace(p.begin(), p.end(), '(', '[');
	  replace(p.begin(), p.end(), ')', ']');
	  of << "\"" << v << "\": " << p;
	}
	if (i < TopologicalMap.nrVars()-1) {
	  of << ",";
	}
	of << endl;
      }
      of << "}";
      of.close();
    } else {
      if (inf_type != "mpe" and inf_type != "map") {
	cout << "Inference type " << inf_type << " is invalid!" << endl;
	return 1;
      }

      // MAP state
      vector<size_t> mpstate = inf->findMaximum();

      ofstream of;
      of.open(outfile);
      of << "{" << endl;
      for( size_t i = 0; i < mpstate.size(); i++ ) {
	of << "\"" << TopologicalMap.var(i) << "\"" << ": " << mpstate[i] << "," << endl;
      }
      of << "\"_logScore_\": " << TopologicalMap.logScore(mpstate);
      of << "}";
      of.close();
    }
  }
  return 0;
}
