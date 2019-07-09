#include <torch/extension.h>

#include "hom.h"
#include "cohom.h"
#include "union_find.h"
#include "complex.h"
#include "backward.h"

namespace py = pybind11;



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SimplicialComplex>(m, "SimplicialComplex")
  .def(py::init<>())
  .def("append", &SimplicialComplex::append)
  .def("betti_numbers", &SimplicialComplex::betti_numbers)
  .def("initialize", &SimplicialComplex::initialize)
  .def("extendFloat", &SimplicialComplex::extend)
  .def("extendFlag", &SimplicialComplex::extend_flag)
  .def("sortedOrder", &SimplicialComplex::sortedOrder)
  .def("dim", &SimplicialComplex::dim)
  .def("printFiltration", &SimplicialComplex::printFiltration)
  .def("printFunctionMap", &SimplicialComplex::printFunctionMap)
  .def("printCritInds", &SimplicialComplex::printCritInds)
  .def("printDims", &SimplicialComplex::printDims)
  .def("printBoundary", &SimplicialComplex::printBoundary)
  .def("numPairs", &SimplicialComplex::numPairs)
  .def("printCells", &SimplicialComplex::printComplex);
  m.def("persistenceForwardCohom", &persistence_forward);
  m.def("persistenceBackward", &persistence_backward);
  m.def("persistenceBackwardFlag", &persistence_backward_flag);
  m.def("persistenceForwardHom", &persistence_forward_hom);
  m.def("persistenceForwardUF", &persistence_forward_uf);
  m.def("persistenceForwardUF2", &persistence_forward_uf2);
  m.def("critEdges", &crit_edges_uf);
  m.def("graphCritEdges", &graph_crit_edges);
}
