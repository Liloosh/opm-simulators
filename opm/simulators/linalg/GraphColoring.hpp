/*
  Copyright 2018 Equinor

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_GRAPHCOLORING_HEADER_INCLUDED
#define OPM_GRAPHCOLORING_HEADER_INCLUDED

#include <opm/common/TimingMacros.hpp>

#include <opm/grid/utility/SparseTable.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

namespace Opm {
namespace Detail {
template <class Graph>
std::size_t colorGraphWelshPowell(const Graph& graph,
                                  std::deque<typename Graph::VertexDescriptor>& orderedVertices,
                                  std::vector<int>& colors,
                                  int color,
                                  int noVertices)
{
    std::vector<int> forbidden(noVertices, false);
    std::size_t noColored = 0;

    for (auto vertex = orderedVertices.begin(),
              vertexEnd = orderedVertices.end(); vertex != vertexEnd; ++vertex) {
        // Skip forbidden vertices
        while (vertex != vertexEnd && forbidden[*vertex])
            ++vertex;
        if (vertex == vertexEnd) {
            break;
        }

        // Color Vertex
        colors[*vertex] = color;
        ++noColored;
        // Forbid neighors
        for (auto edge = graph.beginEdges(*vertex),
                  endEdge = graph.endEdges(*vertex); edge != endEdge; ++edge) {
            forbidden[edge.target()] = true;
        }
    }
    // forbidden vertices will be colored next for coloring
    using Vertex = typename Graph::VertexDescriptor;
    std::erase_if(orderedVertices,
                  [&forbidden](const Vertex& vertex)
                  { return !forbidden[vertex]; });

    return noColored;
}

template <class Graph, class Functor>
std::size_t breadthFirstSearch(const Graph& graph,
                               typename Graph::VertexDescriptor root,
                               Functor functor)
{
    std::vector<int> visited(graph.maxVertex() + 1, false);
    using Vertex = typename Graph::VertexDescriptor;
    std::queue<Vertex> nextVertices;
    std::size_t noVisited = 0;
    nextVertices.push(root);
    visited[root] = true; // We do not visit root.

    while (!nextVertices.empty()) {
        auto current = nextVertices.front();
        for (auto edge = graph.beginEdges(current),
                  endEdge = graph.endEdges(current); edge != endEdge; ++edge) {
            if (!visited[edge.target()]) {
                visited[edge.target()] = true;
                nextVertices.push(edge.target());
                functor(edge.target());
                ++noVisited;
            }
        }
        nextVertices.pop();
    }
    return noVisited;
}
} // end namespace Detail


/// \brief Color the vertices of graph.
///
/// It uses the algorithm of Welsh and Powell for this.
/// \param graph The graph to color. Must adhere to the graph interface of dune-istl.
/// \return A pair of a vector with the colors of the vertices and the number of colors
///         assigned
template <class Graph>
std::tuple<std::vector<int>, int, std::vector<std::size_t>>
colorVerticesWelshPowell(const Graph& graph)
{
    using Vertex = typename Graph::VertexDescriptor;
    std::deque<Vertex> orderedVertices;
    auto noVertices = graph.maxVertex() + 1;
    std::vector<int> degrees(noVertices, 0);
    int maxDegree = 0;
    std::ptrdiff_t firstDegreeChange = 0;

    // populate deque
    for (auto vertex = graph.begin(),
              endVertex = graph.end(); vertex != endVertex; ++vertex) {
        auto currentVertex = *vertex;
        auto& degree = degrees[currentVertex];

        for (auto edge = graph.beginEdges(currentVertex),
                  endEdge = graph.endEdges(currentVertex); edge != endEdge; ++edge) {
            ++degree;
        }

        if (degree >= maxDegree) {
            orderedVertices.emplace_front(currentVertex);
            ++firstDegreeChange;
            if (degree > maxDegree) {
                firstDegreeChange = 1;
                maxDegree = degree;
            }
        } else {
            orderedVertices.emplace_back(currentVertex);
        }
    }

    // order deque by descending degree
    std::stable_sort(orderedVertices.begin() + firstDegreeChange,
                     orderedVertices.end(),
                     [&degrees](const Vertex& v1, const Vertex& v2)
                     { return degrees[v1] > degrees[v2]; });

    // Overwrite degree with color
    auto& colors = degrees;
    std::ranges::fill(colors, -1);

    int color = 0;
    std::vector<std::size_t> verticesPerColor;
    verticesPerColor.reserve(10);

    while (!orderedVertices.empty()) {
        verticesPerColor.push_back(Detail::colorGraphWelshPowell(graph, orderedVertices,
                                                                 colors, color++, noVertices));
    }
    return std::make_tuple(colors, color, verticesPerColor);
}

/// \! Reorder colored graph preserving order of vertices with the same color.
template <class Graph>
std::vector<std::size_t>
reorderVerticesPreserving(const std::vector<int>& colors,
                          int noColors,
                          const std::vector<std::size_t>& verticesPerColor,
                          const Graph& graph)
{
    std::vector<std::size_t> colorIndex(noColors, 0);
    std::vector<std::size_t> indices(graph.maxVertex() + 1);
    std::partial_sum(verticesPerColor.begin(),
                     verticesPerColor.begin() + verticesPerColor.size() - 1,
                     colorIndex.begin() + 1);

    for (const auto& vertex : graph) {
        indices[vertex] = colorIndex[colors[vertex]]++;
    }
    return indices;
}

/// \! Reorder Vetrices in spheres
template <class Graph>
std::vector<std::size_t>
reorderVerticesSpheres(const std::vector<int>& colors,
                       int noColors,
                       const std::vector<std::size_t>& verticesPerColor,
                       const Graph& graph,
                       typename Graph::VertexDescriptor root)
{
    std::vector<std::size_t> colorIndex(noColors, 0);
    const auto notVisitedTag = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> indices(graph.maxVertex() + 1, notVisitedTag);
    using Vertex = typename Graph::VertexDescriptor;
    std::partial_sum(verticesPerColor.begin(),
                     verticesPerColor.begin() + verticesPerColor.size() - 1,
                     colorIndex.begin() + 1);
    std::size_t noVisited = 0;
    auto numberer = [&colorIndex, &colors, &indices](Vertex vertex)
    {
        indices[vertex] = colorIndex[colors[vertex]]++;
    };

    while (noVisited < graph.maxVertex() + 1) {
        numberer(root);
        ++noVisited; // root node already visited and not visited in BFS
        noVisited += Detail::breadthFirstSearch(graph, root, numberer);
        if (noVisited < graph.maxVertex() + 1) {
            // Graph is disconnected search for not yet visited node
            for (auto vertex : graph) {
                if (indices[vertex] == notVisitedTag) {
                    // \todo make sure that this is a peripheral node!
                    root = vertex;
                    break;
                }
            }
        }
    }
    return indices;
}

/// \brief Reorder vertices using the Reverse Cuthill-McKee (RCM) algorithm.
///
/// This reduces matrix bandwidth and profile, significantly improving cache locality
/// and reducing memory traffic during sparse triangular solves (ILU0 apply).
///
/// \param graph The graph to reorder (must adhere to Dune-ISTL MatrixGraph interface).
/// \param numNodes The number of vertices to reorder. If 0, all vertices are reordered.
///                 When numNodes < graph.noVertices() (e.g. interiorSize in parallel MPI),
///                 only the first numNodes vertices are reordered among themselves,
///                 preserving the interior/ghost partition.
/// \return Permutation vector `ordering` where `ordering[old_vertex] = new_vertex`.
template <class Graph>
std::vector<std::size_t>
reorderReverseCuthillMcKee(const Graph& graph, std::size_t numNodes = 0)
{
    const std::size_t totalVertices = graph.noVertices();
    const std::size_t targetVertices = (numNodes > 0 && numNodes <= totalVertices) ? numNodes : totalVertices;

    if (targetVertices <= 1) {
        std::vector<std::size_t> identity(totalVertices);
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    // Step 1: Build symmetric adjacency list for the target subgraph [0, targetVertices - 1]
    std::vector<std::vector<std::size_t>> adj(targetVertices);
    for (std::size_t u = 0; u < targetVertices; ++u) {
        for (auto edge = graph.beginEdges(u), endEdge = graph.endEdges(u); edge != endEdge; ++edge) {
            std::size_t v = edge.target();
            if (v < targetVertices && v != u) {
                adj[u].push_back(v);
                adj[v].push_back(u); // Ensure symmetry (A + A^T)
            }
        }
    }

    // Deduplicate neighbor lists and compute degrees
    std::vector<std::size_t> degree(targetVertices);
    for (std::size_t u = 0; u < targetVertices; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
        degree[u] = adj[u].size();
    }

    // Step 2: Helper to find pseudo-peripheral vertex in a connected component
    auto findPseudoPeripheral = [&](std::size_t start) -> std::size_t {
        std::size_t current = start;
        std::size_t maxDistance = 0;

        for (int sweep = 0; sweep < 5; ++sweep) {
            std::vector<int> dist(targetVertices, -1);
            std::queue<std::size_t> q;
            q.push(current);
            dist[current] = 0;

            std::size_t maxD = 0;
            std::vector<std::size_t> furthestNodes;

            while (!q.empty()) {
                std::size_t u = q.front();
                q.pop();

                std::size_t d = static_cast<std::size_t>(dist[u]);
                if (d > maxD) {
                    maxD = d;
                    furthestNodes.clear();
                    furthestNodes.push_back(u);
                } else if (d == maxD) {
                    furthestNodes.push_back(u);
                }

                for (std::size_t v : adj[u]) {
                    if (dist[v] == -1) {
                        dist[v] = dist[u] + 1;
                        q.push(v);
                    }
                }
            }

            if (maxD <= maxDistance && sweep > 0) {
                break;
            }
            maxDistance = maxD;

            // Pick candidate with minimum degree among furthest nodes
            std::size_t bestCandidate = furthestNodes[0];
            std::size_t minDeg = degree[bestCandidate];
            for (std::size_t cand : furthestNodes) {
                if (degree[cand] < minDeg) {
                    minDeg = degree[cand];
                    bestCandidate = cand;
                }
            }

            if (bestCandidate == current) {
                break;
            }
            current = bestCandidate;
        }
        return current;
    };

    // Step 3: Cuthill-McKee BFS over all connected components
    std::vector<bool> visited(targetVertices, false);
    std::vector<std::size_t> rcmOrder;
    rcmOrder.reserve(targetVertices);

    std::size_t visitedCount = 0;
    while (visitedCount < targetVertices) {
        // Find unvisited vertex with minimum degree
        std::size_t root = targetVertices;
        std::size_t minDeg = std::numeric_limits<std::size_t>::max();
        for (std::size_t u = 0; u < targetVertices; ++u) {
            if (!visited[u] && degree[u] < minDeg) {
                minDeg = degree[u];
                root = u;
            }
        }
        if (root == targetVertices) {
            break;
        }

        // Find pseudo-peripheral vertex for this connected component
        std::size_t startNode = findPseudoPeripheral(root);

        // BFS traversal for Cuthill-McKee
        std::vector<std::size_t> componentCM;
        std::queue<std::size_t> q;

        q.push(startNode);
        visited[startNode] = true;
        componentCM.push_back(startNode);
        ++visitedCount;

        while (!q.empty()) {
            std::size_t u = q.front();
            q.pop();

            // Collect unvisited neighbors
            std::vector<std::size_t> unvisitedNeighbors;
            for (std::size_t v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    ++visitedCount;
                    unvisitedNeighbors.push_back(v);
                }
            }

            // Sort unvisited neighbors by degree ascending (tie-break by index)
            std::stable_sort(unvisitedNeighbors.begin(), unvisitedNeighbors.end(),
                             [&degree](std::size_t a, std::size_t b) {
                                 if (degree[a] != degree[b]) {
                                     return degree[a] < degree[b];
                                 }
                                 return a < b;
                             });

            for (std::size_t v : unvisitedNeighbors) {
                q.push(v);
                componentCM.push_back(v);
            }
        }

        // Reverse Cuthill-McKee for this component
        std::reverse(componentCM.begin(), componentCM.end());
        rcmOrder.insert(rcmOrder.end(), componentCM.begin(), componentCM.end());
    }

    // Step 4: Construct ordering vector `ordering[old_idx] = new_idx`
    std::vector<std::size_t> ordering(totalVertices);
    for (std::size_t newIdx = 0; newIdx < rcmOrder.size(); ++newIdx) {
        ordering[rcmOrder[newIdx]] = newIdx;
    }
    // Any remaining vertices (e.g. ghost cells in parallel) retain their identity
    for (std::size_t k = targetVertices; k < totalVertices; ++k) {
        ordering[k] = k;
    }

    return ordering;
}

/// \brief Specify coloring type.
/// \details The coloring types have been implemented initially to parallelize DILU
///          preconditioner and parallel sparse triangular solves.
///          Symmetric coloring will create a dependency from row i to j if
///          both element A_ij and A_ji exists.
///          Lower coloring creates a dependency from row i to j (where i < j) if
///          A_ij is nonzero.
///          Upper coloring creates a dependecy from row i to j (where i > j) if A_ij is nonzero.
enum class ColoringType { SYMMETRIC, LOWER, UPPER };

/// This coloring algorithm interprets the sparsity structure of a matrix as a graph. Each row is given a color
/// or level where all the rows in the same level only have dependencies from lower levels. The level computation
/// is done with dynamic programming, and to improve caching the rows in the same level stay in matrix order.
/// \brief Given a matrix and dependecy type, returns a SparseTable grouping the rows by which
///        can be executed in parallel without breaking dependencies
/// \param matrix A dune sparse matrix
/// \param coloringType The coloringtype determines what constitutes a dependency,
///        see ColoringType definition above
/// \return SparseTable with rows of the matrix grouped into least number of groups
///         while dependencies only come from groups with lower index
template <class M>
Opm::SparseTable<std::size_t>
getMatrixRowColoring(const M& matrix, ColoringType coloringType)
{
    OPM_TIMEBLOCK(createMatrix);

    std::vector<std::size_t> color(matrix.N(), 0);
    std::vector<std::size_t> rowIndices(matrix.N(), 0);

    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::vector<std::size_t> colorCnt;
    // These dynamic programming computations only rely on the following observation:
    // level[row_i] = 1 + max{level[row_j]} for all j which i depend on
    // This minimizes the level of each row, and every rows dependencies belong to a lower level set.
    if (coloringType == ColoringType::SYMMETRIC) {
        for (auto i = matrix.begin(); i != matrix.end(); ++i) {
            for (auto a_ij = i->begin(); a_ij.index() != i.index(); ++a_ij) {
                auto a_ji = matrix[a_ij.index()].find(i.index());
                if (a_ji != matrix[a_ij.index()].end()) {
                    color[i.index()] = std::max({color[i.index()], color[a_ij.index()] + 1});
                }
            }
            if (color[i.index()] >= colorCnt.size()) {
                colorCnt.push_back(1);
            } else {
                ++colorCnt[color[i.index()]];
            }
        }
    } else if (coloringType == ColoringType::UPPER) {
        for (auto i = matrix.beforeEnd(); i != matrix.beforeBegin(); --i) {
            for (auto a_ij = ++(i->find(i.index())); a_ij != i->end(); ++a_ij) {
                color[i.index()] = std::max({color[i.index()], color[a_ij.index()] + 1});
            }
            if (color[i.index()] >= colorCnt.size()) {
                colorCnt.push_back(1);
            } else {
                ++colorCnt[color[i.index()]];
            }
        }
    } else if (coloringType == ColoringType::LOWER) {
        for (auto i = matrix.begin(); i != matrix.end(); ++i) {
            for (auto a_ij = i->begin(); a_ij.index() != i.index(); ++a_ij) {
                color[i.index()] = std::max({color[i.index()], color[a_ij.index()] + 1});
            }
            if (color[i.index()] >= colorCnt.size()) {
                colorCnt.push_back(1);
            } else {
                ++colorCnt[color[i.index()]];
            }
        }
    }

    std::stable_sort(rowIndices.begin(),
                     rowIndices.end(),
                     [&color](const std::size_t a, const std::size_t b)
                     { return color[a] < color[b]; });

    return {rowIndices.data(), rowIndices.data() + rowIndices.size(),
            colorCnt.data(), colorCnt.data() + colorCnt.size()};
}

} // end namespace Opm

#endif
