#!/usr/bin/env python3

import os
import re
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple, Set

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    print(f"Error: Missing Python visualization library ({e}).")
    print("Please install matplotlib and numpy in your environment:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


# Exact fallback dataset matching SPE1 with 4 MPI ranks if no output log is found
FALLBACK_RANKS_DATA = [
    {
        "rank": 0,
        "owned_blocks": 78,
        "overlap_blocks": 57,
        "total_blocks": 135,
        "owned_scalar": 234,
        "overlap_scalar": 171,
        "total_scalar": 405,
        "int_nnz": 404,
        "boundary_nnz": 66,
        "total_nnz": 759,
        "owned_cells": [0, 1, 2, 10, 11, 12, 22, 32, 33, 42, 43, 52, 53, 54, 62, 63, 64, 72, 73, 74,
                        82, 83, 84, 92, 93, 94, 100, 101, 102, 110, 111, 112, 122, 132, 133, 142, 143,
                        152, 153, 154, 162, 163, 164, 172, 173, 174, 182, 183, 184, 192, 193, 194,
                        200, 201, 202, 210, 211, 212, 222, 232, 233, 242, 243, 252, 253, 254, 262,
                        263, 264, 272, 273, 274, 282, 283, 284, 292, 293, 294],
        "overlap_cells": [3, 13, 20, 21, 23, 31, 34, 41, 44, 51, 55, 61, 65, 71, 75, 81, 85, 91, 95,
                          103, 113, 120, 121, 123, 131, 134, 141, 144, 151, 155, 161, 165, 171, 175,
                          181, 185, 191, 195, 203, 213, 220, 221, 223, 231, 234, 241, 244, 251, 255,
                          261, 265, 271, 275, 281, 285, 291, 295],
        "neighbors": {
            1: {"overlap_blocks": 36, "bnd_nnz": 39},
            2: {"overlap_blocks": 12, "bnd_nnz": 15},
            3: {"overlap_blocks": 9,  "bnd_nnz": 12}
        }
    },
    {
        "rank": 1,
        "owned_blocks": 69,
        "overlap_blocks": 54,
        "total_blocks": 123,
        "owned_scalar": 207,
        "overlap_scalar": 162,
        "total_scalar": 369,
        "int_nnz": 341,
        "boundary_nnz": 57,
        "total_nnz": 653,
        "owned_cells": [20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 75, 76, 80, 81, 85, 86, 90, 91,
                        95, 96, 97, 120, 121, 130, 131, 140, 141, 150, 151, 160, 161, 170, 171, 175,
                        176, 180, 181, 185, 186, 190, 191, 195, 196, 197, 220, 221, 230, 231, 240, 241,
                        250, 251, 260, 261, 270, 271, 275, 276, 280, 281, 285, 286, 290, 291, 295, 296, 297],
        "overlap_cells": [10, 11, 22, 32, 42, 52, 62, 65, 66, 72, 74, 77, 82, 84, 87, 92, 94, 98,
                          110, 111, 122, 132, 142, 152, 162, 165, 166, 172, 174, 177, 182, 184, 187,
                          192, 194, 198, 210, 211, 222, 232, 242, 252, 262, 265, 266, 272, 274, 277,
                          282, 284, 287, 292, 294, 298],
        "neighbors": {
            0: {"overlap_blocks": 39, "bnd_nnz": 39},
            2: {"overlap_blocks": 6,  "bnd_nnz": 6},
            3: {"overlap_blocks": 9,  "bnd_nnz": 12}
        }
    },
    {
        "rank": 2,
        "owned_blocks": 72,
        "overlap_blocks": 45,
        "total_blocks": 117,
        "owned_scalar": 216,
        "overlap_scalar": 135,
        "total_scalar": 351,
        "int_nnz": 378,
        "boundary_nnz": 57,
        "total_nnz": 651,
        "owned_cells": [9, 19, 28, 29, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 55, 56, 57, 58, 59,
                        65, 66, 67, 109, 119, 128, 129, 134, 135, 136, 137, 138, 139, 144, 145, 146, 147,
                        148, 149, 155, 156, 157, 158, 159, 165, 166, 167, 209, 219, 228, 229, 234, 235,
                        236, 237, 238, 239, 244, 245, 246, 247, 248, 249, 255, 256, 257, 258, 259, 265,
                        266, 267],
        "overlap_cells": [8, 18, 24, 25, 26, 27, 33, 43, 54, 64, 68, 69, 75, 76, 77, 108, 118, 124,
                          125, 126, 127, 133, 143, 154, 164, 168, 169, 175, 176, 177, 208, 218, 224,
                          225, 226, 227, 233, 243, 254, 264, 268, 269, 275, 276, 277],
        "neighbors": {
            0: {"overlap_blocks": 12, "bnd_nnz": 15},
            1: {"overlap_blocks": 6,  "bnd_nnz": 6},
            3: {"overlap_blocks": 27, "bnd_nnz": 36}
        }
    },
    {
        "rank": 3,
        "owned_blocks": 81,
        "overlap_blocks": 51,
        "total_blocks": 132,
        "owned_scalar": 243,
        "overlap_scalar": 153,
        "total_scalar": 396,
        "int_nnz": 417,
        "boundary_nnz": 60,
        "total_nnz": 710,
        "owned_cells": [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 68, 69, 77, 78, 79,
                        87, 88, 89, 98, 99, 103, 104, 105, 106, 107, 108, 113, 114, 115, 116, 117, 118,
                        123, 124, 125, 126, 127, 168, 169, 177, 178, 179, 187, 188, 189, 198, 199,
                        203, 204, 205, 206, 207, 208, 213, 214, 215, 216, 217, 218, 223, 224, 225, 226,
                        227, 268, 269, 277, 278, 279, 287, 288, 289, 298, 299],
        "overlap_cells": [2, 9, 12, 19, 22, 28, 33, 34, 35, 36, 37, 58, 59, 67, 76, 86, 97, 102, 109,
                          112, 119, 122, 128, 133, 134, 135, 136, 137, 158, 159, 167, 176, 186, 197,
                          202, 209, 212, 219, 222, 228, 233, 234, 235, 236, 237, 258, 259, 267, 276,
                          286, 297],
        "neighbors": {
            0: {"overlap_blocks": 12, "bnd_nnz": 12},
            1: {"overlap_blocks": 9,  "bnd_nnz": 12},
            2: {"overlap_blocks": 30, "bnd_nnz": 36}
        }
    }
]

# Color palette for MPI ranks
RANK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
RANK_RGBA = [
    (0.12, 0.47, 0.71, 0.90),  # Rank 0 (Blue)
    (1.00, 0.50, 0.05, 0.90),  # Rank 1 (Orange)
    (0.17, 0.63, 0.17, 0.90),  # Rank 2 (Green)
    (0.84, 0.15, 0.16, 0.90)   # Rank 3 (Red)
]


def parse_ranges_string(s: str) -> List[int]:
    """Expand string containing numbers and ranges like '0..2, 10..12, 22' into sorted int list."""
    cells = []
    tokens = [x.strip() for x in s.split(",") if x.strip()]
    for t in tokens:
        if ".." in t:
            parts = t.split("..")
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            cells.extend(range(start, end + 1))
        elif t.isdigit():
            cells.append(int(t))
    return sorted(cells)


def parse_simulation_log(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """Parse rank matrix partitioning and cell distribution details from simulation output file."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Pattern for rank blocks
    pattern = (
        r"Rank\s+(\d+)\s+\|\s+Local Block Rows:\s+\[0\s*\.\.\s*(\d+)\]\s+owned\s+\(\s*(\d+)\s+blks\),\s+\[\s*(\d+)\s*\.\.\s*(\d+)\]\s+overlap\s+\(\s*(\d+)\s+blks\)\s+\|\s+Total:\s+(\d+)\s+blks\s*\n"
        r"\s+\|\s+Scalar Rows:\s+\[0\s*\.\.\s*(\d+)\]\s+owned\s+\(\s*(\d+)\s+eqs\),\s+\[\s*(\d+)\s*\.\.\s*(\d+)\]\s+overlap\s+\(\s*(\d+)\s+eqs\)\s+\|\s+Total:\s+(\d+)\s+eqs\s*\n"
        r"\s+\|\s+Global Cell IDs:\s+\[(.*?)\]\s+owned,\s+\[(.*?)\]\s+overlap\s*\n"
        r"\s+\|\s+Matrix NNZ:\s+(\d+)\s+blks total\s+\(\s*(\d+)\s+int-to-int,\s+(\d+)\s+int-to-overlap boundary\)\s*\n"
        r"\s+\|\s+Neighbors:\s+\[(.*?)\]"
    )

    matches = re.findall(pattern, content)
    if not matches:
        return None

    ranks_data = []
    for m in matches:
        r_id = int(m[0])
        owned_b, ovlp_b, tot_b = int(m[2]), int(m[5]), int(m[6])
        owned_s, ovlp_s, tot_s = int(m[8]), int(m[11]), int(m[12])
        owned_cells_str = m[13]
        ovlp_cells_str = m[14]
        tot_nnz, int_nnz, bnd_nnz = int(m[15]), int(m[16]), int(m[17])
        nbr_str = m[18]

        owned_cells = parse_ranges_string(owned_cells_str)
        ovlp_cells = parse_ranges_string(ovlp_cells_str)

        # Parse neighbors: e.g. "1 (36 ovlp blks, 39 bnd nnz), 2 (12 ovlp blks, 15 bnd nnz)"
        nbr_dict = {}
        nbr_tokens = re.findall(r"(\d+)\s*\(\s*(\d+)\s+ovlp blks,\s+(\d+)\s+bnd nnz\)", nbr_str)
        for n_id, c_ovlp, c_nnz in nbr_tokens:
            nbr_dict[int(n_id)] = {
                "overlap_blocks": int(c_ovlp),
                "bnd_nnz": int(c_nnz)
            }

        ranks_data.append({
            "rank": r_id,
            "owned_blocks": owned_b,
            "overlap_blocks": ovlp_b,
            "total_blocks": tot_b,
            "owned_scalar": owned_s,
            "overlap_scalar": ovlp_s,
            "total_scalar": tot_s,
            "int_nnz": int_nnz,
            "boundary_nnz": bnd_nnz,
            "total_nnz": tot_nnz,
            "owned_cells": owned_cells,
            "overlap_cells": ovlp_cells,
            "neighbors": nbr_dict
        })

    ranks_data.sort(key=lambda d: d["rank"])
    return ranks_data if ranks_data else None


def get_grid_dimensions(total_cells: int = 300) -> Tuple[int, int, int]:
    """Return grid dimensions (Nx, Ny, Nz) for standard reservoir cases (SPE1 = 10x10x3)."""
    if total_cells == 300:
        return 10, 10, 3
    # Fallback heuristic factorization
    nz = 1
    for k in [3, 4, 5, 10]:
        if total_cells % k == 0:
            nz = k
            break
    nxy = total_cells // nz
    nx = int(np.sqrt(nxy))
    while nxy % nx != 0:
        nx -= 1
    ny = nxy // nx
    return nx, ny, nz


def cell_to_ijk(cell_id: int, nx: int, ny: int) -> Tuple[int, int, int]:
    """Map 1D cell index to 3D grid coordinate (i, j, k)."""
    i = cell_id % nx
    j = (cell_id // nx) % ny
    k = cell_id // (nx * ny)
    return i, j, k


def build_7point_stencil(nx: int, ny: int, nz: int) -> Dict[int, List[int]]:
    """Build standard reservoir 7-point stencil neighbor graph for Cartesian grid."""
    total_cells = nx * ny * nz
    adj = {}
    for c in range(total_cells):
        i, j, k = cell_to_ijk(c, nx, ny)
        nbrs = [c]  # Diagonal self coupling
        if i > 0: nbrs.append(c - 1)
        if i < nx - 1: nbrs.append(c + 1)
        if j > 0: nbrs.append(c - nx)
        if j < ny - 1: nbrs.append(c + nx)
        if k > 0: nbrs.append(c - nx * ny)
        if k < nz - 1: nbrs.append(c + nx * ny)
        adj[c] = nbrs
    return adj


# ==============================================================================
# VISUALIZATION 1: Spatial Domain Decomposition (3D Grid & Layer Slices)
# ==============================================================================
def plot_3d_spatial_decomposition(ranks_data: List[Dict[str, Any]],
                                  output_filepath: Optional[str] = None,
                                  show: bool = False):
    """
    Renders 3D Spatial Domain Decomposition of the reservoir grid colored by MPI rank,
    including an isometric 3D voxel representation and 2D planar layer slices.
    """
    num_ranks = len(ranks_data)
    total_cells = sum(d["owned_blocks"] for d in ranks_data)
    nx, ny, nz = get_grid_dimensions(total_cells)

    # Build mapping from cell_id -> rank
    cell_rank_map = {}
    for d in ranks_data:
        r = d["rank"]
        for c in d["owned_cells"]:
            cell_rank_map[c] = r

    # Create figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 9), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")

    # GridSpec: Left is 3D Voxel plot (large), Right is 3 Layer slice subplots
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1.6, 1.0, 1.0, 1.0], wspace=0.32, hspace=0.35)

    # 1. Main 3D Voxel Plot
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_3d.set_facecolor("#fcfcfc")

    # 3D voxel mask and RGBA matrix
    voxels = np.ones((nx, ny, nz), dtype=bool)
    colors = np.zeros((nx, ny, nz, 4), dtype=float)

    for c in range(total_cells):
        i, j, k = cell_to_ijk(c, nx, ny)
        r = cell_rank_map.get(c, 0)
        colors[i, j, k] = RANK_RGBA[r % len(RANK_RGBA)]

    # Plot 3D voxels
    ax_3d.voxels(voxels, facecolors=colors, edgecolors="#2b2b2b", linewidth=0.35, alpha=0.88)
    ax_3d.set_title(f"3D Reservoir Domain Decomposition\n({total_cells} Active Cells: {nx}x{ny}x{nz})",
                    fontsize=14, fontweight="bold", pad=15)
    ax_3d.set_xlabel("X (I Direction: 0..9)", fontsize=10, labelpad=8)
    ax_3d.set_ylabel("Y (J Direction: 0..9)", fontsize=10, labelpad=8)
    ax_3d.set_zlabel("Z (K Layer: 0..2)", fontsize=10, labelpad=6)
    ax_3d.view_init(elev=28, azim=-55)

    # Legend for ranks placed cleanly below or to the side
    legend_patches = [
        patches.Patch(facecolor=RANK_COLORS[r], edgecolor="black",
                      label=f"Rank {r}: {ranks_data[r]['owned_blocks']} cells ({ranks_data[r]['owned_scalar']} eqs)")
        for r in range(num_ranks)
    ]
    ax_3d.legend(handles=legend_patches, loc="lower left", bbox_to_anchor=(-0.12, 0.02),
                 frameon=True, facecolor="#ffffff", framealpha=0.92, fontsize=9.5)

    # 2. Layer-by-Layer Plan Views (K = 0, K = 1, K = 2)
    layer_names = ["Top Layer (K = 0)", "Middle Layer (K = 1)", "Bottom Layer (K = 2)"]
    for k in range(nz):
        ax_k = fig.add_subplot(gs[k, 1:])
        layer_grid = np.zeros((ny, nx), dtype=int)
        for j in range(ny):
            for i in range(nx):
                c = i + j * nx + k * nx * ny
                layer_grid[j, i] = cell_rank_map.get(c, 0)

        # Draw discrete color mesh
        cmap = plt.matplotlib.colors.ListedColormap(RANK_COLORS[:num_ranks])
        bounds = np.arange(num_ranks + 1) - 0.5
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax_k.imshow(layer_grid, cmap=cmap, norm=norm, origin="lower", aspect="auto", alpha=0.88)

        # Annotate cell numbers & coordinates inside squares
        for j in range(ny):
            for i in range(nx):
                c = i + j * nx + k * nx * ny
                r = layer_grid[j, i]
                txt_color = "black" if r == 1 else "white"
                ax_k.text(i, j, f"{c}", ha="center", va="center", color=txt_color,
                          fontweight="bold", fontsize=7.5)

        # Formatting
        ax_k.set_title(f"Plan View: {layer_names[k]} — Global Cells [{k*nx*ny} .. {(k+1)*nx*ny - 1}]",
                       fontsize=11, fontweight="bold", pad=5)
        ax_k.set_xlabel("I Index (0 .. 9)", fontsize=9, labelpad=2)
        ax_k.set_ylabel("J Index (0 .. 9)", fontsize=9, labelpad=2)
        ax_k.set_xticks(range(nx))
        ax_k.set_yticks(range(ny))
        ax_k.grid(color="white", linestyle="-", linewidth=1.2)

    fig.suptitle("Visualization 1: Spatial Domain Decomposition (Zoltan Graph Partitioning)",
                 fontsize=16, fontweight="bold", y=0.98)

    if output_filepath:
        plt.savefig(output_filepath, bbox_inches="tight")
        print(f"[Visualization 1 Saved] Saved 3D Spatial Decomposition to: {output_filepath}")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# VISUALIZATION 2: MPI Communication Topology (Network Graph)
# ==============================================================================
def plot_communication_topology(ranks_data: List[Dict[str, Any]],
                                output_filepath: Optional[str] = None,
                                show: bool = False):
    """
    Renders MPI Communication Topology network graph where nodes represent MPI processes
    and weighted edges indicate inter-process boundary NNZ volume and overlap cells transferred.
    """
    num_ranks = len(ranks_data)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(13, 10.5), dpi=200)
    fig.patch.set_facecolor("#fbfbfb")
    ax.set_facecolor("#ffffff")

    # Node positions arranged in a balanced layout
    pos = {
        0: np.array([-1.2,  1.0]),
        1: np.array([-1.2, -1.0]),
        2: np.array([ 1.2, -1.0]),
        3: np.array([ 1.2,  1.0]),
    }

    # Collect undirected edge metrics
    edges = {}
    for d in ranks_data:
        u = d["rank"]
        for v, metrics in d["neighbors"].items():
            if u < v:
                bnd_nnz = metrics["bnd_nnz"]
                ovlp_u = metrics["overlap_blocks"]
                ovlp_v = 0
                for d_other in ranks_data:
                    if d_other["rank"] == v and u in d_other["neighbors"]:
                        ovlp_v = d_other["neighbors"][u]["overlap_blocks"]
                edges[(u, v)] = {
                    "bnd_nnz": bnd_nnz,
                    "ovlp_u": ovlp_u,
                    "ovlp_v": ovlp_v,
                }

    max_nnz = max(e["bnd_nnz"] for e in edges.values()) if edges else 1

    # Draw Edges
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")

    for (u, v), info in edges.items():
        p1 = pos[u]
        p2 = pos[v]
        nnz = info["bnd_nnz"]
        ovlp_u = info["ovlp_u"]
        ovlp_v = info["ovlp_v"]

        # Thickness proportional to NNZ
        lw = 2.0 + 5.5 * (nnz / max_nnz)

        # Draw edge line
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#4a5568", linewidth=lw,
                solid_capstyle="round", zorder=2, alpha=0.82)

        # Position label along the edge (avoid center intersection for diagonals)
        if (u == 0 and v == 2):
            mid = 0.68 * p1 + 0.32 * p2  # Offset towards Rank 0
        elif (u == 1 and v == 3):
            mid = 0.68 * p1 + 0.32 * p2  # Offset towards Rank 1
        elif (u == 0 and v == 1):
            mid = (p1 + p2) / 2.0 + np.array([-0.24, 0.0])
        elif (u == 2 and v == 3):
            mid = (p1 + p2) / 2.0 + np.array([0.24, 0.0])
        elif (u == 0 and v == 3):
            mid = (p1 + p2) / 2.0 + np.array([0.0, 0.16])
        elif (u == 1 and v == 2):
            mid = (p1 + p2) / 2.0 + np.array([0.0, -0.16])
        else:
            mid = (p1 + p2) / 2.0

        label_txt = f"{nnz} Bnd NNZ\n({ovlp_u} ↔ {ovlp_v} ovlp)"
        bbox_props = dict(boxstyle="round,pad=0.35", facecolor="#fffaf0", edgecolor="#dd6b20",
                          linewidth=1.2, alpha=0.94)
        ax.text(mid[0], mid[1], label_txt, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#7b341e", bbox=bbox_props, zorder=4)

    # Draw Nodes
    node_radius = 0.38
    for d in ranks_data:
        r = d["rank"]
        p = pos[r]
        color = RANK_COLORS[r]

        circle = plt.Circle(p, node_radius, color=color, ec="#1a202c", lw=2.5, zorder=5)
        ax.add_patch(circle)

        node_text = f"Rank {r}\n{d['owned_blocks']} Cells\n({d['owned_scalar']} Eqs)"
        ax.text(p[0], p[1], node_text, ha="center", va="center", color="white",
                fontweight="bold", fontsize=10.5, zorder=6)

        # Label outside node indicating local matrix dimension
        offset_dir = np.sign(p) * 0.54
        local_info = (f"Local Dimension: {d['total_blocks']}x{d['total_blocks']} blks\n"
                      f"ILU0 Solved Rows: [0 .. {d['owned_blocks']-1}]\n"
                      f"Overlap Sync Rows: [{d['owned_blocks']} .. {d['total_blocks']-1}]")
        ax.text(p[0] + offset_dir[0], p[1] + offset_dir[1] * 0.45, local_info,
                ha="center" if offset_dir[0] == 0 else ("right" if offset_dir[0] < 0 else "left"),
                va="center", fontsize=8.5, fontweight="bold", color="#2d3748",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#edf2f7", edgecolor="#cbd5e0", alpha=0.92),
                zorder=4)

    # Informational subtitle & communication mechanics note
    info_box = (
        "MPI Preconditioner Synchronization Mechanics (ParallelOverlappingILU0):\n"
        "  • Owned rows [0 .. N_owned-1] are factored and solved independently on each rank.\n"
        "  • Overlap rows [N_owned .. N_total-1] hold ghost cells updated via MPI copyOwnerToAll.\n"
        "  • Edge thickness & NNZ volume quantify the inter-process flux exchanged per Newton step."
    )
    ax.text(0.0, -1.60, info_box, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#1a202c", bbox=dict(boxstyle="round,pad=0.5", facecolor="#ebf8ff", edgecolor="#3182ce", lw=1.2),
            zorder=3)

    ax.set_title("Visualization 2: MPI Communication Topology & Interface Flux Volume",
                 fontsize=15, fontweight="bold", pad=20)

    if output_filepath:
        plt.savefig(output_filepath, bbox_inches="tight")
        print(f"[Visualization 2 Saved] Saved Communication Topology to: {output_filepath}")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# VISUALIZATION 3: Global Matrix Sparsity Pattern (Spy Plot)
# ==============================================================================
def plot_global_matrix_sparsity(ranks_data: List[Dict[str, Any]],
                                output_filepath: Optional[str] = None,
                                show: bool = False):
    """
    Renders conceptual 2D spy plot distinguishing and color-coding the 4 main diagonal
    blocks (Owned-to-Owned interactions) and highlighting off-diagonal elements
    (Interior-to-Overlap couplings) representing inter-process fluxes.
    """
    num_ranks = len(ranks_data)
    total_cells = sum(d["owned_blocks"] for d in ranks_data)
    nx, ny, nz = get_grid_dimensions(total_cells)
    adj = build_7point_stencil(nx, ny, nz)

    # Rank partitioning order: Rank 0 cells, Rank 1 cells, Rank 2 cells, Rank 3 cells
    # Map cell_id -> global matrix row index [0 .. total_cells-1]
    cell_to_rank = {}
    ordered_cells = []
    rank_boundaries = [0]

    for d in ranks_data:
        r = d["rank"]
        ordered_cells.extend(d["owned_cells"])
        rank_boundaries.append(len(ordered_cells))
        for c in d["owned_cells"]:
            cell_to_rank[c] = r

    cell_to_row = {c: i for i, c in enumerate(ordered_cells)}

    # Build matrix entries
    diag_rows = {r: [] for r in range(num_ranks)}
    diag_cols = {r: [] for r in range(num_ranks)}
    offdiag_rows = []
    offdiag_cols = []
    offdiag_colors = []

    for c in range(total_cells):
        r_row = cell_to_row[c]
        rank_c = cell_to_rank[c]
        for nb in adj[c]:
            r_col = cell_to_row[nb]
            rank_nb = cell_to_rank[nb]
            if rank_c == rank_nb:
                diag_rows[rank_c].append(r_row)
                diag_cols[rank_c].append(r_col)
            else:
                offdiag_rows.append(r_row)
                offdiag_cols.append(r_col)
                # Assign off-diagonal color based on boundary interface
                offdiag_colors.append("#d90429")

    # Set up plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    # Invert Y axis to match matrix standard indexing (row 0 at top)
    ax.set_xlim(-3, total_cells + 3)
    ax.set_ylim(total_cells + 3, -3)

    # Plot diagonal block interactions (Owned-to-Owned) for each rank
    for r in range(num_ranks):
        ax.scatter(diag_cols[r], diag_rows[r], s=14, color=RANK_COLORS[r], marker="s",
                   label=f"Rank {r} Interior (Owned-to-Owned: {len(diag_rows[r])} NNZ)",
                   alpha=0.9, edgecolors="none")

    # Plot off-diagonal interactions (Interior-to-Overlap couplings)
    ax.scatter(offdiag_cols, offdiag_rows, s=22, color="#d90429", marker="o",
               label=f"Inter-Process Flux (Interior-to-Overlap: {len(offdiag_rows)} NNZ)",
               alpha=0.95, edgecolors="#67000d", linewidths=0.4, zorder=5)

    # Draw partition boundary lines & background block highlights
    for b in rank_boundaries[1:-1]:
        ax.axhline(b - 0.5, color="#4a5568", linestyle="--", linewidth=1.2, zorder=3)
        ax.axvline(b - 0.5, color="#4a5568", linestyle="--", linewidth=1.2, zorder=3)

    # Draw rectangle borders around diagonal blocks
    for r in range(num_ranks):
        start = rank_boundaries[r]
        size = rank_boundaries[r + 1] - start
        rect = patches.Rectangle((start - 0.5, start - 0.5), size, size,
                                 linewidth=1.8, edgecolor=RANK_COLORS[r], facecolor=RANK_COLORS[r],
                                 alpha=0.08, zorder=1)
        ax.add_patch(rect)

        # Label inside diagonal block
        center = start + size / 2.0
        ax.text(center, center - size * 0.30, f"Rank {r} Local ILU0\n({size}x{size} blks)",
                ha="center", va="center", color=RANK_COLORS[r], fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=RANK_COLORS[r], alpha=0.85))

    # Axis ticks and annotations
    tick_locs = [rank_boundaries[r] + (rank_boundaries[r+1] - rank_boundaries[r]) / 2.0 for r in range(num_ranks)]
    tick_labels = [f"Rank {r}\n[{rank_boundaries[r]}..{rank_boundaries[r+1]-1}]" for r in range(num_ranks)]

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels, fontsize=10, fontweight="bold")

    ax.set_xlabel("Matrix Column Index (Block Unknowns: 0 .. 299)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Matrix Row Index (Block Equations: 0 .. 299)", fontsize=11, fontweight="bold", labelpad=10)

    # Title & Legend
    total_nnz = sum(len(diag_rows[r]) for r in range(num_ranks)) + len(offdiag_rows)
    ax.set_title(f"Visualization 3: Global System Matrix Sparsity Pattern (Spy Plot)\n"
                 f"Dimension: {total_cells}x{total_cells} 3x3 Blocks ({total_cells*3}x{total_cells*3} Scalar Eqs) | Total NNZ: {total_nnz} Blocks",
                 fontsize=13, fontweight="bold", pad=15)

    ax.legend(loc="lower right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
              framealpha=0.95, fontsize=9.5)

    if output_filepath:
        plt.savefig(output_filepath, bbox_inches="tight")
        print(f"[Visualization 3 Saved] Saved Global Matrix Sparsity Pattern to: {output_filepath}")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# COMPOSITE DASHBOARD: All 3 Visualizations Combined
# ==============================================================================
def plot_composite_dashboard(ranks_data: List[Dict[str, Any]],
                             output_filepath: str,
                             show: bool = False):
    """Generates an all-in-one high-resolution multi-panel dashboard containing all three visualizations."""
    num_ranks = len(ranks_data)
    total_cells = sum(d["owned_blocks"] for d in ranks_data)
    nx, ny, nz = get_grid_dimensions(total_cells)
    adj = build_7point_stencil(nx, ny, nz)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(24, 14), dpi=200)
    fig.patch.set_facecolor("#fbfbfb")

    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: 3D Spatial Domain Decomposition
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor("#fbfbfb")

    cell_rank_map = {}
    for d in ranks_data:
        r = d["rank"]
        for c in d["owned_cells"]:
            cell_rank_map[c] = r

    voxels = np.ones((nx, ny, nz), dtype=bool)
    colors = np.zeros((nx, ny, nz, 4), dtype=float)
    for c in range(total_cells):
        i, j, k = cell_to_ijk(c, nx, ny)
        colors[i, j, k] = RANK_RGBA[cell_rank_map.get(c, 0) % len(RANK_RGBA)]

    ax1.voxels(voxels, facecolors=colors, edgecolors="#2b2b2b", linewidth=0.35, alpha=0.88)
    ax1.set_title("1. Spatial Domain Decomposition (3D Grid: 10x10x3)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("I Direction (0..9)", fontsize=9.5)
    ax1.set_ylabel("J Direction (0..9)", fontsize=9.5)
    ax1.set_zlabel("K Layer (0..2)", fontsize=9.5)
    ax1.view_init(elev=28, azim=-55)

    legend_patches = [
        patches.Patch(facecolor=RANK_COLORS[r], edgecolor="black",
                      label=f"Rank {r}: {ranks_data[r]['owned_blocks']} cells ({ranks_data[r]['owned_scalar']} eqs)")
        for r in range(num_ranks)
    ]
    ax1.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0.0, 0.98),
               frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 2: MPI Communication Topology (Network Graph)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#ffffff")
    ax2.set_xlim(-2.1, 2.1)
    ax2.set_ylim(-1.6, 1.6)
    ax2.axis("off")

    pos = {
        0: np.array([-1.1,  0.9]),
        1: np.array([-1.1, -0.9]),
        2: np.array([ 1.1, -0.9]),
        3: np.array([ 1.1,  0.9]),
    }

    edges = {}
    for d in ranks_data:
        u = d["rank"]
        for v, metrics in d["neighbors"].items():
            if u < v:
                edges[(u, v)] = {
                    "bnd_nnz": metrics["bnd_nnz"],
                    "ovlp_u": metrics["overlap_blocks"],
                    "ovlp_v": ranks_data[v]["neighbors"][u]["overlap_blocks"] if u in ranks_data[v]["neighbors"] else 0
                }

    max_nnz = max(e["bnd_nnz"] for e in edges.values()) if edges else 1

    for (u, v), info in edges.items():
        p1 = pos[u]
        p2 = pos[v]
        nnz = info["bnd_nnz"]
        lw = 1.8 + 4.5 * (nnz / max_nnz)
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#4a5568", linewidth=lw, solid_capstyle="round", zorder=2)

        mid = (p1 + p2) / 2.0
        if (u == 0 and v == 2):
            mid = 0.68 * p1 + 0.32 * p2
        elif (u == 1 and v == 3):
            mid = 0.68 * p1 + 0.32 * p2
        elif (u == 0 and v == 1):
            mid = mid + np.array([-0.22, 0.0])
        elif (u == 2 and v == 3):
            mid = mid + np.array([0.22, 0.0])
        elif (u == 0 and v == 3):
            mid = mid + np.array([0.0, 0.15])
        elif (u == 1 and v == 2):
            mid = mid + np.array([0.0, -0.15])

        ax2.text(mid[0], mid[1], f"{nnz} NNZ\n({info['ovlp_u']}↔{info['ovlp_v']} ovlp)",
                 ha="center", va="center", fontsize=8, fontweight="bold", color="#7b341e",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffaf0", edgecolor="#dd6b20", lw=1.1, alpha=0.92),
                 zorder=4)

    for d in ranks_data:
        r = d["rank"]
        p = pos[r]
        circle = plt.Circle(p, 0.32, color=RANK_COLORS[r], ec="#1a202c", lw=2.2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(p[0], p[1], f"Rank {r}\n{d['owned_blocks']} Cells\n({d['owned_scalar']} Eqs)",
                 ha="center", va="center", color="white", fontweight="bold", fontsize=9.5, zorder=6)

    ax2.set_title("2. MPI Communication Topology & Interface Flux Volume", fontsize=13, fontweight="bold", pad=12)

    # -------------------------------------------------------------
    # Panel 3: Global Matrix Sparsity Pattern (Spy Plot)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#ffffff")
    ax3.set_xlim(-3, total_cells + 3)
    ax3.set_ylim(total_cells + 3, -3)

    cell_to_row = {}
    ordered_cells = []
    rank_boundaries = [0]
    for d in ranks_data:
        ordered_cells.extend(d["owned_cells"])
        rank_boundaries.append(len(ordered_cells))
    for i, c in enumerate(ordered_cells):
        cell_to_row[c] = i

    diag_rows = {r: [] for r in range(num_ranks)}
    diag_cols = {r: [] for r in range(num_ranks)}
    offdiag_rows, offdiag_cols = [], []

    for c in range(total_cells):
        r_row = cell_to_row[c]
        rc = cell_rank_map[c]
        for nb in adj[c]:
            r_col = cell_to_row[nb]
            rnb = cell_rank_map[nb]
            if rc == rnb:
                diag_rows[rc].append(r_row)
                diag_cols[rc].append(r_col)
            else:
                offdiag_rows.append(r_row)
                offdiag_cols.append(r_col)

    for r in range(num_ranks):
        ax3.scatter(diag_cols[r], diag_rows[r], s=12, color=RANK_COLORS[r], marker="s",
                    label=f"Rank {r} Owned ({len(diag_rows[r])} NNZ)", alpha=0.9, edgecolors="none")

    ax3.scatter(offdiag_cols, offdiag_rows, s=18, color="#d90429", marker="o",
                label=f"Inter-Process Flux ({len(offdiag_rows)} NNZ)", alpha=0.95, edgecolors="#67000d",
                linewidths=0.3, zorder=5)

    for b in rank_boundaries[1:-1]:
        ax3.axhline(b - 0.5, color="#718096", linestyle="--", linewidth=1.0)
        ax3.axvline(b - 0.5, color="#718096", linestyle="--", linewidth=1.0)

    for r in range(num_ranks):
        st = rank_boundaries[r]
        sz = rank_boundaries[r + 1] - st
        rect = patches.Rectangle((st - 0.5, st - 0.5), sz, sz, linewidth=1.5,
                                 edgecolor=RANK_COLORS[r], facecolor=RANK_COLORS[r], alpha=0.08)
        ax3.add_patch(rect)

    tick_locs = [rank_boundaries[r] + (rank_boundaries[r+1] - rank_boundaries[r]) / 2.0 for r in range(num_ranks)]
    ax3.set_xticks(tick_locs)
    ax3.set_xticklabels([f"Rank {r}" for r in range(num_ranks)], fontsize=9, fontweight="bold")
    ax3.set_yticks(tick_locs)
    ax3.set_yticklabels([f"Rank {r}" for r in range(num_ranks)], fontsize=9, fontweight="bold")
    ax3.set_xlabel("Matrix Unknown Index (0 .. 299)", fontsize=10)
    ax3.set_ylabel("Matrix Equation Index (0 .. 299)", fontsize=10)
    ax3.set_title("3. Global Matrix Sparsity Pattern (Spy Plot)", fontsize=13, fontweight="bold", pad=12)
    ax3.legend(loc="lower right", frameon=True, fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 4: Partitioning & Workload Statistics Table
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("4. Workload Breakdown & Communication Volume Summary", fontsize=13, fontweight="bold", pad=12)

    table_data = []
    col_labels = ["Rank", "Owned Blks", "Overlap Blks", "Scalar Eqs", "Int NNZ", "Bnd NNZ", "Neighbors & Interfaces"]
    for d in ranks_data:
        r = d["rank"]
        nbr_info = ", ".join([f"R{nbr} ({info['bnd_nnz']} bnd)" for nbr, info in sorted(d["neighbors"].items())])
        table_data.append([
            f"Rank {r}",
            f"{d['owned_blocks']}",
            f"{d['overlap_blocks']}",
            f"{d['owned_scalar']} own / {d['overlap_scalar']} ovlp",
            f"{d['int_nnz']}",
            f"{d['boundary_nnz']}",
            nbr_info
        ])

    table = ax4.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center",
                      colWidths=[0.09, 0.10, 0.11, 0.19, 0.09, 0.09, 0.33],
                      bbox=[0.01, 0.15, 0.98, 0.70])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            rank_idx = row - 1
            if col == 0:
                cell.set_facecolor(RANK_COLORS[rank_idx])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f7fafc" if row % 2 == 0 else "#edf2f7")

    fig.suptitle("OPM Flow — MPI Linear System Matrix Partitioning & Communication Analysis",
                 fontsize=17, fontweight="bold", y=0.98)

    plt.savefig(output_filepath, bbox_inches="tight")
    print(f"[Composite Dashboard Saved] Output saved to: {output_filepath}")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# TERMINAL SUMMARY
# ==============================================================================
def print_summary_table(ranks_data: List[Dict[str, Any]]):
    """Print clean terminal table with exact ranges and counts."""
    print("\n" + "=" * 95)
    print("                    OPM FLOW — MPI MATRIX SPLITTING SUMMARY")
    print("=" * 95)
    print(f"{'Rank':<6} | {'Owned (Solved)':<16} | {'Overlap (Ghost)':<16} | {'Scalar Eqs':<18} | {'NNZ (Int/Bnd)':<15} | {'Neighbors'}")
    print("-" * 95)
    total_owned_b = 0
    total_owned_s = 0
    total_int_nnz = 0
    total_bnd_nnz = 0
    for d in ranks_data:
        r = d["rank"]
        ob, ovb = d["owned_blocks"], d["overlap_blocks"]
        os_cnt, ovs = d["owned_scalar"], d["overlap_scalar"]
        total_owned_b += ob
        total_owned_s += os_cnt
        total_int_nnz += d["int_nnz"]
        total_bnd_nnz += d["boundary_nnz"]

        nbr_str = ", ".join([f"R{n}({info['bnd_nnz']} bnd)" for n, info in sorted(d["neighbors"].items())])
        print(f"Rank {r:<2} | {ob:>3} blks [0..{ob-1:<2}]  | {ovb:>3} blks [{ob:<2}..{ob+ovb-1:<2}] | {os_cnt:>3} own / {ovs:>3} ovlp | {d['int_nnz']:>3} int / {d['boundary_nnz']:>2} bnd | [{nbr_str}]")

    print("-" * 95)
    print(f"TOTAL: {total_owned_b} Active Reservoir Cells | {total_owned_s} Scalar Equations Across All Ranks")
    print(f"COUPLINGS: {total_int_nnz} Interior NNZ (Local Solves) | {total_bnd_nnz} Boundary NNZ (MPI Inter-Rank Fluxes)")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize MPI Linear System Matrix Partitioning in OPM Flow.")
    parser.add_argument("--input", "-i", default="out.txt", help="Simulation log file to parse (default: out.txt)")
    parser.add_argument("--output", "-o", default="mpi_partitioning_dashboard.png", help="Output composite PNG image path")
    parser.add_argument("--save-all", action="store_true", help="Save separate PNG figures for each of the 3 visualizations")
    parser.add_argument("--show", "-s", action="store_true", help="Display interactive figure window")

    args = parser.parse_args()

    parsed_data = parse_simulation_log(args.input)
    if parsed_data:
        print(f"[Log Parser] Successfully parsed partitioning data for {len(parsed_data)} MPI ranks from '{args.input}'.")
        data = parsed_data
    else:
        print(f"[Log Parser] Could not find matrix partitioning block in '{args.input}'. Using exact SPE1 benchmark data.")
        data = FALLBACK_RANKS_DATA

    print_summary_table(data)

    # Always generate composite dashboard
    plot_composite_dashboard(data, args.output, args.show)

    # If --save-all requested, generate individual high-res figures for each visualization
    if args.save_all:
        plot_3d_spatial_decomposition(data, output_filepath="mpi_3d_spatial_decomposition.png", show=args.show)
        plot_communication_topology(data, output_filepath="mpi_communication_topology.png", show=args.show)
        plot_global_matrix_sparsity(data, output_filepath="mpi_global_matrix_sparsity.png", show=args.show)
        print("\n[All Visualizations Generated]:")
        print("  1. mpi_3d_spatial_decomposition.png (3D Reservoir Voxel Grid & Layer Slices)")
        print("  2. mpi_communication_topology.png (MPI Process Network Graph & Flux Couplings)")
        print("  3. mpi_global_matrix_sparsity.png (2D Matrix Spy Plot of Diagonal & Off-Diagonal Blocks)")
        print(f"  4. {args.output} (Composite 4-panel Dashboard)")


if __name__ == "__main__":
    main()
