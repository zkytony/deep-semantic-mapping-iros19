# Generate node_room mappings for all topological mapss
import csv
import os
import re
import copy
import numpy as np
import json
from matplotlib.path import Path
from deepsm.experiments.common import DGSM_DB_ROOT, TOPO_MAP_DB_ROOT, GROUNDTRUTH_ROOT, COLD_ROOT

def get_room_outlines(db_name):
    outlines = {}
    building = re.search("(stockholm|freiburg|saarbrucken)", db_name, re.IGNORECASE)
    if building is None:
        return None
    building = building.group().capitalize()
    
    for floor in os.listdir(os.path.join(GROUNDTRUTH_ROOT, building, "groundtruth")):
        outlines[floor] = {}
        with open(os.path.join(GROUNDTRUTH_ROOT, building, "groundtruth", floor, "labels.json")) as f:
            rooms = json.load(f)['rooms']

            for room_id in rooms:
                outlines[floor][room_id] = rooms[room_id]['outline']
    return outlines

def is_inside_outline(room, x, y):
    """Check whether given point is inside room outline
    From sara_virtual_scan_extractor.Annotator, written
    by Kousuke Ariga."""

    # Edges look something like:
    #     [[[x1, y1, z1], [x2, y2, z2]],
    #      [[x3, y3, z3], [x2, y2, z2]],
    #      ...,
    #      [[xk, yk, zk], [xn, yn, zn]]]
    #
    # Therefore, an edge has a form:
    #      [[x1, y1, z1], [x2, y2, z2]]

    edges = copy.deepcopy(room)
    for edge in edges:
        edge[0] = edge[0][:2]
        edge[1] = edge[1][:2]

    # Now edges look like:
    #      [[x1, y1], [x2, y2]]

    # Sort the edges so that it will be in order of a path. The direction
    # of the path (i.e. clockwise or not) doesn't matter. Put one edge in
    # edges in the front of sorted_edges. This edge is the starting point
    # of path search.
    sorted_edges = [edges[0]]
    edges.remove(edges[0])

    # Until all edge in the edges move to sorted_edges
    while edges:
        # Suppose the last edge in the sorted_edges look like:
        #     [[[xi, yi], [xj, yj]]
        #      ...,
        #      [[xk, yk], [xl, yl]]]
        #
        # For each edge, if the edge is,
        for edge in edges:
            #  [[xl, yl], [xm, ym]]]
            # Append the edge to the end of the sored edges
            if edge[0] == sorted_edges[-1][1]:
                sorted_edges.append(edge)
                edges.remove(edge)
            #  [[xm, ym], [xl, yl]]]
            # Swap the edge and append to the end of the sored edges
            elif edge[1] == sorted_edges[-1][1]:
                swapped_edge = [edge[1], edge[0]]
                sorted_edges.append(swapped_edge)
                edges.remove(edge)
            #  [[xm, ym], [xi, yi]]]
            # Insert the edge in the front of the sored edges
            elif edge[1] == sorted_edges[0][0]:
                sorted_edges.insert(0, edge)
                edges.remove(edge)
            #  [[xi, yi], [xm, ym]]]
            # Swap the edge and insert in the front of the sored edges
            elif edge[0] == sorted_edges[0][0]:
                swapped_edge = [edge[1], edge[0]]
                sorted_edges.insert(0, swapped_edge)
                edges.remove(edge)

    # sorted_edges now looks like:
    #     [[[x1, y1], [x2, y2]],
    #      [[x2, y2], [x3, y3]],
    #      ...,
    #      [[xn-1, yn-1], [xn, yn]]]

    outline = []
    for edge in sorted_edges:
        outline.append(edge[0])

    # outline looks like:
    #     [[x1, y1],
    #      [x2, y2],
    #      ...,
    #      [xn-1, yn-1]]
    #
    # Append [xn, yn] to outline
    outline.append(sorted_edges[-1][1])

    p = Path(np.array(outline))

    return p.contains_point((x, y))

def assign_node_to_room(x, y, outlines, doorway_label="DW"):
    # Below is the same procedure as sara_cold_processing/cold_gen_labels.py,
    # where the labels were initially assigned
    unique_id_using = None
    for unique_id in outlines:  # unique_id is room id of format X-Y-Z
        if is_inside_outline(outlines[unique_id], x, y):
            _, Y, _ = unique_id.split("-") # X Y Z are: floor# abbrev ID
            if unique_id_using is None:
                unique_id_using = unique_id
            else:
                # If the pose is already annotated to some room, but it
                # can also be annotated as doorway, doorway overrides.
                if Y == doorway_label:
                    unique_id_using = unique_id
                    break
            # We prioritize doorway. Break the loop if the pose
            # is already annotated as doorway.
            if Y == doorway_label:
                break
    return unique_id_using
        

def process_one_sequence(seq_id, path_to_node_dat, outlines, save_path):

    node_room_mapping = {}
    floor = seq_id.split("_")[0]
    nodes_to_recover = []
    with open(path_to_node_dat) as f:
        nodes_data_raw = csv.reader(f, delimiter=' ')
        for row in nodes_data_raw:
            nid = int(row[0])
            x, y = tuple(map(float, row[2:4]))
            room_id = assign_node_to_room(x, y, outlines[floor])
            label = row[8]

            # Incase room_id is None, we leverage the groundtruth class stored in nodes.dat
            # to recover a room_id (either the one before or after)
            if room_id is None:
                nodes_to_recover.append((nid, label))
            
            node_room_mapping[nid] = room_id

    for nid, label in nodes_to_recover:

        try_nid = nid-1
        while try_nid in node_room_mapping:
            r = node_room_mapping[try_nid]
            if r is not None:
                _, c, _ = r.split("-")
                if c == label:
                    node_room_mapping[nid] = r
                    break
            try_nid -= 1

        if node_room_mapping[nid] is None:
            try_nid = nid+1
            while try_nid in node_room_mapping:
                r = node_room_mapping[try_nid]
                if r is not None:
                    _, c, _ = r.split("-")
                    if c == label:
                        node_room_mapping[nid] = r
                        break
                try_nid += 1

        if node_room_mapping[nid] is None:
            if label == "UN":
                # Unknown background. A placeholder. See (spn_topo/prepare_dataset.py).
                node_room_mapping[nid] = '0-UN-0'
                continue
            raise Exception("%d cannot be mapped to a room!" % nid)
    
    with open(save_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for nid in sorted(node_room_mapping):
            writer.writerow([nid, node_room_mapping[nid]])



if __name__ == "__main__":
    
    for db_name in os.listdir(TOPO_MAP_DB_ROOT):
        print("Processing %s" % db_name)
        outlines = get_room_outlines(db_name)
        if outlines is None:
            print("Skipping %s" % db_name)
            continue
        
        for seq_id in os.listdir(os.path.join(TOPO_MAP_DB_ROOT, db_name)):
            path_node_dat = os.path.join(TOPO_MAP_DB_ROOT, db_name, seq_id, "nodes.dat")
            save_path = os.path.join(TOPO_MAP_DB_ROOT, db_name, seq_id, "rooms.dat")
            print("   Processing %s" % seq_id)
            process_one_sequence(seq_id, path_node_dat, outlines, save_path)
    print("DONE!")
