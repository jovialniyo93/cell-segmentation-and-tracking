import cv2
import numpy as np
import os

# Custom Vertex and Graph classes for graph-based tracking
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices += 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

# Helper function to calculate cell centers
def cell_center(seg_img):
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_x, all_points_y = np.where(seg_img == label)
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label] = [avg_x, avg_y]
    return results

# Use custom Graph class for cell location graph
def compute_cell_location(seg_img):
    g = Graph()
    centers = cell_center(seg_img)
    all_labels = np.unique(seg_img)

    # Create vertices
    for i in all_labels:
        if i != 0:
            g.add_vertex(i)

    # Create edges based on distances between cell centers
    for i in all_labels:
        if i != 0:
            for j in all_labels:
                if j != 0 and i != j:
                    pos1 = centers[i]
                    pos2 = centers[j]
                    distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                    g.add_edge(i, j, cost=distance)

    return g

# Main tracking function, structured like the overlap model
def predict_dataset_2(path, output_path, threshold=0.15):

    # Check if path exists
    if not os.path.isdir(path):
        print('Input path is not a valid path')
        return

    names = os.listdir(path)
    names = [name for name in names if '.tif' in name and 'predict' in name]
    print(names)
    names.sort()

    img = cv2.imread(os.path.join(path, names[0]), -1)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}

    old_img = np.zeros((mi, ni))
    index = 1

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)
        labels = np.unique(img)[1:]

        g1 = compute_cell_location(old_img)
        g2 = compute_cell_location(img)

        parent_cells = []

        for label in labels:
            mask = (img == label) * 1
            mask_size = np.sum(mask)

            candidates = np.unique(mask * old_img)[1:]
            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                score = np.sum((mask * old_img) == candidate) / mask_size
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # No parent cell detected, create a new track
                records[index] = [i, i, 0]
                result += (mask * index).astype(np.uint16)
                index += 1
            else:
                if max_candidate not in parent_cells:
                    # Prolonging track
                    records[max_candidate][1] = i
                    result += (mask * max_candidate).astype(np.uint16)
                else:
                    # Division detected, assign a new track ID
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        m_mask = (result == max_candidate)
                        result -= (m_mask * max_candidate).astype(np.uint16)
                        result += (m_mask * index).astype(np.uint16)

                        records[index] = [i, i, max_candidate]
                        index += 1

                    records[index] = [i, i, max_candidate]
                    result += (mask * index).astype(np.uint16)
                    index += 1

                parent_cells.append(max_candidate)

        # Store result for the current frame
        cv2.imwrite(os.path.join(output_path, name), result.astype(np.uint16))
        old_img = result

    # Store tracking data in the same format as the overlap model
    print('Generating the tracking file.')
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write(f'{key} {records[key][0]} {records[key][1]} {records[key][2]}\n')

    print("Tracking complete!")

if __name__ == "__main__":
    predict_result = "data/res_result/"
    track_result = "data/track_result/"
    predict_dataset_2(predict_result, track_result)
