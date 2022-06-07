from load_graph import load_graph_from_db, load_graph_from_file

if __name__ == '__main__':
    db_name = 'linajea_120828_gt_side_2'
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    filename = '120828/tracks/tracks_side_2.txt'

    db_graph = load_graph_from_db(db_name, db_host, 0, 450)
    file_graph = load_graph_from_file(filename)

    db_nodes_set = set(db_graph.nodes)
    db_edges_set = set(db_graph.edges)
    file_nodes_set = set(file_graph.nodes)
    file_edges_set = set(file_graph.edges)
    print("%d nodes in db, %d nodes in file",
          len(db_nodes_set), len(file_nodes_set))
    print("nodes in db not in file")
    print(db_nodes_set - file_nodes_set)
    print("nodes in file not in db")
    print(file_nodes_set - db_nodes_set)
    assert db_nodes_set == file_nodes_set

    print("%d edges in db, %d edges in file",
          len(db_edges_set), len(file_edges_set))
    print("%d edges in db, %d edges in file",
          len(db_edges_set), len(file_edges_set))
    print("edges in db not in file")
    print(db_edges_set - file_edges_set)
    print("edges in file not in db")
    print(file_edges_set - db_edges_set)
    assert db_graph.edges == file_graph.edges
