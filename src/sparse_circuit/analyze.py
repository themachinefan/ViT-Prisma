import torch












if __name__ == "__main__":

    # load the output from demo.py 
    parsed_nodes_path =  r"F:\ViT-Prisma_fork\data\circuit_output\testing_stuff_2_nodes_parsed.pt"
    edges_path = r"F:\ViT-Prisma_fork\data\circuit_output\testing_stuff_2_edges.pt"
    nodes_indices_loaded, nodes_values_loaded = torch.load(parsed_nodes_path)
    edges_loaded= torch.load(edges_path)

    print(nodes_indices_loaded, nodes_values_loaded)