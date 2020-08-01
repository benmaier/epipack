if __name__=="__main__":
    import netwulf as nw
    import networkx as nx

    k0 = 20
    N = 5000
    p = k0 / (N-1)

    G = nx.fast_gnp_random_graph(N, p)

    network, config  = nw.visualize(G)

    nw.save('./random_network.json',network, config)

