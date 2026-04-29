import pypsa

def create_network(snapshots):
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    return network