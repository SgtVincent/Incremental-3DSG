from .gcn import GCN

def parse_model(model_name, params=None):
    # TODO: add more models
    if model_name == "GCN":
        # return GCN(**params)
        return GCN(num_node_features=25,
                   num_classes=21)
    else:
        raise NotImplementedError