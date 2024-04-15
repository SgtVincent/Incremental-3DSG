import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from pathlib import Path
import torch.nn.functional as functional

from .scannet_data_utils import *
from .graph_utils import create_graph_scannet

# class to process and save raw ScanNet dataset
class RawScanNetDataset(Dataset):
    def __init__(self,
                 root,
                 scenes,
                 transform=None,
                 pre_transform=None,
                 word_embedding_path=None,
                 scene_type_mapping_file=None,
                 type_label_mapping_file=None):

        # attributes need to be set before calling init function of parent class
        self.scenes = scenes
        self.scans_dir = Path(root) / "scans"
        self.word_embedding_path = word_embedding_path
        self.scene_type_mapping_file = scene_type_mapping_file
        self.type_label_mapping_file = type_label_mapping_file

        super(RawScanNetDataset, self).__init__(root, transform, pre_transform)


    @property
    def processed_dir(self):
        return Path(self.root) / "graphs"

    @property
    def processed_file_names(self):
        return [f'{scene}.pt' for scene in self.scenes]

    def process(self):
        i = 0
        assert(os.path.exists(self.word_embedding_path))
        assert(os.path.exists(self.scene_type_mapping_file))
        assert(os.path.exists(self.type_label_mapping_file))

        self.scene_classes, self.num_class = read_scene_labels(self.scenes,
            self.scene_type_mapping_file, self.type_label_mapping_file)

        for i in range(len(self.scenes)):
            scene = self.scenes[i]
            scene_class = self.scene_classes[i]
            try:
                features, edges, edge_weights, positions = create_graph_scannet(
                    self.scans_dir, scene, self.word_embedding_path)

                data = Data(
                    x=torch.tensor(features, dtype=torch.float),
                    edge_index=torch.tensor(edges, dtype=torch.long),
                    edge_attr=torch.tensor(edge_weights, dtype=torch.float),
                    pos=torch.tensor(positions, dtype=torch.float),
                    y=torch.tensor([scene_class], dtype=torch.long),
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'{scene}.pt'))
            except Exception as e:
                print(f"Graph creation for scene {scene} failed!")
                print(e)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        scene = self.scenes[idx]
        data = torch.load(os.path.join(self.processed_dir, f'{scene}.pt'))
        return data

class ScanNetDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 scenes,
                 split="train",
                 transform=None,
                 pre_transform=None,
                 num_classes=21):

        self.scenes = scenes
        self.graph_dir = Path(root) / "graphs"
        self.split = split
        self._num_classes = num_classes
        super(ScanNetDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return self.graph_dir / self.split

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def num_classes(self):
        return self._num_classes

    def process(self):
        data_list = []
        for scene in self.scenes:

            graph_file = self.graph_dir / f'{scene}.pt'
            assert(os.path.exists(graph_file))
            data = torch.load(self.graph_dir / f'{scene}.pt')

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

