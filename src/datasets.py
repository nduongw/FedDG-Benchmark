import os
import torch
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.wilds_dataset import WILDSSubset

class FourierSubset(WILDSSubset):
    def __getitem__(self, idx):
        x, y, metadata, amp, pha = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, metadata, amp, pha


class FourierIwildCam(IWildCamDataset):
    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        amp, pha = self.get_fourier(idx)
        return x, y, [metadata, amp, pha]

    def get_fourier(self, idx):
        path = os.path.join(self._data_dir, 'fourier/')
        amp = torch.load(os.path.join(path, "amp_{}.pt".format(idx)))
        pha = torch.load(os.path.join(path, "pha_{}.pt".format(idx)))
        return amp, pha

