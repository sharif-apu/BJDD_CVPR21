#normMean = [0.485, 0.456, 0.406]
#normStd = [0.229, 0.224, 0.225]
normMean = [0.5, 0.5, 0.5]
normStd = [0.5, 0.5, 0.5]

class UnNormalize(object):
    def __init__(self):


        self.std = normStd
        self.mean = normMean

    def __call__(self, tensor, imageNetNormalize=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if imageNetNormalize:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        else:
            tensor = (tensor * 0.5) + 0.5
        
        return tensor