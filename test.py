import torch
import random
import cv2

Ia_path = self.pair[idx][0]
Ia_data = cv2.imread(Ia_path)
Ia_data = cv2.resize(Ia_data, (Ib_data.shape[1], Ib_data.shape[0]))
Ia_data = cv2.resize(Ia_data, (256, 256))
Ia_data = Ia_data.transpose((2, 0, 1)).astype(np.float32)
Ia_data /= 255

label = torch.tensor([real_label for _ in range(condition_Ia.shape[0])])
label = torch.tensor([random.uniform(0.0, 0.3) for _ in range(condition_Ia.shape[0])])