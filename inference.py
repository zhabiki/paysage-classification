import matplotlib.pyplot as plt
from typing import Callable, Literal
from PIL import Image

from preprocess import process_image
   

def inference(
    model,
    path_image,
    labels_list: list[str],
    transforms: Callable,
    output: Literal['mpl', 'std']
):
    tensor = process_image(path_image, transforms)
    if tensor is None:
        return
    tensor = tensor.unsqueeze(0)

    (preds, probas) = model.predict(tensor)
    
    if output == 'std':
        print(labels_list[preds.tolist()[0]])
    else:
        with Image.open(path_image) as img:
            img_rgb = img.convert('RGB') # На всякий случай

            probas_with_labels = ', '.join(
                [f'{labels_list[i]} - {p*100:.2f}%' for i, p in enumerate(probas.tolist()[0])]
            )

            plt.imshow(img_rgb)
            plt.suptitle(labels_list[preds.tolist()[0]])
            plt.title(probas_with_labels, size='small', wrap=True)

            plt.show()
