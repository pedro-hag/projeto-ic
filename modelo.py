import cv2
from maskterial import MaskTerial, load_models
from maskterial.structures import Flake
import torch
from IPython.display import display
import matplotlib.pyplot as plt

def display_results(
    image: np.ndarray,
    flakes: list[Flake],
    colors: list[tuple[int, int, int]] = [(255, 0, 0),(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 41, 255),],
):
    for flake in flakes:
        mask = flake.mask.astype(np.uint8)
        class_id = int(flake.thickness)

        # Draw outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colors[class_id], 2)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(mask)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[class_id], 2)

        # Add class label
        label = f"Class {class_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Adjust text position to keep it within bounds
        text_y = y - 5 if y - text_height - 10 >= 0 else y + h + text_height + 5
        bg_y1 = text_y - text_height - 5 if y - text_height - 10 >= 0 else y + h
        bg_y2 = text_y + 5 if y - text_height - 10 >= 0 else y + h + text_height + 10

        # Draw background rectangle for text
        cv2.rectangle(image, (x, bg_y1), (x + text_width, bg_y2), colors[class_id], -1)

        # Draw text
        cv2.putText(
            image, label, (x, text_y), font, font_scale, (255, 255, 255), thickness
        )

    fig, axis = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
    plt.imshow(image[:, :, ::-1])
    plt.axis("off")
    plt.show()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/content/maskterial_repo/demo/images/teste/" #mudar o diretório

SEG_MODEL = "M2F"
SEG_MODEL_ROOT = "/content/models/segmentation_models/M2F/GrapheneH" #mudar o diretorio

CLS_MODEL = "AMM"
CLS_MODEL_ROOT = "/content/models/classification_models/AMM/GrapheneH" #mudar o diretorio

PP_MODEL = None
PP_MODEL_ROOT = None

SCORE_THRESHOLD = 0.1
MIN_CLASS_OCCUPANCY = 0.5
SIZE_THRESHOLD = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/content/maskterial_repo/demo/images/teste/" #mudar o diretorio

SEG_MODEL = "M2F"
SEG_MODEL_ROOT = "/content/models/segmentation_models/M2F/GrapheneH"#mudar o diretorio

CLS_MODEL = "AMM"
CLS_MODEL_ROOT = "/content/models/classification_models/AMM/GrapheneH"#mudar o diretorio

PP_MODEL = None
PP_MODEL_ROOT = None

SCORE_THRESHOLD = 0.1
MIN_CLASS_OCCUPANCY = 0.5
SIZE_THRESHOLD = 200

segmentation_model, classification_model, postprocessing_model = load_models(
    seg_model_type=SEG_MODEL,
    seg_model_root=SEG_MODEL_ROOT,
    cls_model_type=CLS_MODEL,
    cls_model_root=CLS_MODEL_ROOT,
    pp_model_type=PP_MODEL,
    pp_model_root=PP_MODEL_ROOT,
    device=DEVICE,
)

predictor = MaskTerial(
    segmentation_model=segmentation_model,
    classification_model=classification_model,
    postprocessing_model=postprocessing_model,
    score_threshold=SCORE_THRESHOLD,
    min_class_occupancy=MIN_CLASS_OCCUPANCY,
    size_threshold=SIZE_THRESHOLD,
    device=DEVICE,
)