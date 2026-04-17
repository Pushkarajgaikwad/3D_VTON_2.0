from utils.garment_processor import prepare_garment_texture

class U2NetHandler:
    def __init__(self, device="cpu"):
        self.device = device
        print("INFO: U2Net Segmentation Handler Ready.")

    def segment(self, image_bytes):
        # This name now matches the call in tryon_routes.py
        return prepare_garment_texture(image_bytes)
