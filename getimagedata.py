import sys
from PIL import Image

im = Image.open(f"hd/{sys.argv[1]}.png")
print(f"{im.info}")