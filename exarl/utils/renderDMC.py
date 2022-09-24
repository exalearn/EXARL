import re
import os
from PIL import Image
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def render_tmp(file_name, dur=50):
    print("Rendering...")
    frames=[]
    for f in sorted_alphanumeric(os.listdir("tmp")):
        frame = Image.open('tmp/'+f)
        frames.append(frame)
        os.remove('tmp/'+f)
    frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=dur)