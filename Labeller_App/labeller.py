import os
import shutil

import tkinter as tk
from PIL import Image, ImageTk

# sudo -H apt-get install python3-pil.imagetk

DATA_DIR = '../OriginalData/DCD-master/Mohit/not-sure'
OUTPUT_DIR = '../OriginalData/DCD-master/Mohit/not-sure-classified'
MAX_SIZE = (960, 960)


def image_generator():
    for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
        for filename in filenames:
            if '.jpg' in filename:
                source_path = os.sep.join([dirpath, filename])
                answer = yield source_path
                dest_path = os.sep.join([OUTPUT_DIR, answer, filename])
                print('Moving', source_path, 'to', dest_path)
                shutil.move(source_path, dest_path)
    raise StopIteration


def display_image(source_path):
    global last_img_ref
    global last_logo
    img = Image.open(source_path)
    img.thumbnail(MAX_SIZE, Image.ANTIALIAS)
    last_logo = logo = ImageTk.PhotoImage(img)
    if last_img_ref is None:
        last_img_ref = canvas.create_image(20, 20, anchor=tk.NW, image=logo)
    else:
        canvas.itemconfigure(last_img_ref, image=logo)


def callback(answer):
    source_path = gen.send(answer)
    display_image(source_path)


if __name__ == '__main__':
    last_img_ref = None
    gen = image_generator()
    damage_levels = ['01-minor', '02-moderate', '03-severe']
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        for damage in damage_levels:
            os.mkdir(os.path.join(OUTPUT_DIR, damage))

    root = tk.Tk()
    root.geometry("1000x768")
    root.title("The Awesome Labelling App")

    canvas = tk.Canvas(root, height=720, width=970, bd=1, highlightthickness=1)
    canvas.grid(row=0, column=0, columnspan=3)

    first_image = gen.__next__()
    display_image(first_image)

    b_minor = tk.Button(root, text='Minor', width=15, command=lambda: callback(damage_levels[0]))
    b_moderate = tk.Button(root, text='Moderate', width=15, command=lambda: callback(damage_levels[1]))
    b_severe = tk.Button(root, text='Severe', width=15, command=lambda: callback(damage_levels[2]))

    b_minor.grid(row=1, column=0)
    b_moderate.grid(row=1, column=1)
    b_severe.grid(row=1, column=2)

    root.mainloop()
