import tkinter as tk
from collections import defaultdict
from tkinter import messagebox, ttk, filedialog
from converter import Converter, NoneSelected
from typing import DefaultDict, Dict, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from image import Image


class OverlayTool:
    def __init__(self, root_path: str):
        self.root: tk.Tk = tk.Tk()
        self.converter: Converter = Converter(root_path)
        self.root.geometry("800x700")
        # self.labelttk.Label(self.root, text="Some Image application")
        # self.label_.grid(column=2, row=0, columnspan=2, padx=5, pady=5)
        self.root.title("Image Application")
        self.radio_buttons: DefaultDict[str, List[tk.IntVar]] = defaultdict(list)
        self.selected = []
        self._setup_canvas()
        self._images_to_radios()
        self.root.mainloop()

    def _setup_canvas(self):
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.grid(column=0, row=4, columnspan=2, padx=5, pady=5)
        tk.Button(self.root, text="Overlay", command=self._overlay).grid(
            column=0, row=3, padx=5, pady=5
        )
        tk.Button(self.root, text="Save", command=self._save).grid(
            column=1, row=3, padx=5, pady=5
        )

    def _images_to_radios(self) -> None:
        for col, directory in enumerate(self.converter.round_directories):
            frame1 = tk.LabelFrame(self.root, text=directory)
            frame1.grid(column=col, row=1, padx=5, pady=5, ipadx=5, ipady=5)
            for row, name in enumerate(self.converter.round_to_tiffs_dict[directory]):
                var = tk.IntVar()
                tk.Checkbutton(frame1, text=name.image_path.name, variable=var).grid(
                    column=col, row=row + 1
                )
                self.radio_buttons[directory].append(var)

    def _get_selected(self) -> Dict[str, List[Image]]:
        selected = {}
        for rnds in self.radio_buttons:
            selected[rnds] = []
            for pos, var in enumerate(
                self.radio_buttons[rnds]
            ):  # var = the radio button / the image
                if var.get() == 1:
                    selected[rnds].append(self.converter.round_to_tiffs_dict[rnds][pos])
        return selected

    def _overlay(self):
        try:
            self.selected = self._get_selected()
            over = self.converter.overlay(self.selected)
            self._show_image(over)
            return over
        except NoneSelected:
            messagebox.showerror("Error", "No images selected")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Something went wrong")

    def _show_image(self, image):
        fig = plt.figure(figsize=(5, 4))
        fig = Figure(figsize=(5, 5), dpi=100)
        a = fig.add_subplot(111)  # type: ignore
        a.imshow(image, cmap="gray")
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=4, columnspan=2, padx=5, pady=5)

    def _save(self):
        if (overlay_image := self._overlay()) is not None:
            self.converter.save(overlay_image, "test.png")
            messagebox.showinfo("Saved", "Image saved")
        else:
            messagebox.showerror("Error", "Image not saved, something went wrong")


def overlay_tool_gui(root_path: str, old_root: tk.Tk):
    old_root.destroy()
    OverlayTool(root_path)


def shift_all(data_path: str):
    """
    Steps to shift all images
    1) All images need to be stacked i.e. images of different views should be 
        overlaid together
    2) Compare the stacked images from 1) from Rnd2 onwards to Rnd1 and figure out the\
        translation. This way all images are aligned with those from round 1
    3) Once these translations have been found, then we can shift all the images and save them
    """
    converter: Converter = Converter(data_path)
    converter.shift_all()
    messagebox.showinfo(title="Success", message="Successfully shifted images")


def run_directory_selector(file_path: str) -> None:
    """
    Run the GUI to select the directory of the data
    """
    root = tk.Tk()
    root.geometry("600x100")
    file_path_var = tk.StringVar(root)
    file_path_var.initialize(file_path)

    root.columnconfigure(index=0, weight=3)
    root.columnconfigure(index=1, weight=8)
    root.columnconfigure(index=2, weight=2)

    def _get_file() -> None:
        selected_file_path = filedialog.askdirectory(
            initialdir=file_path, title="Select folder where images are."
        )
        file_path_var.set(selected_file_path)

    tk.Label(root, text="Select the directory where the files are").grid(
        row=0, columnspan=3, sticky=tk.NSEW, column=0
    )
    tk.Button(root, text="Select folder", command=_get_file).grid(column=0, row=1)
    tk.Entry(root, textvariable=file_path_var).grid(
        row=1, column=1, columnspan=7, sticky=tk.NSEW
    )

    tk.Button(
        root,
        text="Continue to overlay tool",
        command=lambda: (overlay_tool_gui(file_path_var.get(), root)),
    ).grid(row=2, column=0)

    tk.Button(
        root,
        text="Just shift all images",
        command=lambda: shift_all(file_path_var.get()),
    ).grid(row=2, column=1)

    print(f"You have selected {file_path_var.get()}")
    root.mainloop()


def main(file_path: str = "."):
    run_directory_selector(file_path)
