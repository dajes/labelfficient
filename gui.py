from __future__ import division

import os
import tkinter as tk
import tkinter.messagebox
import warnings
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from commons.annotating import create_annotation, parse_annotation, img_name_to_annotation
from commons.datatypes import Detection
from commons.images_dataset import ImagesDataset
from commons.siam_mask.siam_tracker import SiamTracker
from commons.system_information.screen import get_target_size, RESOLUTION
from commons.utils import get_all_files, makedirs2file

DEFAULT_PATH = r'D:\datasets\nature_small'
# colors for the bounding boxes
COLORS = ['red', 'blue', 'green', 'black', 'pink']
IMG_SIZE = 64
N_COMPONENTS = 64
BATCH_SIZE = 32
FORMAT = ['.jpg', '.jpeg', '.png']
POSIX = os.name == 'posix'

HALF = 1 / 2 ** 1.5


class Modifiers:
    SHIFT = 1 << 0
    CAPS_LOCK = 1 << 1
    CONTROL = 1 << 2
    ALT_L = 1 << 3
    NUM_LOCK = 1 << 4
    ALT_R = 1 << 7


class KeyCodes:
    BackSpace = 22 if POSIX else 8
    Esc = 9 if POSIX else 27
    L_Shift = 50 if POSIX else 16
    R_Shift = 62 if POSIX else 16
    C = 55 if POSIX else 67
    V = 54 if POSIX else 86
    A = 38 if POSIX else 65
    ARROW_R = 114 if POSIX else 39
    ARROW_L = 113 if POSIX else 37


# noinspection PyTypeChecker,PyUnresolvedReferences
class Labelfficient:
    BBOX_FORMAT = '%s [%d, %d, %d, %d]'
    RELATIVE_SIZE = 0.7
    POINT_RADIUS = 7
    TARGET_IMG_SIZE = 512

    def uncased_bind(self, key, func):
        key = key.split('-')
        lower_key = '-'.join([k.lower() if len(k) == 1 or (len(k) == 2 and k.endswith('>')) else k for k in key])
        upper_key = '-'.join([k.upper() if len(k) == 1 or (len(k) == 2 and k.endswith('>')) else k for k in key])
        self.main_panel.bind(lower_key, func)
        self.main_panel.bind(upper_key, func)
        self.uncased_binds.append((lower_key, func))
        self.uncased_binds.append((upper_key, func))

    def uncased_unbind_all(self):
        for key, _ in self.uncased_binds:
            self.main_panel.unbind_all(key)

    def uncased_return_binds(self):
        for key, func in self.uncased_binds:
            self.parent.bind(key, func)

    @staticmethod
    def disable_keyboard(entry: tk.Entry, root):
        state = {'SELECTION_START': None}

        def f(event):
            if event.keycode == KeyCodes.BackSpace:
                try:
                    entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
                except tk.TclError:
                    entry.delete(len(entry.get()) - 1, tk.END)
            elif event.keycode == KeyCodes.Esc:
                root.focus()
            elif event.keycode == KeyCodes.L_Shift or event.keycode == KeyCodes.R_Shift:
                try:
                    entry.index(tk.SEL_FIRST)
                except tk.TclError:
                    # if there is no selection, remove selection start point, else keep everything as is
                    state['SELECTION_START'] = None
            elif event.keycode == KeyCodes.V and event.state & Modifiers.CONTROL:
                clipboard = root.clipboard_get()
                try:
                    start = entry.index(tk.SEL_FIRST)
                    entry.delete(start, tk.SEL_LAST)
                except tk.TclError:
                    start = entry.index(tk.INSERT)
                entry.insert(start, clipboard)
            elif event.keycode == KeyCodes.C and event.state & Modifiers.CONTROL:
                try:
                    clipboard = entry.selection_get()
                    root.clipboard_clear()
                    root.clipboard_append(clipboard)
                except tk.TclError:
                    pass
            elif event.keycode == KeyCodes.ARROW_R:
                cursor = entry.index(tk.INSERT)
                if event.state & Modifiers.SHIFT:
                    try:
                        start_of_selection = state['SELECTION_START'] or entry.index(tk.SEL_FIRST)
                    except tk.TclError:
                        start_of_selection = cursor
                        state['SELECTION_START'] = start_of_selection
                    end_of_selection = cursor + 1
                    start_of_selection, end_of_selection = sorted([start_of_selection, end_of_selection])
                    entry.select_range(start_of_selection, end_of_selection)
                else:
                    entry.select_clear()
                entry.icursor(cursor + 1)
            elif event.keycode == KeyCodes.ARROW_L:
                cursor = entry.index(tk.INSERT)
                if event.state & Modifiers.SHIFT:
                    start_of_selection = cursor - 1
                    try:
                        end_of_selection = state['SELECTION_START'] or entry.index(tk.SEL_LAST)
                    except tk.TclError:
                        end_of_selection = cursor
                        state['SELECTION_START'] = start_of_selection
                    start_of_selection, end_of_selection = sorted([start_of_selection, end_of_selection])
                    entry.select_range(start_of_selection, end_of_selection)
                else:
                    entry.select_clear()
                entry.icursor(cursor - 1)
            elif event.keycode == KeyCodes.A and event.state & Modifiers.CONTROL and not event.state & Modifiers.SHIFT:
                entry.select_range(0, tk.END)
            elif str(event.char).isprintable():
                entry.insert(tk.END, str(event.char))
            else:
                print(event.keysym, repr(event.char), event.keycode)
            return "break"

        entry.bind('<Key>', f)

    def __init__(self, master):
        self.tracker = None
        self.parent = master
        self.parent.attributes("-fullscreen", True)
        self.parent.title("Labelfficient")
        self.default_cursor = ''
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=1)
        self.parent.resizable(width=tk.FALSE, height=tk.FALSE)

        self.parent.iconphoto(False, ImageTk.PhotoImage(file='icon.png'))

        self.image_list = []
        self.features = []
        self.watched = []
        self.cur = 0
        self.total = 0
        self.labeled = 0
        self.labeled_arr = None
        self.image_name = ''
        self.label_path = ''
        self.image_path = ''
        self.tk_img = None
        self.first_visit = False
        self.img = None
        self.height = 1
        self.width = 1
        self.k = 1
        self.scale = 1
        self.class_names = []
        self.undo_list = deque(maxlen=50)
        self.offset = np.zeros(2, dtype=int)

        self.STATE = {'click': False, 'x': 0, 'y': 0, 'create': False,
                      'tracking_box': None, 'resizing_box': None, 'mouse_pos': None}

        self.bbox_id_list = []
        self.bbox_id = None
        self.class_list = []
        self.bbox_list = []
        self.color_list = []
        self.points = []
        self.hl = None
        self.vl = None
        self.img_id = None

        self.label = tk.Label(self.frame, text="Dataset:")
        self.label.grid(row=0, column=0, sticky=tk.E)
        self.entry = tk.Entry(self.frame)
        self.disable_keyboard(self.entry, self.parent)
        self.entry.insert(tk.END, DEFAULT_PATH)
        self.entry.grid(row=0, column=1, sticky=tk.W + tk.E)
        self.ld_btn = tk.Button(self.frame, text="Load", command=self.load_dir)
        self.ld_btn.grid(row=0, column=2, sticky=tk.W + tk.E)

        self.uncased_binds = []
        self.main_panel = tk.Canvas(self.frame)
        self.main_panel.bind("<Button-1>", self.mouse_click)
        self.main_panel.bind("<Button-3>", self.mouse_right_click)
        self.main_panel.bind("<ButtonRelease-1>", self.mouse_release)
        self.main_panel.bind("<Motion>", self.mouse_move)
        self.uncased_bind("<Control-z>", self.undo)
        self.uncased_bind("<Control-s>", self.save_image)
        self.uncased_bind("<Escape>", self.cancel_bbox)
        self.uncased_bind("r", self.load_labels)
        self.uncased_bind("n", self.toggle_box_creation)
        self.uncased_bind("s", self.cancel_bbox)
        self.uncased_bind("a", self.prev_image)
        self.uncased_bind("d", self.next_image)
        self.uncased_bind("g", self.predict_next_image)
        self.uncased_bind("f", self.find_outlier)
        self.main_panel.grid(row=1, column=1, rowspan=9, sticky=tk.W + tk.N)
        self.main_panel.bind('<Enter>', self._enter_main)
        self.main_panel.bind('<Leave>', self._leave_main)
        self.main_panel.config(width=int(RESOLUTION[0]), height=int(RESOLUTION[1]))

        self.btn_clear = tk.Button(self.frame, text='Clear labels', command=self.clear_bbox)
        self.btn_clear.grid(row=2, column=2, sticky=tk.W + tk.E + tk.N)

        self._sort_var = tk.IntVar()
        self.sort_check = tk.Checkbutton(self.frame, text='Sort pixel distance when loading', variable=self._sort_var)
        self.sort_check.grid(row=3, column=2, sticky=tk.E)
        self._cuda_var = tk.IntVar()
        self.cuda_check = tk.Checkbutton(self.frame, text='Use cuda for tracking', variable=self._cuda_var)
        self.cuda_check.grid(row=4, column=2, sticky=tk.E + tk.N)

        self.lb3 = tk.Label(self.frame, text='Type class to add:')
        self.lb3.grid(row=4, column=2, sticky=tk.W + tk.E + tk.S)
        self.class_entry = tk.Entry(self.frame)
        self.disable_keyboard(self.class_entry, self.parent)
        self.class_entry.grid(row=5, column=2, sticky=tk.W + tk.E)
        self.class_btn = tk.Button(self.frame, text="Add", command=self.press_class_btn)
        self.class_btn.grid(row=6, column=2, sticky=tk.W)
        self.paste_class_btn = tk.Button(self.frame, text="Clipboard", command=self.press_paste_class_btn)
        self.paste_class_btn.grid(row=6, column=2, sticky=tk.E)
        self._autopaste_var = tk.IntVar()
        self.autopaste_check = tk.Checkbutton(self.frame, text='Auto paste when empty', variable=self._autopaste_var)
        self.autopaste_check.grid(row=7, column=2, sticky=tk.E)
        self.lb2 = tk.Label(self.frame, text='Choose class for new box:')
        self.lb2.grid(row=8, column=2, sticky=tk.W + tk.E + tk.N)
        self.class_listbox = tk.Listbox(self.frame, width=22, height=12)
        self.class_listbox.grid(row=9, column=2, sticky=tk.N)

        self.ctr_panel = tk.Frame(self.frame)
        self.ctr_panel.grid(row=10, column=1, columnspan=2, sticky=tk.W + tk.E)
        self.reload_btn = tk.Button(self.ctr_panel, text='Reload annotation', width=15, command=self.load_labels)
        self.reload_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.prev_btn = tk.Button(self.ctr_panel, text='<< Prev', width=10, command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.next_btn = tk.Button(self.ctr_panel, text='Next >>', width=10, command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.progress_bar = tk.Label(self.ctr_panel, text="")
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        self.eg_panel = tk.Frame(self.frame, border=10)
        self.eg_panel.grid(row=1, column=0, rowspan=5, sticky=tk.N)
        self.tmp_label2 = tk.Label(self.eg_panel, text="Examples:")
        self.tmp_label2.pack(side=tk.TOP, pady=5)
        self.eg_labels = []
        for _ in range(3):
            self.eg_labels.append(tk.Label(self.eg_panel))
            self.eg_labels[-1].pack(side=tk.TOP)

        self.position_indicator = tk.Label(self.ctr_panel, text='')
        self.position_indicator.pack(side=tk.RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)
        self.distance_thresh = 0
        self.next_img = None

    def undo(self, _=None):
        if self.STATE['tracking_box'] is None \
                and self.STATE['resizing_box'] is None \
                and not self.STATE['click'] \
                and len(self.undo_list) > 0:
            undo_action = self.undo_list.pop()
            if undo_action['action'] == 'create_bbox':
                idx = undo_action['id']
                self.del_bbox(idx)
            elif undo_action['action'] in {'resize', 'move'}:
                idx = undo_action['id']

                coords = self.bbox_list[idx]
                for i, c in enumerate(undo_action['initial']):
                    coords[i] = c
                self.change_bbox(idx)
            elif undo_action['action'] == 'delete':
                label = undo_action['label']
                bbox = undo_action['bbox']
                idx = undo_action['id']
                color = undo_action['color']

                _coords = (np.array(bbox) * self.k * self.scale).round().astype(int)
                self.bbox_id_list.insert(idx, self.main_panel.create_rectangle(*_coords, width=2, outline=color))
                self.class_list.insert(idx, label)
                self.bbox_list.insert(idx, bbox)
                self.color_list.insert(idx, color)

    @staticmethod
    def _get_mouse_direction(event):
        if POSIX:
            amount = (-1) ** (event.num == 4)
        else:
            amount = int(-(event.delta / 120))
        return amount

    def _on_mousewheel(self, event, amount=None):
        if amount is None:
            amount = self._get_mouse_direction(event)
        self.main_panel.yview_scroll(amount, "units")
        self.clear_points()

    def _on_horizontal_mousewheel(self, event, amount=None):
        if amount is None:
            amount = self._get_mouse_direction(event)
        if POSIX:
            self._on_mousewheel(event, -amount)
        self.main_panel.xview_scroll(amount, "units")
        self.clear_points()

    def redraw_boxes(self):
        for i in range(len(self.bbox_list)):
            self.change_bbox(i)
        self.clear_points()
        self.mouse_move()

    def _on_zoom(self, event, gamma=0.9, amount=None):

        x, y = self.get_pos(event)

        if amount is None:
            amount = self._get_mouse_direction(event)
        if POSIX:
            self._on_mousewheel(event, -amount)
        scale = gamma ** amount
        real_width = self.width * self.k * self.scale
        real_height = self.height * self.k * self.scale
        reduce_width = (1 - scale) * real_width
        reduce_height = (1 - scale) * real_height
        x_offset = int(reduce_width * x / real_width)
        y_offset = int(reduce_height * y / real_height)
        self.offset[0] += x_offset
        self.offset[1] += y_offset
        if self.STATE['click']:
            self.STATE['x'] += x_offset
            self.STATE['y'] += y_offset

        self.scale *= scale
        self.adjust2k()
        self.redraw_boxes()

    def _enter_main(self, _=None):
        if POSIX:
            self.main_panel.bind("<Button-4>", self._on_mousewheel)
            self.main_panel.bind("<Button-5>", self._on_mousewheel)
            self.main_panel.bind_all("<Shift-Button-4>", self._on_horizontal_mousewheel)
            self.main_panel.bind_all("<Shift-Button-5>", self._on_horizontal_mousewheel)
            self.main_panel.bind_all("<Control-Button-4>", self._on_zoom)
            self.main_panel.bind_all("<Control-Button-5>", self._on_zoom)
        else:
            self.main_panel.bind_all("<MouseWheel>", self._on_mousewheel)
            self.main_panel.bind_all("<Shift-MouseWheel>", self._on_horizontal_mousewheel)
            self.main_panel.bind_all("<Control-MouseWheel>", self._on_zoom)
        self.uncased_return_binds()

    def _leave_main(self, _=None):
        if POSIX:
            self.main_panel.unbind_all("<Button-4>")
            self.main_panel.unbind_all("<Button-5>")
            self.main_panel.unbind_all("<Shift-Button-4>")
            self.main_panel.unbind_all("<Shift-Button-5>")
            self.main_panel.unbind_all("<Control-Button-4>")
            self.main_panel.unbind_all("<Control-Button-5>")
        else:
            self.main_panel.unbind_all("<MouseWheel>")
            self.main_panel.unbind_all("<Shift-MouseWheel>")
            self.main_panel.unbind_all("<Control-MouseWheel>")
        self.uncased_unbind_all()

    def pseudo_iter(self, _=None):
        raise NotImplementedError('To be done')

    def load_dir(self, _=False, image_dir=None):
        if image_dir is None:
            image_dir = self.entry.get()
        images = []
        for _format in FORMAT:
            images += get_all_files(image_dir, _format)

        images = np.array(sorted(images))

        if self._sort_var.get():
            dataset = ImagesDataset(images, resize=IMG_SIZE)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=dataset.collate_fn)
            dataloader = tqdm(dataloader, desc=f'Arranging images', leave=False)

            all_images = []
            for img_batch, in dataloader:
                all_images += img_batch

            if len(all_images) == 0:
                print('No .JPEG images found in the specified dir!')
                return
            all_images = np.array(all_images).reshape([len(all_images), -1])
            assert len(all_images) > 0
            pca = PCA(min([N_COMPONENTS, *all_images.shape]))
            features = pca.fit_transform(all_images)

            possible_idx = list(range(1, len(features)))
            rearrange = [0]
            while len(possible_idx) > 0:
                feature = 2 * features[rearrange[-1]] - features[rearrange[-2:][0]]
                distances = np.sum((features[possible_idx] - feature) ** 2, axis=1)
                rearrange.append(possible_idx.pop(int(np.argmin(distances))))
            images = images[rearrange]
            features = features[rearrange]
        else:
            features = None
        self.image_list = images
        self.features = features
        self.labeled = 0
        self.labeled_arr = np.zeros(len(self.image_list), dtype=bool)

        self.cur = 0
        self.watched = [self.cur]
        self.total = len(self.image_list)
        self.load_image()

    def find_outlier(self, _=None):
        if self.features is None:
            tk.messagebox.showerror("Not implemented",
                                    "For the sake of performance, you can find outliers only when "
                                    "loading images with pixel sorting enabled")
        unwatched = list(range(len(self.image_list)))
        for idx in sorted(set(self.watched), reverse=True):
            del unwatched[idx]

        watched_features = self.features[self.watched][None]
        unwatched_features = self.features[unwatched][:, None]

        distances = np.sum((watched_features - unwatched_features) ** 2, axis=2)
        min_distances = np.min(distances, axis=1)

        self.go_to_image(idx=unwatched[np.argmax(min_distances)])

    def load_labels(self, _=None, objects=None):
        self.clear_bbox()
        class_names = set()
        if os.path.exists(self.label_path) and objects is None:
            with open(self.label_path, 'r') as ann:
                _, objects = parse_annotation(ann.read())
        if objects is not None:
            for ann in objects:
                bbox = ann['bbox']
                label = ann['name']
                if label not in self.class_names:
                    class_names.add(label)
                color = COLORS[(len(self.bbox_list) - 1) % len(COLORS)]
                self.class_list.append(label)
                self.bbox_list.append(bbox)
                self.color_list.append(color)
                _bbox = self._to_real_coords(bbox)
                rect_id = self.main_panel.create_rectangle(_bbox[0], _bbox[1], _bbox[2], _bbox[3], width=2,
                                                           outline=color)
                self.bbox_id_list.append(rect_id)
        self.add_classes(class_names)

    def add_classes(self, class_names):
        class_names = [name for name in class_names if name not in self.class_names]
        if len(class_names) > 0:
            class_names = list(sorted(class_names))
            self.class_names = class_names + self.class_names
            for i, class_name in reversed(list(enumerate(class_names))):
                self.class_listbox.insert(0, class_name)
                self.class_listbox.itemconfig(0,
                                              fg=COLORS[(len(self.class_names) - len(class_names) + i) % len(COLORS)])

    def press_class_btn(self, new_class=None):
        if new_class is None:
            new_class = self.class_entry.get()
        new_class = ' '.join(new_class.split())
        self.add_classes([new_class])
        self.class_listbox.focus()
        idx = self.class_names.index(new_class)
        self.class_listbox.see(idx)
        self.class_listbox.select_set(idx)
        self.class_entry.delete(0, tk.END)

    def press_paste_class_btn(self):
        clipboard = self.parent.clipboard_get()
        self.press_class_btn(clipboard)

    def update_resolution(self, width=None, height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        _, self.k = get_target_size((self.width, self.height), self.RELATIVE_SIZE, return_k=True)
        self.adjust2k()

    def adjust2k(self):
        self.distance_thresh = self.POINT_RADIUS / (min(self.width, self.height) * self.k * self.scale)
        self.tk_img = ImageTk.PhotoImage(self.img.resize((int(self.width * self.k * self.scale),
                                                          int(self.height * self.k * self.scale)),
                                                         Image.NEAREST))
        if self.img_id is not None:
            self.main_panel.delete(self.img_id)
        self.img_id = self.main_panel.create_image(self.offset[0], self.offset[1], image=self.tk_img, anchor=tk.NW)

    def load_image(self, load_labels=True):
        # load image
        self.image_path = self.image_list[self.cur]
        self.first_visit = self.cur not in self.watched
        if self.first_visit:
            self.watched.append(self.cur)
        self.image_list[self.cur] = self.image_path
        self.img = Image.open(self.image_path)
        self.update_resolution(self.img.width, self.img.height)
        self.progress_bar.config(text="%04d/%04d (%04d labeled)" % (self.cur + 1, self.total, self.labeled))

        self.image_name = '.'.join(os.path.split(self.image_path)[-1].split('.')[:-1])
        self.label_path = img_name_to_annotation(self.image_path)
        if load_labels:
            self.load_labels()

    def save_image(self, _=None):
        if len(self.bbox_list) == 0:
            self.labeled -= self.labeled_arr[self.cur]
            self.labeled_arr[self.cur] = False
            if os.path.exists(self.label_path):
                os.remove(self.label_path)
            return
        self.labeled += not self.labeled_arr[self.cur]
        self.labeled_arr[self.cur] = True
        with open(makedirs2file(self.label_path), 'w+') as f:
            f.write(create_annotation(self.image_name, self.class_list, self.bbox_list, self.width, self.height))

    def remove_target_lines(self):
        if self.vl:
            self.main_panel.delete(self.vl)
        if self.hl:
            self.main_panel.delete(self.hl)

    def toggle_box_creation(self, _=None):
        self.clear_points()
        self.STATE['create'] ^= True
        self.STATE['click'] = False
        self.default_cursor = 'tcross' if self.STATE['create'] else ''
        self.main_panel.config(cursor=self.default_cursor)
        self.remove_target_lines()

    def clear_points(self):
        for point in self.points:
            self.main_panel.delete(point)

    @staticmethod
    def get_edge_points(bbox):
        x_center = (bbox[2] + bbox[0]) // 2
        y_center = (bbox[3] + bbox[1]) // 2
        coords = []
        for i in range(0, 4, 2):
            for j in range(1, 4, 2):
                coords.append((bbox[i], bbox[j]))
            coords.append((bbox[i], y_center))
            coords.append((x_center, bbox[i + 1]))
        return coords

    @staticmethod
    def point_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def get_closest_box(self, mouse_pos):
        closest_box = None
        min_distance = np.inf
        inside = False
        for box_id, bbox in enumerate(self.bbox_list):
            point_in_box = self.point_in_box(mouse_pos, bbox)
            for point in self.get_edge_points(bbox):
                distance = (((point[0] - mouse_pos[0]) / self.width) ** 2 +
                            ((point[1] - mouse_pos[1]) / self.height) ** 2) ** 0.5
                if distance < min_distance and (point_in_box or distance < self.distance_thresh):
                    min_distance = distance
                    closest_box = box_id
                    inside = point_in_box
        return closest_box, min_distance, inside

    def _to_real_coords(self, coords):
        return (np.array(coords) * (self.k * self.scale) + self.offset[[0, 1] * (len(coords) // 2)]).round().astype(int)

    def _get_real_bbox(self, idx):
        return self._to_real_coords(self.bbox_list[idx])

    def set_focus(self, bbox_id=None):
        if bbox_id is None:
            return
        bbox = self._get_real_bbox(bbox_id)
        color = self.color_list[bbox_id]

        for point in self.get_edge_points(bbox):
            oval = self.main_panel.create_oval(point[0] - self.POINT_RADIUS, point[1] - self.POINT_RADIUS,
                                               point[0] + self.POINT_RADIUS, point[1] + self.POINT_RADIUS,
                                               fill=color, outline='black')
            self.points.append(oval)

    def get_pos(self, event):
        if event is None:
            assert self.STATE['mouse_pos'] is not None
            return self.STATE['mouse_pos']
        x = self.main_panel.canvasx(event.x)
        y = self.main_panel.canvasy(event.y)
        x -= self.offset[0]
        y -= self.offset[1]
        x = min(max(x, 0), self.width * self.k * self.scale)
        y = min(max(y, 0), self.height * self.k * self.scale)
        self.STATE['mouse_pos'] = (x, y)
        return x, y

    def mouse_release(self, event):
        self.clear_points()
        x, y = self.get_pos(event)
        if self.STATE['resizing_box'] is not None:
            self.undo_list.append({'action': 'resize',
                                   'id': self.STATE['resizing_box'][0],
                                   'initial': self.STATE['resizing_box'][5:9]})
            self.STATE['resizing_box'] = None
        elif self.STATE['tracking_box'] is not None:
            self.undo_list.append({'action': 'move',
                                   'id': self.STATE['tracking_box'][0],
                                   'initial': self.STATE['tracking_box'][3:7]})
            self.STATE['tracking_box'] = None
        elif self.STATE['click']:
            _x, _y = x + self.offset[0], y + self.offset[1]
            x1, x2 = min(self.STATE['x'], _x), max(self.STATE['x'], _x)
            y1, y2 = min(self.STATE['y'], _y), max(self.STATE['y'], _y)

            sel = self.class_listbox.curselection()
            if len(sel) == 0:
                if self._autopaste_var.get():
                    self.press_paste_class_btn()
                    sel = self.class_listbox.curselection()
                else:
                    tk.messagebox.showerror("Please select a class",
                                            "You should select a class you are trying to add from the right listbox")
                    return
            elif len(sel) > 2:
                tk.messagebox.showerror("Please select only 1 class",
                                        "You should select 1 class you are trying to add from the right listbox")
                return
            label = self.class_names[sel[0]]

            self.class_list.append(label)
            bbox = np.round((np.array([x1, y1, x2, y2]) - self.offset[[0, 1, 0, 1]])
                            / (self.k * self.scale)).astype(int)
            self.bbox_list.append(bbox)
            self.color_list.append(self.STATE['cur_color'])
            del self.STATE['cur_color']
            self.undo_list.append({'action': 'create_bbox',
                                   'id': len(self.bbox_id_list)})
            self.bbox_id_list.append(self.bbox_id)
            self.bbox_id = None
            self.toggle_box_creation()

    def get_mouse_pos(self, event):
        x, y = self.get_pos(event)
        return int(round(x / (self.k * self.scale))), int(round(y / (self.k * self.scale)))

    def mouse_right_click(self, event):
        self.clear_points()
        if self.STATE['resizing_box'] is not None or self.STATE['tracking_box'] is not None or self.STATE['create']:
            return
        mouse_pos = self.get_mouse_pos(event)
        closest_box, distance, inside = self.get_closest_box(mouse_pos)
        if closest_box is None:
            return
        if inside:
            self.del_bbox(closest_box, save=True)

    def mouse_click(self, event):
        self.clear_points()
        x, y = self.get_pos(event)
        if self.STATE['resizing_box'] is not None or self.STATE['tracking_box'] is not None:
            self.mouse_release(event)
        elif self.STATE['create']:
            if self.STATE['click']:
                self.mouse_release(event)
            else:
                _x, _y = x + self.offset[0], y + self.offset[1]
                self.STATE['x'], self.STATE['y'] = _x, _y
                self.STATE['click'] = True
        else:
            mouse_pos = self.get_mouse_pos(event)
            closest_box, distance, inside = self.get_closest_box(mouse_pos)
            if closest_box is None:
                return
            bbox = self.bbox_list[closest_box]
            if distance < 2 * self.distance_thresh:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                scores = np.array([bbox[0] + 0.25 * width - mouse_pos[0],
                                   bbox[1] + 0.25 * height - mouse_pos[1],
                                   mouse_pos[0] - (bbox[2] - 0.25 * width),
                                   mouse_pos[1] - (bbox[3] - 0.25 * height)])
                threshold = max(scores[np.argsort(scores)[-3]], 0)
                mask = scores > threshold

                self.STATE['resizing_box'] = (closest_box,
                                              *mask,
                                              *bbox)
            elif inside:
                self.STATE['tracking_box'] = (closest_box,
                                              (bbox[0] + bbox[2]) / 2 - mouse_pos[0],
                                              (bbox[1] + bbox[3]) / 2 - mouse_pos[1],
                                              *bbox)

    def mouse_move(self, event=None):
        x, y = self.get_pos(event)
        _x, _y = x + self.offset[0], y + self.offset[1]
        mouse_pos = self.get_mouse_pos(event)
        self.position_indicator.config(text='x: %d, y: %d' % mouse_pos)
        if self.tk_img:
            self.remove_target_lines()
            if self.STATE['create']:
                self.hl = self.main_panel.create_line(self.offset[0], _y,
                                                      self.tk_img.width() + self.offset[0], _y, width=2)
                self.vl = self.main_panel.create_line(_x, self.offset[1], _x,
                                                      self.tk_img.height() + self.offset[1], width=2)
        self.clear_points()
        if self.bbox_id:
            self.main_panel.delete(self.bbox_id)
        if self.STATE['resizing_box'] is not None:
            box_id, x1_mask, y1_mask, x2_mask, y2_mask, *_ = self.STATE['resizing_box']
            coords = self.bbox_list[box_id]
            mask = [x1_mask, y1_mask, x2_mask, y2_mask]
            for i in range(4):
                if mask[i]:
                    coords[i] = mouse_pos[i % 2]
            if np.any(mask):
                self.change_bbox(box_id)
        elif self.STATE['tracking_box'] is not None:
            box_id, x_off, y_off, *_ = self.STATE['tracking_box']
            coords = self.bbox_list[box_id]
            width = abs(coords[2] - coords[0])
            height = abs(coords[3] - coords[1])
            offsets = [-width // 2, -height // 2, width // 2, height // 2]
            g_off = [x_off, y_off]
            for i in range(4):
                coords[i] = mouse_pos[i % 2] + offsets[i] + g_off[i % 2]
            self.change_bbox(box_id)
        elif self.STATE['click']:
            if 'cur_color' not in self.STATE:
                self.STATE['cur_color'] = COLORS[len(self.bbox_list) % len(COLORS)]
            self.bbox_id = self.main_panel.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                            _x, _y, width=2,
                                                            outline=self.STATE['cur_color'])
        else:
            closest_box, *_ = self.get_closest_box(mouse_pos)
            self.set_focus(closest_box)

    def cancel_bbox(self, _=None):
        if self.STATE['click'] and self.bbox_id:
            self.main_panel.delete(self.bbox_id)
            self.bbox_id = None
            self.STATE['click'] = False

    def change_bbox(self, idx):
        self.main_panel.delete(self.bbox_id_list[idx])
        _coords = self._get_real_bbox(idx)
        self.bbox_id_list[idx] = self.main_panel.create_rectangle(*_coords, width=2, outline=self.color_list[idx])

    def del_bbox(self, idx: int = None, save: bool = False):
        self.main_panel.delete(self.bbox_id_list[idx])
        self.bbox_id_list.pop(idx)
        label = self.class_list.pop(idx)
        bbox = self.bbox_list.pop(idx)
        color = self.color_list.pop(idx)
        if save:
            self.undo_list.append({'action': 'delete',
                                   'id': idx,
                                   'label': label,
                                   'bbox': bbox,
                                   'color': color
                                   })
        self.redraw_boxes()

    def clear_bbox(self):
        for idx in range(len(self.bbox_id_list)):
            self.main_panel.delete(self.bbox_id_list[idx])
        self.bbox_id_list.clear()
        self.class_list.clear()
        self.bbox_list.clear()
        self.color_list.clear()
        self.undo_list.clear()

    def prev_image(self, _=None):
        self.save_image()
        if self.cur > 0:
            self.cur -= 1
            self.load_image()

    def next_image(self, _=None, load_labels=True):
        self.save_image()
        if self.cur < self.total - 1:
            self.cur += 1
            self.load_image(load_labels=load_labels)

    def go_to_image(self, _=None, idx=0):
        self.save_image()
        self.cur = idx
        self.load_image()

    def resize_img(self, img):
        target_size, k = get_target_size(tuple(img.shape[1 - i] for i in range(2)), self.TARGET_IMG_SIZE, return_k=True)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        return k, img

    def track(self, prev_img, cur_img, prev_det):
        old_k, prev_img = self.resize_img(prev_img)
        for det in prev_det:
            det.bbox = (det.bbox * old_k).round().astype(np.int64)
        k, cur_img = self.resize_img(cur_img)
        if self.tracker is None:
            self.tracker = SiamTracker(SiamTracker.CONFIG_DAVIS)

        self.tracker.set_device(0 if self._cuda_var.get() else None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            tracked = self.tracker.predict((prev_img, cur_img), prev_det, calc_mask=True)
        return [{'bbox': det.bbox / k, 'name': det.name} for det in tracked]

    def predict_next_image(self, _=None):
        prev_det = [Detection(label, bbox) for bbox, label in zip(self.bbox_list, self.class_list)]
        prev_img = np.array(self.img)
        self.next_image(_, load_labels=False)
        cur_img = np.array(self.img)
        tracked = self.track(prev_img, cur_img, prev_det)
        self.load_labels(objects=tracked)


if __name__ == '__main__':
    root = tk.Tk()
    tool = Labelfficient(root)
    root.mainloop()
