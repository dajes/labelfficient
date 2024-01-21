import gc
import io
import os
import time
from typing import Tuple, List
import tkinter as tk
import tkinter.font as tkFont
from tkinter import messagebox
from PIL import Image, ImageTk

from predictors.basic import BasicPredictor
from commons.annotating import parse_annotation, img_name_to_annotation, create_annotation

cache_path = '.cache'
if os.path.exists(cache_path):
    with open(cache_path, 'r') as f:
        DEFAULT_PATH = f.read()
else:
    DEFAULT_PATH = ''

FORMAT = ['jpg', 'jpeg', 'png']
try:
    SYSTEM = os.uname()[0]
except AttributeError:
    SYSTEM = 'Windows'
MACOS = SYSTEM == 'Darwin'
POSIX = SYSTEM != 'Windows'

MAX_PING = 80


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
    C = 54 if POSIX else 67
    V = 55 if POSIX else 86
    A = 38 if POSIX else 65
    ARROW_R = 114 if POSIX else 39
    ARROW_L = 113 if POSIX else 37


colors = \
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), \
        (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (
        255, 250, 200), \
        (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)


def rgb2hex(*rgb_color):
    if isinstance(rgb_color[0], tuple):
        rgb_color = rgb_color[0]
    assert len(rgb_color) == 3
    return "#%02x%02x%02x" % rgb_color


def get_target_size(image_size, target=0.9, return_k=False):
    if isinstance(target, float):
        target = (1920 * target, 1080 * target)
    elif isinstance(target, int):
        target = (target, target)
    k = min(target[0] / image_size[0], target[1] / image_size[1])
    target_size = int(round(image_size[0] * k)), int(round(image_size[1] * k))
    if return_k:
        return target_size, k
    return target_size


# noinspection PyTypeChecker,PyUnresolvedReferences
class Labelfficient:
    RELATIVE_SIZE = 0.7
    POINT_RADIUS = 7
    TARGET_IMG_SIZE = 512
    PREDICTOR_CLASS = BasicPredictor

    def uncased_bind(self, key, func):
        key = key.split('-')
        lower_key = '-'.join(
            k.lower() if len(k) == 1 or (len(k) == 2 and k.endswith('>')) else k
            for k in key
        )

        upper_key = '-'.join(
            k.upper() if len(k) == 1 or (len(k) == 2 and k.endswith('>')) else k
            for k in key
        )

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

    def get_color(self, class_name):
        try:
            return self.class_colors[self.class_names.index(class_name)]
        except ValueError:
            return '#aaaaaa'

    def cyrillic_support(self, event):
        key_pressed = event.char
        cyrillic = "йцукенгшщзхїфівапролджєячсмитьбюЙЦУКЕНГШЩЗХЇФІВАПРОЛДЖЄЯЧСМИТЬБЮ"
        latin = "qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP[]ASDFGHJKL;'ZXCVBNM,."
        if key_pressed in cyrillic:
            key_pressed = latin[cyrillic.index(event.char)]
            self.main_panel.event_generate(f"<KeyPress-{key_pressed}>")
        # for <control-...> keypresses
        elif event.keysym in cyrillic:
            key_pressed = latin[cyrillic.index(event.keysym)]
            self.main_panel.event_generate(f"<Control-{key_pressed}>")

    def __init__(self, master):
        self.images = []
        self.labels = []
        self.tasks = []
        self.labeled_list = []
        self.directory = ''
        self.parent = master
        self.parent.title("Labelfficient")
        self.default_cursor = ''
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=1)
        self.parent.resizable(width=tk.TRUE, height=tk.TRUE)
        self.parent.bind("<Key>", self.cyrillic_support)

        self.cur = 0
        self.tk_img = None
        self.img = None
        self.img_shape = None
        self.height = 1
        self.width = 1
        self.k = 1
        self.scale = 1
        self.class_names = []
        self.class_colors = []
        self.undo_list = []
        self.offset = [0, 0]

        self.STATE = {'click': False, 'x': 0, 'y': 0, 'create': False,
                      'tracking_box': None, 'resizing_box': None, 'mouse_pos': None, 'changing_class': False,
                      'yaro_mod': False}

        self.bbox_id_list = []
        self.bbox_id = None
        self.class_list = []
        self.bbox_list = []
        self.color_list = []
        self.points = []
        self.hl = None
        self.vl = None
        self.img_id = None

        self.label = tk.Label(self.frame, text="Directory:")
        self.label.grid(row=0, column=0, sticky=tk.E)
        self.entry = tk.Entry(self.frame)
        self.disable_keyboard(self.entry, self.parent)
        self.entry.insert(tk.END, DEFAULT_PATH)
        self.entry.grid(row=0, column=1, sticky=tk.W + tk.E)
        self.ld_btn = tk.Button(self.frame, text="Load", command=self.load_dir)
        self.ld_btn.grid(row=0, column=2, sticky=tk.W + tk.E)
        self.last_clear = time.monotonic()

        self.uncased_binds = []
        self.main_panel = tk.Canvas(self.frame)
        self.main_panel.bind("<Button-1>", self.mouse_click)
        self.main_panel.bind(f"<Button-{2 if MACOS else 3}>", self.mouse_right_click)
        self.main_panel.bind("<ButtonRelease-1>", self.mouse_release)
        self.main_panel.bind("<Motion>", self.mouse_move)
        self.uncased_bind("<Control-z>", self.undo)
        self.uncased_bind("<Control-s>", self.save_image)
        self.uncased_bind("<Escape>", self.cancel_bbox)
        self.uncased_bind("r", self.load_labels)
        self.uncased_bind("n", self.toggle_box_creation)
        self.uncased_bind("y", self.toggle_yaro_mod)
        self.uncased_bind("s", self.cancel_bbox)
        self.uncased_bind("a", self.prev_image)
        self.uncased_bind("d", self.next_image)
        self.uncased_bind("g", self.predict_next_image)
        self.uncased_bind("c", self.clear_bbox)
        for i in range(10):
            self.uncased_bind(str(i), self.hotkey)
        self.main_panel.grid(row=1, column=0, rowspan=21, columnspan=2, sticky=tk.W + tk.N)
        self.main_panel.bind('<Enter>', self._enter_main)
        self.main_panel.bind('<Leave>', self._leave_main)
        self.main_panel.config(width=1700, height=920)

        self.lb3 = tk.Label(self.frame, text='Type class to add:')
        self.lb3.grid(row=4, column=2, sticky=tk.W + tk.E + tk.S)
        self.class_entry = tk.Entry(self.frame)
        self.disable_keyboard(self.class_entry, self.parent)
        self.class_entry.grid(row=5, column=2, sticky=tk.W + tk.E + tk.S)
        self.class_btn = tk.Button(self.frame, text="Add", command=self.press_class_btn)
        self.class_btn.grid(row=6, column=2, sticky=tk.W + tk.S)
        self.paste_class_btn = tk.Button(self.frame, text="Clipboard", command=self.press_paste_class_btn)
        self.paste_class_btn.grid(row=6, column=2, sticky=tk.E + tk.S)
        self._autopaste_var = tk.IntVar()
        self.autopaste_check = tk.Checkbutton(self.frame, text='Auto paste when empty', variable=self._autopaste_var)
        self.autopaste_check.grid(row=7, column=2, sticky=tk.E + tk.S)
        self.lb2 = tk.Label(self.frame, text='Choose class for new box:')
        self.lb2.grid(row=8, column=2, sticky=tk.W + tk.E + tk.S)
        self.class_listbox = tk.Listbox(self.frame, width=19, height=10)
        self.class_listbox.grid(row=9, column=2, sticky=tk.S + tk.E)
        self.class_hotkeys = tk.Label(self.frame, text='\n'.join(str(i % 10) for i in range(1, 11)), width=2, height=10)
        self.class_hotkeys.grid(row=9, column=2, sticky=tk.S + tk.W)

        self.ctr_panel = tk.Frame(self.frame)
        self.ctr_panel.grid(row=10, column=1, columnspan=2, sticky=tk.W + tk.E + tk.S)
        self.reload_btn = tk.Button(self.ctr_panel, text='Reload annotation', width=15, command=self.load_labels)
        self.reload_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.prev_btn = tk.Button(self.ctr_panel, text='<< Prev', width=10, command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.next_btn = tk.Button(self.ctr_panel, text='Next >>', width=10, command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.progress_bar = tk.Label(self.ctr_panel, text="")
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.clr_pointer = 0
        self.color_texts = ['Light theme', 'Dark theme']
        self.color_btn = tk.Button(self.ctr_panel, text=self.color_texts[self.clr_pointer], width=10,
                                   command=self.swap_color)
        self.color_btn.pack(side=tk.LEFT, padx=5, pady=3)
        self.loaded = -1

        self.loaded_names = set()

        self.path = None
        self.files = set()
        self.pointer = 1
        self.max_pointer = 1
        self.position_indicator = tk.Label(self.ctr_panel, text='')
        self.position_indicator.pack(side=tk.RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)
        self.distance_thresh = 0
        self.bg = '#222222'
        self.fg = '#AAAAAA'
        self.set_color()
        self.class_usage = {}
        self.last_load = 0
        self.predictor = self.load_predictor()

    def load_predictor(self):
        return self.PREDICTOR_CLASS()

    def select_class(self, idx):
        self.class_listbox.select_clear(0, tk.END)
        self.class_listbox.select_set(idx)

    def hotkey(self, event):
        try:
            i = int(event.char)
        except ValueError:
            return
        scroll = self.class_listbox.yview()[0]
        selected = (i - 1) % 10
        class_idx = int(round(len(self.class_names) * scroll)) + selected
        self.select_class(class_idx)

    def swap_color(self, *_):
        self.clr_pointer = (self.clr_pointer + 1) % len(self.color_texts)
        self.color_btn.configure(text=self.color_texts[self.clr_pointer])
        self.fg, self.bg = self.bg, self.fg
        self.set_color()

    def set_color(self):
        for attr in dir(self):
            field = getattr(self, attr)
            try:
                configure = getattr(field, 'configure')
                try:
                    configure(bg=self.bg, fg=self.fg)
                except:
                    configure(bg=self.bg)
                try:
                    configure(highlightbackground=self.bg, highlightcolor=self.bg)
                except:
                    pass
            except AttributeError:
                pass

    def undo(self, _=None):
        if self.STATE['tracking_box'] is None \
                and self.STATE['resizing_box'] is None \
                and not self.STATE['click'] \
                and len(self.undo_list) > 0:
            undo_action = self.undo_list.pop()
            if undo_action['action'] == 'create_bbox':
                idx = undo_action['id']
                self.del_bbox(idx)
            elif undo_action['action'] == 'change_class':
                idx = undo_action['id']
                self.class_list[idx] = undo_action['label']
                self.color_list[idx] = undo_action['color']
                self.change_bbox(idx)
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

                _coords = [int(round(c * self.k * self.scale)) for c in bbox]
                self.bbox_id_list.insert(idx, self.draw_bbox(_coords, width=2, outline=color, text=label))
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

    def append_new_data(self):
        potential = self.files - self.loaded_names
        if not potential:
            return
        task_id = min(potential)
        self.loaded_names.add(task_id)

        img_path = os.path.join(self.path, task_id)
        with open(img_path, 'rb') as f:
            image = f.read()
        ann_path = img_name_to_annotation(img_path)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                head, objects = parse_annotation(f.read())
            classes = [obj['name'] for obj in objects]
            size = [float(head[0]['width']), float(head[0]['height'])]
            boxes = [[float(c) / size[i % 2] for i, c in enumerate(obj['bbox'])] for obj in objects]
        else:
            boxes = []
            classes = []
        self.images.append(image)
        self.labels.append((boxes, classes))
        self.tasks.append(task_id)
        self.labeled_list.append(False)
        return

    def load_dir(self, _=False, path=None):
        self.cur = 0
        self.loaded = -1
        self.pointer = 1
        self.max_pointer = 1
        self.images.clear()
        self.labels.clear()
        self.tasks.clear()
        self.labeled_list.clear()
        self.loaded_names.clear()
        if path is None:
            path = self.entry.get()

        with open(cache_path, 'w+') as f:
            f.write(path)
        self.path = path
        self.files = {p for p in os.listdir(self.path) if p.rsplit('.', 1)[-1] in FORMAT}

        self.append_new_data()
        self.update_bar()
        self.load_image()
        self.parent.focus()

    def _bubble_class(self, name, move_selection=True):
        if move_selection:
            try:
                sel = self.class_names[self.class_listbox.curselection()[0]]
            except IndexError:
                sel = None
        else:
            sel = None
        try:
            idx = self.class_names.index(name)
        except ValueError:
            return
        self.class_listbox.delete(idx)
        self.class_listbox.insert(0, name)
        self.class_names.insert(0, self.class_names.pop(idx))
        self.class_colors.insert(0, self.class_colors.pop(idx))
        self.class_listbox.itemconfig(0, fg=self.class_colors[0])
        if sel is not None:
            sel_idx = self.class_names.index(sel)
            self.select_class(sel_idx)

    def comparator(self, name):
        return -self.class_usage.get(name, 0), name

    def bubble_classes(self, names: set):
        if not isinstance(names, set):
            names = set(names)

        order = []
        order.extend(sorted(names, key=self.comparator))
        order.extend(sorted(set(self.class_names) - names, key=self.comparator))

        for name in order[::-1]:
            self._bubble_class(name)

    def load_labels(self, _=None, provided_label=None):
        if self.cur >= len(self.labels):
            return
        self.clear_bbox()
        boxes, classes = provided_label or self.labels[self.cur]
        shape = [self.img.width, self.img.height] * 2
        for bbox, label in zip(boxes, classes):
            scaled = [c * k for c, k in zip(bbox, shape)]
            if label not in self.class_names:
                self.add_classes(label)
            color = self.get_color(label)
            self.class_list.append(label)
            self.bbox_list.append(scaled)
            self.color_list.append(color)
            _bbox = self._to_real_coords(scaled)
            rect_id = self.draw_bbox(_bbox, width=2, outline=color, text=label)
            self.bbox_id_list.append(rect_id)
        self.bubble_classes(set(classes))

    def add_classes(self, class_name):
        if not class_name or len(class_name) > 32:
            return
        self.class_listbox.insert(0, class_name)
        fg = rgb2hex(colors[len(self.class_colors) % len(colors)])
        self.class_names.insert(0, class_name)
        self.class_colors.insert(0, fg)
        self.class_listbox.itemconfig(0, fg=fg)

    def press_class_btn(self, new_class=None):
        if new_class is None:
            new_class = self.class_entry.get()
        new_class = ' '.join(new_class.split())
        if not new_class:
            return
        self.add_classes(new_class)
        self.class_listbox.focus()
        idx = self.class_names.index(new_class)
        self.class_listbox.see(idx)
        self.select_class(idx)
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
                                                         Image.Resampling.NEAREST))
        if self.img_id is not None:
            self.main_panel.delete(self.img_id)
        self.img_id = self.main_panel.create_image(self.offset[0], self.offset[1], image=self.tk_img, anchor=tk.NW)

    def update_bar(self):
        return self.progress_bar.config(text="%04d/%04d" % (self.pointer, len(self.files)))

    def _load_image(self):
        if self.cur >= len(self.images) or self.loaded == self.cur:
            return
        self.img = Image.open(io.BytesIO(self.images[self.cur]))
        self.img_shape = self.img.width, self.img.height
        self.update_resolution(self.img.width, self.img.height)
        self.loaded = self.cur
        self.last_load = time.monotonic()

    def load_image(self, load_labels=True):
        if time.monotonic() - self.last_load > .1:
            self._load_image()
        if load_labels:
            self.load_labels()

    def save_image(self, *_):
        boxes = self.bbox_list
        shape = [self.img.width, self.img.height] * 2
        relative = [[round(c / shape[i], 5) for i, c in enumerate(box)] for box in boxes]
        classes = list(self.class_list)
        task = self.tasks[self.cur]
        changed = self.labels[self.cur] != (relative, classes)
        self.labels[self.cur] = (relative, classes)
        if changed or not self.labeled_list[self.cur]:
            ann_path = img_name_to_annotation(os.path.join(self.path, task))
            scaled_boxes = [
                [int(round(self.img_shape[i % 2] * c)) for i, c in enumerate(box)]
                for box in relative
            ]
            ann = create_annotation(task, classes, scaled_boxes, self.img_shape[0], self.img_shape[1])
            os.makedirs(os.path.dirname(ann_path), exist_ok=True)
            with open(ann_path, 'w+') as f:
                f.write(ann)
            self.labeled_list[self.cur] = True

    def remove_target_lines(self):
        if self.vl:
            self.main_panel.delete(self.vl)
        if self.hl:
            self.main_panel.delete(self.hl)

    def toggle_box_creation(self, _=None):
        self.clear_points()
        self.STATE['create'] ^= True
        if self.STATE.get("yaro_mod", False):
            self.STATE['create'] = True
        self.STATE['click'] = False
        self.default_cursor = 'tcross' if self.STATE['create'] else ''
        self.main_panel.config(cursor=self.default_cursor)
        self.remove_target_lines()

    def toggle_yaro_mod(self, _=None):
        self.STATE['yaro_mod'] ^= True

    def clear_points(self):
        for point in self.points:
            self.main_panel.delete(point)
        self.points.clear()

    @staticmethod
    def get_edge_points(bbox):
        x_center = (bbox[2] + bbox[0]) // 2
        y_center = (bbox[3] + bbox[1]) // 2
        coords = []
        for i in range(0, 4, 2):
            coords.extend((bbox[i], bbox[j]) for j in range(1, 4, 2))
            coords.extend(((bbox[i], y_center), (x_center, bbox[i + 1])))
        return coords

    @staticmethod
    def point_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def get_closest_box(self, mouse_pos):
        closest_box = None
        min_distance = float('inf')
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
        return [
            int(round(c * (self.k * self.scale) + self.offset[i % 2]))
            for i, c in enumerate(coords)
        ]

    def _from_real_coords(self, coords):
        return [
            int((c - self.offset[i % 2]) / (self.k * self.scale))
            for i, c in enumerate(coords)
        ]

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

    def get_sel_label(self, fail_safe=False):

        sel = self.class_listbox.curselection()
        if len(sel) == 0:
            if not fail_safe:
                messagebox.showerror("Please select a class",
                                        "You should select a class you are trying to add from the right listbox")
            return '' if fail_safe else None
        elif len(sel) > 2:
            if not fail_safe:
                messagebox.showerror("Please select only 1 class",
                                        "You should select 1 class you are trying to add from the right listbox")
            return '' if fail_safe else None
        return self.class_names[sel[0]]

    def mouse_release(self, event):
        self.change_class(event)
        self.STATE['changing_class'] = False
        self.clear_points()
        x, y = self.get_pos(event)
        if self.STATE['resizing_box'] is not None:
            idx = self.STATE['resizing_box'][0]
            self.undo_list.append({'action': 'resize',
                                   'id': idx,
                                   'initial': self.STATE['resizing_box'][5:9]})
            self.bbox_list[idx][0], self.bbox_list[idx][2] = sorted([self.bbox_list[idx][0], self.bbox_list[idx][2]])
            self.bbox_list[idx][1], self.bbox_list[idx][3] = sorted([self.bbox_list[idx][1], self.bbox_list[idx][3]])
            self.STATE['resizing_box'] = None
        elif self.STATE['tracking_box'] is not None:
            self.undo_list.append({'action': 'move',
                                   'id': self.STATE['tracking_box'][0],
                                   'initial': self.STATE['tracking_box'][3:7]})
            self.STATE['tracking_box'] = None
        elif self.STATE['click']:
            label = self.STATE['label']
            if label is not None:
                _x, _y = x + self.offset[0], y + self.offset[1]
                x1, x2 = min(self.STATE['x'], _x), max(self.STATE['x'], _x)
                y1, y2 = min(self.STATE['y'], _y), max(self.STATE['y'], _y)
                self.class_usage[label] = 1 + self.class_usage.get(label, 0)
                self.class_list.append(label)
                bbox = self._from_real_coords([x1, y1, x2, y2])
                self.bbox_list.append(bbox)
                self.color_list.append(self.STATE['cur_color'])
                del self.STATE['cur_color']
                self.undo_list.append({'action': 'create_bbox',
                                       'id': len(self.bbox_id_list)})
                self.bbox_id_list.append(self.bbox_id)
                self.bbox_id = None
                self.toggle_box_creation()
        if len(self.undo_list) > 50:
            self.undo_list.pop(0)

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
        self.del_bbox(closest_box, save=True)

    def change_class(self, event):
        if not self.STATE['changing_class']:
            return
        mouse_pos = self.get_mouse_pos(event)
        closest_box, distance, inside = self.get_closest_box(mouse_pos)
        if closest_box is None:
            return
        label = self.get_sel_label(fail_safe=True)
        if not label:
            self.STATE['changing_class'] = False
            return
        if self.class_list[closest_box] == label:
            return
        self.class_usage[label] = 1 + self.class_usage.get(label, 0)
        old_label = self.class_list[closest_box]
        self.class_list[closest_box] = label
        old_color = self.color_list[closest_box]
        self.color_list[closest_box] = self.get_color(label)
        self.change_bbox(closest_box)
        self.undo_list.append({'action': 'change_class',
                               'id': closest_box,
                               'label': old_label,
                               'color': old_color
                               })
        if len(self.undo_list) > 50:
            self.undo_list.pop(0)

    def mouse_click(self, event):
        self.clear_points()
        if event.state & Modifiers.CONTROL:
            self.STATE['changing_class'] = True
            self.change_class(event)
        else:
            x, y = self.get_pos(event)
            if self.STATE['resizing_box'] is not None or self.STATE['tracking_box'] is not None:
                self.mouse_release(event)
            elif self.STATE['create']:
                if self.STATE['click']:
                    self.mouse_release(event)
                else:
                    _x, _y = x + self.offset[0], y + self.offset[1]
                    self.STATE['x'], self.STATE['y'] = _x, _y

                    self.STATE['label'] = self.get_sel_label()
                    if self.STATE['label'] is not None:
                        self.STATE['cur_color'] = self.get_color(self.STATE['label'])
                        self.STATE['click'] = True
                        self.bbox_id = self.draw_bbox([_x, _y, _x, _y], width=2,
                                                      outline=self.STATE['cur_color'],
                                                      text=self.get_sel_label(fail_safe=True))
            else:
                mouse_pos = self.get_mouse_pos(event)
                closest_box, distance, inside = self.get_closest_box(mouse_pos)
                if closest_box is None:
                    return
                bbox = self.bbox_list[closest_box]
                if distance < 2 * self.distance_thresh:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    scores = [
                        bbox[0] + 0.25 * width - mouse_pos[0],
                        bbox[1] + 0.25 * height - mouse_pos[1],
                        mouse_pos[0] - (bbox[2] - 0.25 * width),
                        mouse_pos[1] - (bbox[3] - 0.25 * height)
                    ]
                    threshold = max(sorted(scores)[-3], 0)
                    mask = [score > threshold for score in scores]

                    self.STATE['resizing_box'] = (closest_box, *mask, *bbox)
                elif inside:
                    self.STATE['tracking_box'] = (closest_box,
                                                  (bbox[0] + bbox[2]) / 2 - mouse_pos[0],
                                                  (bbox[1] + bbox[3]) / 2 - mouse_pos[1],
                                                  *bbox)

    def draw_bbox(self, bbox, width, outline, text=''):
        rect_id = self.main_panel.create_rectangle(bbox[0], bbox[1], bbox[2], bbox[3], width=width, outline=outline)
        bbox_ids = [rect_id]
        if text:
            font = tkFont.Font(family="Arial", size=14)
            text_id = self.main_panel.create_text(min(bbox[0], bbox[2]), min(bbox[1], bbox[3]) - 3, text=text,
                                                  anchor=tk.SW, fill=outline, font=font)

            bbox_ids.append(text_id)
        return bbox_ids

    def delete_bbox(self, bbox_id):
        for idx in bbox_id:
            self.main_panel.delete(idx)

    def mouse_move(self, event=None):
        if self.loaded != self.cur:
            self._load_image()
            self.load_labels()
        self.change_class(event)
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
            self.delete_bbox(self.bbox_id)
        if self.STATE['resizing_box'] is not None:
            box_id, x1_mask, y1_mask, x2_mask, y2_mask, *_ = self.STATE['resizing_box']
            coords = self.bbox_list[box_id]
            mask = [x1_mask, y1_mask, x2_mask, y2_mask]
            for i in range(4):
                if mask[i]:
                    coords[i] = mouse_pos[i % 2]
            if any(mask):
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
            self.bbox_id = self.draw_bbox([self.STATE['x'], self.STATE['y'], _x, _y], width=2,
                                          outline=self.STATE['cur_color'], text=self.get_sel_label(fail_safe=True))
        else:
            closest_box, *_ = self.get_closest_box(mouse_pos)
            self.set_focus(closest_box)

    def cancel_bbox(self, _=None):
        if self.STATE['click'] and self.bbox_id:
            self.delete_bbox(self.bbox_id)
            self.bbox_id = None
            self.STATE['click'] = False

    def change_bbox(self, idx):
        _bbox_idx = self.bbox_id_list[idx]
        if _bbox_idx is not None:
            self.delete_bbox(_bbox_idx)
        _coords = self._get_real_bbox(idx)
        self.bbox_id_list[idx] = self.draw_bbox(_coords, width=2, outline=self.color_list[idx],
                                                text=self.class_list[idx])

    def del_bbox(self, idx: int = None, save: bool = False):
        self.delete_bbox(self.bbox_id_list[idx])
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
            if len(self.undo_list) > 50:
                self.undo_list.pop(0)
        self.redraw_boxes()

    def clear_bbox(self, *_):
        for idx in range(len(self.bbox_id_list)):
            self.delete_bbox(self.bbox_id_list[idx])
        self.bbox_id_list = []
        self.class_list = []
        self.bbox_list = []
        self.color_list = []
        self.undo_list = []
        # Do garbage collection every 5 minutes
        if time.monotonic() - self.last_clear > 300:
            gc.collect()
            self.last_clear = time.monotonic()

    @staticmethod
    def event_ping(event):
        return int(time.monotonic() * 1e3) - event.time if hasattr(event, 'time') and event.time != 0 else 0.0

    def prev_image(self, event=None):
        if self.event_ping(event) > MAX_PING:
            return
        self.save_image()
        if self.cur > 0:
            self.cur -= 1
            self.pointer -= 1
            self.update_bar()
        self.load_image()

    def next_image(self, event=None, load_labels=True):
        if self.event_ping(event) > MAX_PING:
            print(f"event droped because of ping: {self.event_ping(event)}ms")
            return
        self.save_image()
        if self.cur >= len(self.images) - 1:
            self.append_new_data()

        if self.cur < len(self.images) - 1:
            self.pointer += 1
            self.max_pointer = max(self.max_pointer, self.pointer)
            self.cur += 1
            self.update_bar()
            self.load_image(load_labels=load_labels)
        else:
            self.load_image()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def track(self, prev_img: Image, cur_img: Image, rel_boxes: List[Tuple[float, float, float]],
              prev_classes: List[str]) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
        return self.predictor.track(prev_img, cur_img, rel_boxes, prev_classes)

    def predict_next_image(self, _=None):
        boxes = self.bbox_list
        shape = [self.img.width, self.img.height] * 2
        relative = [[round(c / shape[i], 5) for i, c in enumerate(box)] for box in boxes]
        classes = list(self.class_list)
        prev_img = self.img
        self.next_image(_, load_labels=False)
        cur_img = self.img

        tracked = self.track(prev_img, cur_img, relative, classes)
        self.load_labels(provided_label=tracked)


def main():
    root = tk.Tk()
    Labelfficient(root)
    root.mainloop()


if __name__ == '__main__':
    main()
