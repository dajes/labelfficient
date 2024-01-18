<img src=".github/labelfficient-logo.svg" width="600">

Labelfficient is a simple yet customizable and efficient tool for creating object detection labels.


# Installation
#### The tool was tested on Windows 10 and Ubuntu 18, python 3.6+

Go to the project directory

### For minimalistic version (without tracking):
```
pip install pillow
```
Then when you have PIL you can run the GUI!
```
python simple_gui.py
```

### For advanced (with tracking)

### Install dependencies 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Run the GUI
```
python gui.py
```


# About Labelfficient functionality

---
## Tracking
Labelefficient has an object tracker from the [SiamMask](https://github.com/foolwood/SiamMask) repository, which is 
really helpful when labeling similar pictures, frames from a video as an example.<br>

### Here is a GIF demonstrating the speed difference of video annotation with and without tracker:<br>

<div align="center">
  <img src="https://user-images.githubusercontent.com/18720858/107127200-2f80a600-68bd-11eb-862b-8209448cdc8d.gif"/>
</div>

***Note that manual labeling required approximately 1 click per second which is pretty hard to do for a long time. <br>
Meanwhile with a tracking assistance the major part of time was elapsed on calculating tracker forward pass
and checking the results, so intensity of clicking is at least 4 times smaller and total elapsed time is 1.6 times 
smaller in this example.***

---
## Hotkeys
Labelfficient main priority is labeling with a minimal amount of effort, so the tool design is heavily based on using 
hotkeys and almost all functionality of the tool is available only by using hotkeys.

### List of all hotkeys and of their use cases
|         Key |                                                                                                   Description |
|------------:|--------------------------------------------------------------------------------------------------------------:|
|           N |                                                                              Start/stop creating bounding box |
|           S |                                                       Reset bounding box if one is being drawn at the moment. |
| Esc         |               If some text field is active, remove focus from it. Also resets bounding box as the former one. |
|         A/D |                                                                        Load previous/next image from sequence |
| G           |                                               Predict position of each of the current boxes on the next frame |
| C           |                                                 Copy position and class of the selected box to the next frame |
| V           |                                                        Predict position of the selected box on the next frame |
| F           |  Find outlier image and move to it, works only when a directory was  loaded with pixel sorting option enabled |
| R           |                                                                                  Reload annotations from disk |
| Ctrl-Z      |        Undo last action (up to 50 at a time). Warning! When loading another image, history is being  cleared. |
| Ctrl-S      |                             Save annotation for the current image. Autosaved when switching to another image. |
| Ctrl-Scroll |                                                                             Zoom in and out using mouse wheel |
| Right-Click |                                                           Deletes a bounding box if the cursor points on any. |
| Ctrl-Click  | Until the mouse key is released, all boxes that the mouse is hovered on are changed to the active class name. |

---

## Bounding box editing
Bounding boxes can be easily corrected by using only your mouse.<br>
### To modify a box, hover the mouse over the box and do one of the actions:

| Action with a bounding box | How to do                                    |
|----------------------------|----------------------------------------------|
| Delete                     | Right click                                  |
| Move                       | Left-Click on the inner area                 |
| Resize                     | Left-Click on the corner or edge of interest |

***If you are hovering the mouse over several boxes, you can see what box is going to be affected
as it is indicated by dots on corners and edges of the box.<br>
Focused box in this case is determined by proximity of the mouse to an edge of the boxes, the closest one is chosen.***

---

## Annotation format
Labelfficient supports any annotation format that is parsable by regex pattern matching, you can edit the format
by changing the ```annotation_format.txt``` file.
#### If you change the annotation format after creating annotations for some images they can be misinterpreted when loading

Annotation files can be stored in 2 formats, depending on whether the dataset has or has not "Images" subfolder. Examples:

| Image path                | Annotation path                      |
|---------------------------|--------------------------------------|
| ../dataset/Images/cat.jpg | ../dataset/Annotations/cat.xml       |
| ../dataset/cat.jpg        | ../dataset_Annotations/cat.xml       |
| ../dataset/Image/cat.png  | ../dataset/Image_Annotations/cat.xml |


