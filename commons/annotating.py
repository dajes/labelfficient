import os
import re


def img_name_to_annotation(img_name: str) -> str:
    """
    Tries to get annotation path from the image path

    Args:
        img_name: path to the image

    Returns:
        str: probable path to annotation

    """
    img_name = '.'.join(img_name.split('.')[:-1] + [annotation_format['file']])
    img_name = img_name.replace('\\', '/').split('/')
    if 'Images' in img_name:
        img_name[len(img_name) - 1 - img_name[::-1].index('Images')] = 'Annotations'
    else:
        img_name[-2] += '_Annotations'
    img_name = '/'.join(img_name)
    return img_name


def to_pattern(format_string):
    format_string = format_string.replace('/', r'\/')
    fields = list(re.findall(r'{(.*)}', format_string))

    for field in fields:
        format_string = format_string.replace('{' + field + '}', "(?P<{}>.*)".format(field))

    return format_string, fields


def parse_annotation(annotation):
    head = re.findall(annotation_pattern['head'][0], annotation)
    objects = re.findall(annotation_pattern['object'][0], annotation)
    head = [{k: v for k, v in zip(annotation_pattern['head'][1], _head)} for _head in head]
    objects = [{k: v for k, v in zip(annotation_pattern['object'][1], obj)} for obj in objects]
    for obj in objects:
        for field in {'xmin', 'xmax', 'ymin', 'ymax'}:
            obj[field] = float(obj[field])
        obj['bbox'] = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
    return head, objects


essential_fields = {'{name}', '{xmin}', '{ymin}', '{xmax}', '{ymax}'}


def create_annotation(name: str, labels: str, bboxes, width: int,
                      height: int) -> str:
    """
    Create raw annotation from the values

    Args:
        name: image name
        labels: list of class names
        bboxes: list of boxes
        width: width of the image
        height: height of the image

    Returns:
        str: raw annotation

    """
    assert len(bboxes) == len(labels)
    annotation = annotation_format['head'].format(name=name, width=width, height=height)

    for label, bbox in zip(labels, bboxes):
        _format = annotation_format['object']
        while True:
            try:
                annotation += _format.format(name=label, xmin=bbox[0],
                                             ymin=bbox[1], xmax=bbox[2], ymax=bbox[3])
            except KeyError as e:
                _format = _format.replace('{' + e.args[0] + '}', '')
            else:
                break

    annotation += annotation_format['ending']
    return annotation


def get_annotation_format():
    if not os.path.exists('annotation_format.txt'):
        with open('annotation_format.txt', 'w') as f:
            f.write('''------------------------FILE FORMAT----------------------------------------------
xml
------------------------HEAD-----------------------------------------------------
<annotation>
    <filename>{name}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
    </size>
-----------------------FOR EACH OBJECT--------------------------------------------
    <object>
        <name>{name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
----------------------ENDING-----------------------------------------------------
</annotation>''')
    with open('annotation_format.txt', 'r') as f:
        annotation_format = f.read()
    lines = annotation_format.split('\n')
    file_format_line = None
    head_lines = None
    object_lines = None
    ending_lines = None
    for i, line in enumerate(lines):
        if '--FILE FORMAT--' in line:
            file_format_line = i + 1
        if '--HEAD--' in line:
            head_lines = i + 1
        if '--FOR EACH OBJECT--' in line:
            object_lines = i + 1
        if '--ENDING--' in line:
            ending_lines = i + 1

    assert file_format_line is not None, 'Could not find format line in annotation_format.txt'
    assert head_lines is not None, 'Could not find head line in annotation_format.txt'
    assert object_lines is not None, 'Could not find head line in annotation_format.txt'

    all_lines = [file_format_line - 1, head_lines - 1, object_lines - 1,
                 ending_lines - 1 if ending_lines is not None else 0, len(lines)]
    head_lines = (head_lines, min(filter(lambda x: x > head_lines, all_lines)))
    object_lines = (object_lines, min(filter(lambda x: x > object_lines, all_lines)))

    _format = {
        'file': lines[file_format_line].strip(),
        'head': '\n'.join(lines[head_lines[0]:head_lines[1]]),
        'object': '\n' + '\n'.join(lines[object_lines[0]:object_lines[1]])
    }

    for field in essential_fields:
        assert field in _format['object'], 'Objects must store the {} field'.format(field)

    if ending_lines is not None:
        ending_lines = (ending_lines, min(filter(lambda x: x > ending_lines, all_lines)))
        _format['ending'] = '\n' + '\n'.join(lines[ending_lines[0]:ending_lines[1]])
    else:
        _format['ending'] = ''

    return _format


annotation_format = get_annotation_format()
annotation_pattern = {k: to_pattern(v) for k, v in annotation_format.items()}
