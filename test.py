from PIL import Image, ImageDraw
from google.cloud import vision_v1

import pickle
import re
from typing import List

with open('result', 'rb') as f:
    document: vision_v1.TextAnnotation = pickle.load(f)

columns = [350, 546, 818, 990, 1202, 1402, 1629, 1755]
rows    = [52, 105, 157, 210, 262, 315, 368, 420, 473, 509]

subcategory = "passer"

def draw_boxes(image, bounds, color):
    draw = ImageDraw.Draw(image)

    for symbols, bound in bounds:
        word = ''.join([x.text for x in symbols])

        draw.polygon(
            [
                bound.vertices[0].x, bound.vertices[0].y,
                bound.vertices[1].x, bound.vertices[1].y,
                bound.vertices[2].x, bound.vertices[2].y,
                bound.vertices[3].x, bound.vertices[3].y,
            ],
            outline='blue' if word == '2' else color
        )

    return image

def does_cross(bound: vision_v1.BoundingPoly):
    vertices = bound.vertices
    for i in range(len(vertices)):
        y1 = vertices[i].y
        y2 = vertices[(i + 1) % len(vertices)].y

        for target in rows:
            if (y1 <= target <= y2) or (y2 <= target <= y1):
                return (True, target)
            
    return (False, None)

def create_bound(symbols: List[vision_v1.Symbol]) -> vision_v1.BoundingPoly:
    if not symbols:
        return vision_v1.BoundingPoly(vertices=[])
    
    x_coords = [v.x for symbol in symbols for v in symbol.bounding_box.vertices]
    y_coords = [v.y for symbol in symbols for v in symbol.bounding_box.vertices]

    return vision_v1.BoundingPoly(vertices=[
        vision_v1.Vertex(x=min(x_coords), y=min(y_coords)),
        vision_v1.Vertex(x=max(x_coords), y=min(y_coords)),
        vision_v1.Vertex(x=max(x_coords), y=max(y_coords)),
        vision_v1.Vertex(x=min(x_coords), y=max(y_coords))
    ])

def split_bound(symbols: List[vision_v1.Symbol], row: int):
    above_symbols = []
    below_symbols = []

    for symbol in symbols:
        ys = [v.y for v in symbol.bounding_box.vertices]

        if all(y < row for y in ys):
            above_symbols.append(symbol)
        elif all(y > row for y in ys):
            below_symbols.append(symbol)
        else:
            return []
        
    return [(above_symbols, create_bound(above_symbols)), (below_symbols, create_bound(below_symbols))]

image  = Image.open('cropped.png')
bounds = []

for page in document.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                bounds.append((word.symbols, word.bounding_box))

all_new_bounds = []
for symbols, bound in bounds:
    crosses, row = does_cross(bound)
    if crosses:
        split_bounds = split_bound(symbols, row)

        for split_bound_ in split_bounds:
            all_new_bounds.append(split_bound_)

    else:
        all_new_bounds.append((symbols, bound))

bounds = all_new_bounds
image  = draw_boxes(image, bounds, 'green')

sections = {x: [] for x in rows}
for symbols, bound in bounds:
    y = max(v.y for v in bound.vertices)
    for pos in rows:
        if y < pos:
            sections[pos].append((symbols, bound))
            break

sections = [sorted(section, key=lambda item: item[1].vertices[0].x) for section in sections.values()]
groups   = {row: {col: [] for col in columns} for row in rows}
entries  = set()

for i, section in enumerate(sections):
    for symbols, bound in section:
        x = min(v.x for v in bound.vertices)
        for col in columns:
            if x < col:
                entry = (rows[i], col, ''.join(x.text for x in symbols))

                if entry not in entries:
                    groups[rows[i]][col].append(''.join(x.text for x in symbols))
                    entries.add(entry)

                break

if subcategory == "passer":
    print("{:<30} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format('Player', 'QBR', 'Comp.', 'Att.', 'TDs', 'Ints', 'Sacks', 'Yards', 'Long', 'Conflicts'))
    
    for cols in groups.values():
        name, qbr, comp_str, tds, ints, sacks, yards, long, *conflicts = (''.join(x) for x in cols.values())
        name = re.sub(r'(\d+)?@', '', name)
        comp, att = re.search(r'\d+%\((\d+)\/(\d+)\)', comp_str).groups()
        print(f"{name:<30} {qbr:^10} {comp:^10} {att:^10} {tds:^10} {ints:^10} {sacks:^10} {yards:^10} {long:^10} {conflicts or ''}")

image.save('result.png')
