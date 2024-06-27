import asyncio
import cv2
import itertools
import numpy
import pickle
import pytesseract
import re

from google.cloud import vision_v1
from shapely.geometry import Polygon
from typing import List, Dict, Any

AMOUNTS = {
    "passer": 8,
    "receiver": 8,
    "corner": 8,
    "defender": 7
}
REPLACEMENTS = ['o', 'о']

with open('result', 'rb') as f:
    document: vision_v1.TextAnnotation = pickle.load(f)

GRID_TEXT = """
[0, 402, 566, 778, 990, 1202, 1414, 1642, 1755]
[0, 52, 105, 157, 210, 262, 315, 368, 420, 473, 509]
"""

columns = [int(x) for x in re.findall(r'\d+', GRID_TEXT.split('\n')[1])]
rows = [int(x) for x in re.findall(r'\d+', GRID_TEXT.split('\n')[2])]

subcat = "defender"

grid: Dict[int, Dict[int, List[vision_v1.Symbol]]] = {
    row: {col: [] for col in columns} for row in rows
}

entries = set()
for page in document.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                row = next(r for r in rows if word.bounding_box.vertices[-1].y < r)
                col = next(c for c in columns if word.bounding_box.vertices[0].x < c)
                entry = (row, col, ''.join(x.text for x in word.symbols))

                if entry in entries:
                    continue
                entries.add(entry)
                
                for symbol in word.symbols:
                    row = next(r for r in rows if symbol.bounding_box.vertices[-1].y < r)
                    col = next(c for c in columns if symbol.bounding_box.vertices[0].x < c)
                    grid[row][col].append(symbol)

def find_name_index() -> int:
    for row, col in itertools.product(range(1, len(rows)), range(1, len(columns))):
        word = ''.join([symbol.text for symbol in grid[rows[row]][columns[col]]]).strip()
        if re.search(r'(\d+)?\s?@', word):
            return col
        
async def process_cell(image: numpy.ndarray, name_index: int, row: int, col: int) -> Dict[str, Any]:
    result = {'cell': (row, col), 'text': '', 'confidence': 0, 'is_name': name_index == columns.index(col)}

    confidence = 0
    if grid[row][col]:
        confidence = numpy.average([x.confidence for x in grid[row][col]])
        polygons   = [Polygon([(vertex.x, vertex.y) for vertex in symbol.bounding_box.vertices]) for symbol in grid[row][col]]

        for first_polygon, second_polygon in itertools.combinations(polygons, 2):
            if (first_polygon.intersection(second_polygon).area / second_polygon.area) >= 0.5:
                confidence = 0
                break

    if not result['is_name'] and confidence < 0.8:
        roi    = image[rows[rows.index(row)-1]+2:row-2, columns[columns.index(col)-1]+2:col-2]
        output = await asyncio.to_thread(pytesseract.image_to_data, image=roi, lang='eng', config='--oem 1 --psm 10', output_type=pytesseract.Output.DICT)

        output_data = [(output['text'][index], output['conf'][index]) for index in range(len(output['text'])) if output['text'][index] and output['conf'][index] > 75]

        if output_data:
            result['text'] = ''.join([x[0] for x in output_data]).strip() + 'ₚ'
            result['confidence'] = numpy.mean([x[1] for x in output_data]) / 100

        if ')' in result['text']:
            print(output)

        return result
    
    for symbol in grid[row][col]:
        if not result['is_name'] and symbol.text.lower() in REPLACEMENTS:
            symbol.text = '0'

        result['text'] += symbol.text

    if grid[row][col]:
        result['confidence'] = numpy.mean([symbol.confidence for symbol in grid[row][col]])

    return result

async def main():  
    image      = cv2.imread('cropped.png')
    name_index = find_name_index()

    results = await asyncio.gather(*[
        process_cell(image, name_index, rows[row], columns[col]) for row, col in itertools.product(range(1, len(rows)), range(name_index, name_index + AMOUNTS.get(subcat)))
    ])

    for _, data in itertools.groupby(results, key=lambda result: result['cell'][0]):
        data = sorted(data, key=lambda result: result['cell'][1])
        
        for i in range(len(data)):
            if i == 0:
                print(f"{data[i]['text'].replace('@', ' @'):<22}", end=' ')
            else:
                if data[i]['text']:
                    print(f"{data[i]['text'].lower():^10}", end=' ')
                else:
                    print(f"{data[i]['text'].lower() or '❌':^9}", end=' ')
        print()

    print(f"\nConfidence: {numpy.mean([x['confidence'] for x in results]):.2%}")

asyncio.run(main())