import cv2
import numpy
import pickle
import pytesseract
import re

from google.cloud import vision_v1
from typing import List, Dict

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

with open('result', 'rb') as f:
    document: vision_v1.TextAnnotation = pickle.load(f)

grid_text = """
[0, 358, 546, 786, 1002, 1202, 1414, 1626, 1755]
[0, 52, 105, 157, 210, 262, 315, 368, 420, 473, 509]
"""

columns = [int(''.join([x for x in num if x.isdigit()])) for num in grid_text.split('\n')[1].split(', ')]
rows    = [int(''.join([x for x in num if x.isdigit()])) for num in grid_text.split('\n')[2].split(', ')]

grid: Dict[int, Dict[int, List[vision_v1.Symbol]]] = {
    row: {
        col: [] for col in columns
    } for row in rows
}

entries = set()
for page in document.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                row   = [row for row in rows if word.bounding_box.vertices[-1].y < row][0]
                col   = [col for col in columns if word.bounding_box.vertices[0].x < col][0]
                entry = (row, col, ''.join(x.text for x in word.symbols))

                if entry in entries:
                    continue

                entries.add(entry)

                for symbol in word.symbols:
                    row = [row for row in rows if symbol.bounding_box.vertices[-1].y < row][0]
                    col = [col for col in columns if symbol.bounding_box.vertices[0].x < col][0]

                    grid[row][col].append(symbol)
                    
image = cv2.imread('cropped.png')

for i in range(1, len(rows)):
    for j in range(1, len(columns)):
        if not grid[rows[i]][columns[j]]:
            confidence = 0
        else:
            confidence = numpy.average([x.confidence for x in grid[rows[i]][columns[j]]])

        if confidence < 0.8:
            roi    = image[rows[i-1]+2:rows[i]-2, columns[j-1]+2:columns[j]-2]
            output = pytesseract.image_to_string(roi, lang='eng', config='--oem 1 --psm 10')
            print(f"'{output.strip()}'", end=', ')
            continue

        for symbol in grid[rows[i]][columns[j]]:
            print(symbol.text, end='')
            vertices = numpy.array([[vertex.x, vertex.y] for vertex in symbol.bounding_box.vertices], numpy.int32)

            cv2.polylines(
                image,
                pts = [vertices.reshape((-1, 1, 2))],
                color = (255, 255, 255) if symbol.confidence > 0.95 else (0, 255, 0) if symbol.confidence > 0.90 else (0, 255, 255) if symbol.confidence > 0.85 else (0, 128, 255) if symbol.confidence > 0.8 else (0, 0, 255),
                thickness = 1,
                isClosed = True
            )
        print(', ', end='')
    print()

cv2.destroyAllWindows()

cols = [grid[row][col] for row in rows for col in grid[row]]
conf = numpy.average([symbol.confidence for col in cols for symbol in col])
print(f"Confidence: {conf:.2%}")

cv2.imwrite('result.png', image)