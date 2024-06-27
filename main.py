import asyncio
import cv2
import discord
import itertools
import numpy
import os
import pickle
import pytesseract
import re

from concurrent.futures import ThreadPoolExecutor
from google.cloud import vision_v1
from shapely.geometry import Polygon
from typing import List, Dict, Any, Optional

import pickle

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"credentials.json"

categories    = ["my stats", "current game", "server", "global"]
subcategories = ["passer", "runner", "receiver", "corner", "defender", "kicker", "other"]

AMOUNTS = {"passer": 8, "receiver": 8, "corner": 8, "defender": 7}
REPLACEMENTS = ['o', 'о']

async def find_stat_category(loop: asyncio.AbstractEventLoop, image):
    category_text = subcategory_text = None

    with ThreadPoolExecutor() as pool:
        _, binary   = await loop.run_in_executor(pool, lambda: cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY))

        edges       = await loop.run_in_executor(pool, lambda: cv2.Canny(binary, 50, 150, apertureSize=3))
        contours, _ = await loop.run_in_executor(pool, lambda: cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

        rectangles = []
        for contour in contours:
            approx = await loop.run_in_executor(pool, lambda: cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True))
            if len(approx) == 4:
                rectangles.append(contour)

        rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)[:5]
        
        if len(rectangles) < 2:
            raise ValueError("Not enough stat categories")

        for i, rect in enumerate(rectangles):
            x, y, w, h = cv2.boundingRect(rect)
            roi = image[y:y+h, x:x+w]

            config = ('-l eng --oem 1 --psm 7')
            output = await loop.run_in_executor(pool, lambda: pytesseract.image_to_data(roi, config=config, output_type='dict'))
            text   = ' '.join([x for x in output['text'] if x])

            if not text:
                continue

            text = text.lower()

            for category in categories:
                if category in text:
                    category_text = category

            for subcategory in subcategories:
                if subcategory in text:
                    subcategory_text = subcategory

    if not category:
        raise ValueError("No category")
    
    if not subcategory:
        raise ValueError("No subcategory")

    return (category_text, subcategory_text)

async def find_stat_box(loop: asyncio.AbstractEventLoop, image) -> numpy.ndarray:
    with ThreadPoolExecutor() as pool:
        image     = await loop.run_in_executor(pool, lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        _, binary = await loop.run_in_executor(pool, lambda: cv2.threshold(image, 127, 255, cv2.THRESH_BINARY))

        contours, _ = await loop.run_in_executor(pool, lambda: cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        
        stat_box = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(stat_box)
        
        return image[y:y+h, x:x+w]
    
def find_columns(stats: numpy.ndarray) -> List[int]:
    col_means   = numpy.mean(stats, axis=0)
    differences = numpy.abs(numpy.diff(col_means))

    if not len(col_means) or not len(differences):
        raise ValueError("There are no columns")

    boundaries = [i + 1 for i, diff in enumerate(differences) if diff > 0]

    if not len(boundaries) > 0:
        raise ValueError("There are no columns")
    
    avg_distance = numpy.mean(numpy.diff(boundaries)) + 10
    columns = []
    for i in range(1, len(boundaries)):
        if boundaries[i] - boundaries[i - 1] > avg_distance:
            columns.append((boundaries[i] + boundaries[i - 1]) // 2)

    if columns and columns[-1] > stats.shape[1] - avg_distance / 2:
        columns.pop()

    return columns
    
def find_rows(stats: numpy.ndarray) -> List[int]:
    row_means   = numpy.mean(stats, axis=1)
    differences = numpy.abs(numpy.diff(row_means))
    
    if not len(row_means) or not len(differences):
        raise ValueError("There are no sections")
    
    boundaries = [i + 1 for i, diff in enumerate(differences) if diff > 19]
    
    if not len(boundaries) > 1:
        raise ValueError("There are not atleast two sections")
    
    return boundaries

def draw_sections(stats: numpy.ndarray, columns: list, rows: list) -> numpy.ndarray:
    for x in columns:
        cv2.line(stats, (x, 0), (x, stats.shape[0]), (0, 0, 255), 2)

    for y in rows:
        cv2.line(stats, (0, y), (stats.shape[1], y), (0, 0, 255), 2)
        
    return stats

async def get_stats(path: str, retreive: bool):
    loop = asyncio.get_event_loop()
    file = discord.File(path)
    
    file_bytes = file.fp.read()
    image      = cv2.imdecode(numpy.frombuffer(file_bytes, numpy.uint8), cv2.IMREAD_COLOR)

    category, subcategory = await find_stat_category(loop, image)

    print(f"      Image: {path}")
    print(f"   Category: {category}")
    print(f"Subcategory: {subcategory}\n")
    
    if category.lower() != "current game":
        raise ValueError(f"The stat category must be current game and not {category.lower()}")
    
    image = await find_stat_box(loop, image)
    stats = await loop.run_in_executor(None, lambda: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    columns  = find_columns(image)
    rows     = find_rows(image)
    sections = draw_sections(stats, columns, rows)

    if retreive:
        await send_to_google(sections, subcategory, [0] + columns + [sections.shape[1]], [0] + rows + [sections.shape[0]])
    else:
        cv2.imshow(path, sections)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

def find_name_index(grid, rows: List[int], columns: List[int]) -> int:
    for row, col in itertools.product(range(1, len(rows)), range(1, len(columns))):
        word = ''.join([symbol.text for symbol in grid[rows[row]][columns[col]]]).strip()
        if re.search(r'(\d+)?\s?@', word):
            return col
        
async def process_cell(image: numpy.ndarray, grid, rows: List[int], columns: List[int], name_index: int, row: int, col: int) -> Dict[str, Any]:
    result = {'cell': (row, col), 'text': '', 'confidence': 0, 'is_name': name_index == columns.index(col)}

    if grid[row][col]:
        result['confidence'] = numpy.average([x.confidence for x in grid[row][col]])
        polygons   = [Polygon([(vertex.x, vertex.y) for vertex in symbol.bounding_box.vertices]) for symbol in grid[row][col]]

        for first_polygon, second_polygon in itertools.combinations(polygons, 2):
            if (first_polygon.intersection(second_polygon).area / second_polygon.area) >= 0.5:
                result['confidence'] = 0
                break

    if not result['is_name'] and result['confidence'] < 0.8:
        roi    = image[rows[rows.index(row)-1]+2:row-2, columns[columns.index(col)-1]+2:col-2]
        output = await asyncio.to_thread(pytesseract.image_to_data, image=roi, lang='eng', config='--oem 1 --psm 10', output_type=pytesseract.Output.DICT)

        output = [(output['text'][index], output['conf'][index]) for index in range(len(output['text'])) if output['text'][index] and output['conf'][index] > 75]

        if output:
            result['text'] = ''.join([x[0] for x in output]).strip() + 'ₚ'
            result['confidence'] = numpy.mean([x[1] for x in output]) / 100

        return result
    
    for symbol in grid[row][col]:
        if not result['is_name'] and symbol.text.lower() in REPLACEMENTS:
            symbol.text = '0'

        result['text'] += symbol.text

    if grid[row][col]:
        result['confidence'] = numpy.mean([symbol.confidence for symbol in grid[row][col]])

    return result
    
async def send_to_google(sections, subcategory: str, columns: list, rows: list):
    client  = vision_v1.ImageAnnotatorAsyncClient()
    
    image   = vision_v1.types.Image(content=cv2.imencode('.png', sections)[1].tobytes())
    feature = vision_v1.types.Feature(type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)
    request = vision_v1.AnnotateImageRequest(image=image, features=[feature])
    
    response = await client.batch_annotate_images(requests=[request])

    if False:
        with open('result', 'wb') as f:
            pickle.dump(response.responses[0].full_text_annotation, f)

        print(columns)
        print(rows)

        cv2.imwrite('cropped.png', sections)

    grid: Dict[int, Dict[int, List[vision_v1.Symbol]]] = {
        row: {col: [] for col in columns} for row in rows
    }

    entries = set()
    for page in response.responses[0].full_text_annotation.pages:
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

    name_index = find_name_index(grid, rows, columns)
    results    = await asyncio.gather(*[
        process_cell(sections, grid, rows, columns, name_index, rows[row], columns[col]) 
        for row, col in itertools.product(range(1, len(rows)), range(name_index, name_index + AMOUNTS.get(subcategory)))
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
    
async def main(retreive: bool, path: Optional[str]=None):
    if path:
        return await get_stats(path=path, retreive=retreive)

    for path in os.listdir('stats'):
        await get_stats(path=f"stats/{path}", retreive=retreive)
        print()

asyncio.run(main(retreive=True))