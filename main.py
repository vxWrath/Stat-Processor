import asyncio
import cv2
import discord
import numpy
import pytesseract
import os
import re

from concurrent.futures import ThreadPoolExecutor
from google.cloud import vision_v1
from typing import List, Optional

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"credentials.json"

categories    = ["my stats", "current game", "server", "global"]
subcategories = ["passer", "runner", "receiver", "corner", "defender", "kicker", "other"]

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

    print("Preparing Image...")
    category, subcategory = await find_stat_category(loop, image)

    print(f"   Category: {category}")
    print(f"Subcategory: {subcategory}")
    
    if category.lower() != "current game":
        raise ValueError(f"The stat category must be current game and not {category.lower()}")
    
    image = await find_stat_box(loop, image)
    stats = await loop.run_in_executor(None, lambda: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    columns  = find_columns(image)
    rows     = find_rows(image)
    sections = draw_sections(stats, columns, rows)

    if retreive:
        await send_to_google(sections, subcategory, columns + [sections.shape[1]], rows + [sections.shape[0]])
    else:
        cv2.imshow(path, sections)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    
async def send_to_google(sections, subcategory: str, columns: list, rows: list):
    client  = vision_v1.ImageAnnotatorAsyncClient()
    
    image   = vision_v1.types.Image(content=cv2.imencode('.png', sections)[1].tobytes())
    feature = vision_v1.types.Feature(type_=vision_v1.Feature.Type.TEXT_DETECTION)
    request = vision_v1.AnnotateImageRequest(image=image, features=[feature])
    
    response = await client.batch_annotate_images(requests=[request])

    bounds = []
    for page in response.responses[0].full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    bounds.append(([s.text for s in word.symbols], word.bounding_box))

    sections = {x: [] for x in rows}

    for symbols, bound in bounds:
        y = max([v.y for v in bound.vertices])

        for pos in rows:
            if y < pos:
                sections[pos].append((symbols, bound))
                break

    sections = [sorted(section, key=lambda item: item[1].vertices[0].x) for section in sections.values()]
    groups   = {row: {col: [] for col in columns} for row in rows}

    for i, section in enumerate(sections):
        for symbols, bound in section:
            x = min([v.x for v in bound.vertices])
            for col in columns:
                if x < col:
                    groups[rows[i]][col].append(''.join(symbols))
                    break

    if subcategory == "passer":
        print("{:<30} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format('Player', 'QBR', 'Comp.', 'Att.', 'TDs', 'Ints', 'Sacks', 'Yards', 'Long', 'Conflicts'))
        
        for cols in groups.values():
            name, qbr, comp_str, tds, ints, sacks, yards, long, *conflicts = (''.join(x) for x in cols.values())

            name      = re.sub(r'(\d+)?@', '', name)
            comp, att = re.search(r'\d+%\((\d+)\/(\d+)\)', comp_str).groups()

            print(f"{name:<30} {qbr:^10} {comp:^10} {att:^10} {tds:^10} {ints:^10} {sacks:^10} {yards:^10} {long:^10} {conflicts or ''}")

    elif subcategory == "receiver":
        print("{:<30} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format('Player', 'Catches', 'Targets', 'TDs', 'Ints All.', 'YAC', 'Yards', 'Yards', 'Long', 'Conflicts'))
        for cols in groups.values():
            name, catches, targets, tds, ints_all, yac, yards, long, *conflicts = (''.join(x) for x in cols.values() if x)

            name = re.sub(r'(\d+)?@', '', name)

            print(f"{name:<30} {catches:^10} {targets:^10} {tds:^10} {ints_all:^10} {yac:^10} {yards:^10} {long:^10} {conflicts or ''}")

    else:
        raise NotImplementedError("Only passer & receiver stats are completed")
    
async def main(retreive: bool, path: Optional[str]=None):
    if path:
        return await get_stats(path=path, retreive=retreive)

    for path in os.listdir('stats'):
        await get_stats(path=f"stats/{path}", retreive=retreive)

asyncio.run(main(retreive=True, path='stats/qb1.png'))