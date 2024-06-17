import asyncio
import cv2
import discord
import numpy
import pytesseract
import os
import re
from google.cloud import vision_v1
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import pyperclip

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"credentials.json"

categories    = ["my stats", "current game", "server", "global"]
subcategories = ["passer", "runner", "receiver", "corner", "defender", "kicker", "other"]

async def find_stat_category(loop: asyncio.AbstractEventLoop, image):
    category = subcategory = None

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
            
            text = await loop.run_in_executor(pool, lambda: pytesseract.image_to_string(roi).strip())

            if not text:
                continue

            if text.lower() in categories:
                category = text.lower()
            elif text.lower() in subcategories:
                subcategory = text.lower()

    if not category:
        raise ValueError("No category")
    
    if not subcategory:
        raise ValueError("No subcategory")

    return (category, subcategory)

async def find_stat_box(loop: asyncio.AbstractEventLoop, image) -> numpy.ndarray:
    with ThreadPoolExecutor() as pool:
        image     = await loop.run_in_executor(pool, lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        _, binary = await loop.run_in_executor(pool, lambda: cv2.threshold(image, 127, 255, cv2.THRESH_BINARY))

        contours, _ = await loop.run_in_executor(pool, lambda: cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        
        stat_box = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(stat_box)
        
        return image[y:y+h, x:x+w]
    
def find_boundaries(stats: numpy.ndarray) -> List[int]:
    row_means   = numpy.mean(stats, axis=1)
    differences = numpy.abs(numpy.diff(row_means))
    
    if not len(row_means) or not len(differences):
        raise ValueError("There are no sections")
    
    boundaries = [i + 1 for i, diff in enumerate(differences) if diff > 19]
    
    if not len(boundaries) > 1:
        raise ValueError("There are not atleast two sections")
    
    return boundaries

def draw_sections(stats: numpy.ndarray, boundaries: list) -> numpy.ndarray:
    for y in boundaries:
        cv2.line(stats, (0, y), (stats.shape[1], y), (0, 0, 255), 2)
        
    return stats

async def main(retreive: bool):
    loop = asyncio.get_event_loop()
    file = discord.File("stats/wr1.png")
    
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

    boundaries = find_boundaries(image)
    sections   = draw_sections(stats, boundaries)

    if retreive:
        await send_to_google(sections, subcategory)
    else:
        cv2.imshow("Stats", sections)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    
async def send_to_google(sections, subcategory: str):
    client  = vision_v1.ImageAnnotatorAsyncClient()
    
    image   = vision_v1.types.Image(content=cv2.imencode('.png', sections)[1].tobytes())
    feature = vision_v1.types.Feature(type_=vision_v1.Feature.Type.TEXT_DETECTION)
    request = vision_v1.AnnotateImageRequest(image=image, features=[feature])
    
    response = await client.batch_annotate_images(requests=[request])
    text     = response.responses[0].full_text_annotation.text

    pyperclip.copy(text)

    stats    = [x for x in re.split(r"\n?[0-9]{0,2}?.? ?@", text) if x]

    if subcategory.lower() == "passer":
        print(f"{'NAME':<20} {'CMP/ATT':<10} {'YARDS':<10} {'TDS':<10} {'INTS':<10}  {'CONFLICTS'}")
        for statline in stats:
            statline = [x.strip() for x in statline.split("\n")]

            if len(statline) < 8:
                print(f"\nSkipping line due to insufficient data: {statline}\n")
                continue
            
            display_name, _, comp_str, tds, ints, _, yards, _, *rest = statline
            
            comp_att_match = re.search("[0-9]+/[0-9]+", comp_str)
            if comp_att_match:
                comp, att = comp_att_match.group().split('/')
            else:
                comp, att = '0', '0'

            print(f"{display_name:<20} {f'{comp}/{att}':<10} {yards:<10} {tds:<10} {ints:<10}  {rest if rest else ''}")
    else:
        raise NotImplementedError("Only passer stats are completed")
    
asyncio.run(main(retreive=False))