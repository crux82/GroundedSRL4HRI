from PIL import Image, ImageDraw, ImageFont
import logging
import os


logger = logging.getLogger(__name__)


def scale_bbox_for_image(bb_coordinates: list, img_width: int, img_height: int):
    x1, y1, x2, y2 = bb_coordinates
    
    # if x1 starts with 0, it is a relative coordinate
    if str(x1).startswith("0"):
        # Convert to absolute coordinates
        x1 = int(x1 * img_width)
        y1 = int(y1 * img_height)
        x2 = int(x2 * img_width)
        y2 = int(y2 * img_height)
    # else check if all values, after converting to int, are between 0 and 1024
    elif all(0 <= int(coord) <= 1024 for coord in [x1, y1, x2, y2]):
        pass
    else:
        raise ValueError(f"Invalid bounding box coordinates: {bb_coordinates}")
    
    return int(x1), int(y1), int(x2), int(y2)


def create_image_with_bb(image_path, frame_semantics_with_bb):
    """
    Creates an image with bounding boxes drawn on it based on the bbs present in the frame semantics and the labels.
    """
    # Check if the image already exists
    new_image_path = image_path.replace(".png", "_bb_from_interpretation.png")
    if not os.path.exists(new_image_path):
        # Load the image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        colors = ["cyan", "purple", "orange"]
        i = 0

        # Load a scalable font
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
        except IOError:
            font = ImageFont.load_default()

        # Parse the frame semantics with bounding boxes
        frames = frame_semantics_with_bb.split(" - ")
        for frame in frames:
            frame_elements = "(".join(frame.split("(")[1:]).split("), ")
            for element in frame_elements:
                frame_element_value = element.split("(")[1].split(",")[0].strip()
                frame_element_bb_tag = ",".join(element.split("(")[1].split(",")[1:]).strip().replace("))", "")
                if frame_element_bb_tag.startswith("[") and frame_element_bb_tag.endswith("]"):
                    try:
                        # remove leading 0s for each coordinate
                        frame_element_bb_tag = frame_element_bb_tag.replace(", 0", ", ").replace(", 0", ", ").replace("[0", "[").replace("[0", "[")
                        # Extract the bounding box coordinates
                        bb_coordinates = eval(frame_element_bb_tag)
                    except Exception as e:
                        logger.warning(f"Error parsing bounding box coordinates:\n\n{e}")
                        quit()
                        
                    # Scale the bounding box coordinates to the image dimensions
                    img_width, img_height = img.size
                    x1, y1, x2, y2 = scale_bbox_for_image(bb_coordinates, img_width, img_height)
                    
                    # Draw the bounding box on the image
                    draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=3)
                    
                    # Compute the size of the label text
                    text = frame_element_value
                    bbox = font.getbbox(text)
                    text_height = bbox[3] - bbox[1]

                    # Place the label above the box (with padding)
                    text_x = x1
                    text_y = max(0, y1 - text_height - 8)

                    # Draw black border around the text (by offsetting in 4 directions)
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            draw.text((text_x + dx, text_y + dy), text, fill="black", font=font)

                    # Draw the label text in the current colour
                    draw.text((text_x, text_y), text, fill=colors[i], font=font)
                    
                    # Increment the color index
                    i = (i + 1) % len(colors)

        # Save the modified image on file
        img.save(new_image_path)
        return img
    else:
        return Image.open(new_image_path)    