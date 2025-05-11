from ocr_tamil.ocr import OCR

ocr = OCR()

ocr = OCR(detect=True,lang=["tamil"])
image_path = r"test_images\tamil_handwritten.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize",text_list)


