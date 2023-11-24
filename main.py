from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import re
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from paddleocr import PaddleOCR
import os


app = Flask(__name__)


# OCR and figure extraction functions
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


def analyze_document_with_azure_ocr(image_file):
    endpoint = "https://centralindia.api.cognitive.microsoft.com/"
    key = "867f899e93f840c89c046e56d0bf9475"

    # Initialize the Document Analysis client
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    # Analyze the document
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-layout", image_file
    )
    result = poller.result()

    # Extracted tables
    extracted_tables = []

    # Iterate through tables in the result
    for table_idx, table in enumerate(result.tables):
        table_data = {
            "table_number": table_idx + 1,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "cells": [],
        }

        # Iterate through cells in the table
        for cell in table.cells:
            cell_info = {
                "row_index": cell.row_index,
                "column_index": cell.column_index,
                "content": cell.content,
            }
            table_data["cells"].append(cell_info)

        extracted_tables.append(table_data)

    return extracted_tables


ocr = PaddleOCR(use_gpu=False)


def extract_figures_from_image(image_path):
    # Perform OCR on the image
    result = ocr.ocr(image_path)

    extracted_text = []
    for text_result in result:
        for region in text_result:
            detected_text, confidence = region[1]
            # Remove commas from detected text
            detected_text_without_comma = detected_text.replace(",", "")
            extracted_text.append(detected_text_without_comma)

    return extracted_text


def sum_figures_in_image(image_path):
    extracted_text = extract_figures_from_image(image_path)

    # Extract numbers from the extracted text using regular expressions
    figures = []
    for text in extracted_text:
        numbers = re.findall(r"\d+\.?\d*", text)
        figures.extend(numbers)

    # Convert figures to floats and return them
    return [float(fig) for fig in figures]


@app.route("/extract", methods=["POST"])
def extract_data():
    try:
        extraction_type = request.form.get("extraction_type")

        if extraction_type is None:
            return jsonify({"error": "Extraction type not provided"}), 400

        if not extraction_type.isdigit():
            return jsonify({"error": "Extraction type should be an integer"}), 400

        extraction_type = int(extraction_type)

        if extraction_type == 1:  # text-snip
            if "image" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            image_file = request.files["image"]
            img = Image.open(image_file)
            text = extract_text_from_image(img)
            return jsonify({"text": text})

        elif extraction_type == 2:
            if "image" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            image_file = request.files["image"]
            tables = analyze_document_with_azure_ocr(image_file)
            return jsonify({"tables": tables})

        elif extraction_type == 3:
            if "image" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            file = request.files["image"]

            if file.filename == "":
                return jsonify({"error": "No selected file"})

            # Save the file temporarily
            image_path = "temp_image.png"  # Change to an appropriate temporary location
            file.save(image_path)

            figures = sum_figures_in_image(image_path)

            if figures:
                # Create a string with the figures separated by ' + '
                figures_str = " + ".join(str(float(fig)) for fig in figures)

                # Calculate the total sum
                total_sum = sum(figures)

                # Prepare response
                formula = f"{figures_str}"
                sum_result = f"{total_sum}"
                response = {"formula": formula, "result": sum_result}

                os.remove(image_path)
                return jsonify(response)
            else:
                return jsonify({"error": "No figures found in the image"})

    except Exception as e:
        app.logger.error(str(e))
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
