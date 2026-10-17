import base64
import io
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TextItem, TableItem
from docling.chunking import HybridChunker
from model import load_llm


## Loading Models
llm = load_llm()

docling_converter = None

def get_docling_converter():
    global docling_converter
    if docling_converter is None:
            ##Configuring docling
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.do_ocr = True
        
            # Ensure table structure model is enabled
            pipeline_options.do_table_structure = True
        
            docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
    return docling_converter


def parse_pdf(PDF_PATH, EXTRACTED_VISUAL_PATH):
    

    output_dir = Path(EXTRACTED_VISUAL_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Docling layout and table analysis...")
    conv_res = get_docling_converter().convert(PDF_PATH)
    doc = conv_res.document
    return doc



#CHUNK TABLE -> Langchain Doc
def chunk_table(
    table_item,
    table_index: int,
    max_rows_per_chunk: int = 15
) -> list[Document]:
    """
    Chunks a Docling table into row batches while preserving column headers
    in every markdown segment.
    """
    df: pd.DataFrame = table_item.export_to_dataframe()
    
    # Guard against empty/malformed tables
    if df.empty:
        return []

    total_rows = len(df)
    page_no = table_item.prov[0].page_no if table_item.prov else None
    columns = [str(c) for c in df.columns]

    documents = []

    # Iterate in row slices (single-iteration if total_rows <= max_rows_per_chunk)
    for start_idx in range(0, total_rows, max_rows_per_chunk):
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        sub_df = df.iloc[start_idx:end_idx]

        # Markdown conversion retains the header row automatically
        table_markdown = sub_df.to_markdown(index=False)

        # Append row slice indicator only if the table was actually split
        slice_tag = f" [Rows {start_idx + 1}–{end_idx} of {total_rows}]" if total_rows > max_rows_per_chunk else ""
        
        content = (
            f"### Table {table_index + 1} (Page {page_no}){slice_tag}\n\n"
            f"{table_markdown}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "chunk_id": f"table_{table_index + 1}_rows_{start_idx + 1}_{end_idx}",
                    "type": "table",
                    "page_number": page_no,
                    "rows_range": f"{start_idx + 1}-{end_idx}",
                    "total_rows": total_rows,
                    "columns": columns,
                }
            )
        )

    return documents


## Chart/Image Transcription
def get_surrounding_context(target_item, document, window_size=2) -> str:
    body_items = list(document.iterate_items())
    target_idx = -1
    for idx, (item, _) in enumerate(body_items):
        if item == target_item:
            target_idx = idx
            break

    if target_idx == -1:
        return "No adjacent context available."

    start_idx = max(0, target_idx - window_size)
    end_idx = min(len(body_items), target_idx + window_size + 1)

    context_snippets = []
    for i in range(start_idx, end_idx):
        if i == target_idx:
            continue
        curr_item, _ = body_items[i]
        if isinstance(curr_item, TextItem) and curr_item.text.strip():
            context_snippets.append(curr_item.text.strip())

    return "\n".join(context_snippets)

## RESIZE and TRANSCRIBE
def process_and_resize_image(pil_img, max_side: int = 1568) -> str:
    """
    Converts image to RGB, downsizes only if the longest side exceeds max_side
    (aspect ratio preserved), and returns the base64 data string.
    """
    img = pil_img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _content_to_text(content) -> str:
    """
    Claude via LangChain returns either a str or a list of content blocks
    (thinking / text / ...). Keep only the text blocks.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p).strip()


def transcribe_chart_with_langchain(pil_image, surrounding_context: str) -> str:

    base64_image = process_and_resize_image(pil_image)
    prompt_text = f"""
You are an expert technical and financial document analyst.
Analyze this cropped figure/chart from an enterprise document.

SURROUNDING DOCUMENT TEXT:
---
{surrounding_context}
---

INSTRUCTIONS:
1. Ground your interpretation in the surrounding text.
2. Transcribe axes, legends, units, and data points into a clean Markdown table.
3. Provide a concise 2-sentence summary of the main insight.
"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    text = _content_to_text(llm.invoke([message]).content)
    return text or "(No transcription produced for this figure.)"

##IMAGE FILTERING

def get_picture_metrics(picture, doc):
    """Extract image dimensions and page-relative bounding-box area."""
    image = picture.get_image(doc)

    if image is None:
        return None

    width, height = image.size

    metrics = {
        "image": image,
        "width": width,
        "height": height,
        "pixel_area": width * height,
        "page_area_ratio": None,
        "aspect_ratio": width / max(height, 1),
    }

    # Try to calculate the picture's area relative to its PDF page.
    try:
        if picture.prov:
            prov = picture.prov[0]
            page = doc.pages[prov.page_no]
            bbox = prov.bbox

            page_width = page.size.width
            page_height = page.size.height

            bbox_width = abs(bbox.r - bbox.l)
            bbox_height = abs(bbox.t - bbox.b)

            page_area = page_width * page_height

            if page_area > 0:
                metrics["page_area_ratio"] = (
                    bbox_width * bbox_height / page_area
                )
    except (AttributeError, KeyError, IndexError, TypeError):
        # Pixel-based filtering can still be used.
        pass

    return metrics


def is_relevant_picture(
    picture,
    doc,
    min_width=100,
    min_height=80,
    min_pixel_area=12_000,
    min_page_area_ratio=0.002,
):
    """
    Heuristic filter: reject very small images and tiny page regions.

    Tune the thresholds against your own PDFs.
    """
    metrics = get_picture_metrics(picture, doc)

    if metrics is None:
        return False, "no_image", None

    width = metrics["width"]
    height = metrics["height"]
    page_ratio = metrics["page_area_ratio"]

    if width < min_width or height < min_height:
        return False, "small_dimensions", metrics

    if metrics["pixel_area"] < min_pixel_area:
        return False, "small_pixel_area", metrics

    if (
        page_ratio is not None
        and page_ratio < min_page_area_ratio
    ):
        return False, "small_page_area", metrics

    return True, "passed_heuristics", metrics





def create_langchain_documents(PDF_PATH, EXTRACTED_VISUAL_PATH):
    
    doc = parse_pdf(PDF_PATH, EXTRACTED_VISUAL_PATH)
    langchain_documents: list[Document] = []

    ##TABLE CHUNK to Doc
    for idx, table in enumerate(doc.tables):
        # Chunks tables slices with preserved headers
        table_chunks = chunk_table(
            table_item=table,
            table_index=idx,
            max_rows_per_chunk=10,  # No. of rows per chunk
        )
        langchain_documents.extend(table_chunks)

    
    ##Image to Doc
    filter_stats = Counter()
    visual_documents = []

    print(f"Found {len(doc.pictures)} picture candidates.")

    for idx, picture in enumerate(doc.pictures):
        keep, reason, metrics = is_relevant_picture(picture, doc)
        filter_stats[reason] += 1

        if not keep:
            print(f"Skipping picture {idx}: {reason}")
            continue

        pil_image = metrics["image"]
        page_no = picture.prov[0].page_no if picture.prov else None

        img_path = EXTRACTED_VISUAL_PATH / f"figure_{idx + 1}.png"
        pil_image.save(img_path)

        # Only call the expensive vision model for candidates
        # that have passed the initial filtering.
        context_text = get_surrounding_context(
            picture, doc, window_size=2
        )

        transcription = transcribe_chart_with_langchain(
            pil_image, context_text
        )

        visual_documents.append(
            Document(
                page_content=(
                    f"### Visual Data / Figure\n"
                    f"{transcription}\n\n"
                    f"**Surrounding Context:**\n{context_text}"
                ),
                metadata={
                    "chunk_id": f"visual_{idx + 1}",
                    "type": "chart_or_figure",
                    "page_number": page_no,
                    "image_path": str(img_path),
                    "image_width": metrics["width"],
                    "image_height": metrics["height"],
                    "page_area_ratio": metrics["page_area_ratio"],
                },
            )
        )

    langchain_documents.extend(visual_documents)

    ## TEXT to doc
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=512,
        merge_peers=True,
    )
    doc_chunks = list(chunker.chunk(doc))
    for idx, chunk in enumerate(doc_chunks):
        # Skip chunk if it only contains table items (since we explicitly handled tables above)
        has_only_tables = chunk.meta.doc_items and all(
            isinstance(item, TableItem) for item in chunk.meta.doc_items
        )
        if has_only_tables:
            continue

        page_no = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else None
        langchain_documents.append(
            Document(
                page_content=chunker.contextualize(chunk=chunk),
                metadata={
                    "chunk_id": f"text_chunk_{idx}",
                    "type": "text",
                    "page_number": page_no,
                    "image_path": None,
                },
            )
        )
    return langchain_documents



        


