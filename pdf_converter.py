import pymupdf4llm

class PdfConverter:
    @staticmethod
    def pdf_to_markdown(pdf_file_path):
        # Convert PDF to markdown pages with full metadata
        pages = pymupdf4llm.to_markdown(pdf_file_path, page_chunks=True)

        print(f"Converted {pdf_file_path} to Markdown with {len(pages)} pages.")

        return pages
