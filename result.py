from fpdf import FPDF

def convert_readme_to_pdf(readme_file, pdf_file):
    # Create instance of FPDF class
    pdf = FPDF()
    
    # Add a page
    pdf.add_page()
    
    # Set font for the text
    pdf.set_font("Arial", size = 12)
    
    # Open README file and read its content
    with open(readme_file, "r") as file:
        text = file.read()
    
    # Add text to PDF
    pdf.multi_cell(0, 10, text)
    
    # Save the PDF
    pdf.output(pdf_file)

# Example usage
readme_file = "/data/wfq/code/readme.md"
pdf_file = "./output.pdf"
convert_readme_to_pdf(readme_file, pdf_file)
