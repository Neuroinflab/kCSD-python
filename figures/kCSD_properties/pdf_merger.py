from PyPDF2 import PdfFileMerger
import glob

prefix = '/home/mkowalska/Marta/kCSD-python/figures/kCSD_properties/large_srcs_minus_5'
pdfs = [f for f in glob.glob(prefix + "/*.pdf")]

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(open(pdf, 'rb'))

with open(prefix + '/result.pdf', 'wb') as fout:
    merger.write(fout)
