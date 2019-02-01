from PyPDF2 import PdfFileMerger

path = '/home/mkowalska/Marta/kCSD-python/figures/kCSD_properties/'
name = 'large_srcs_all_ele'

merger = PdfFileMerger()

for i in range(100):
    pdf = path + name + '/' + str(i) + '.pdf'
    merger.append(open(pdf, 'rb'))

with open(path + name + '/' + name + '_err.pdf', 'wb') as fout:
    merger.write(fout)
