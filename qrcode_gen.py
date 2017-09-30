from PIL import Image, ImageDraw, ImageFont
from hubarcode.datamatrix import DataMatrixEncoder
import os
from random import choice

def gen_qr_code(encoded_str, temp_file_path):

    # Generate QR code
    encoder = DataMatrixEncoder(encoded_str)
    encoder.save("datamatrix_temp.png", cellsize=9)
    datamatrix = Image.open("datamatrix_temp.png")
    datamatrix = datamatrix.resize((180, 180), Image.ANTIALIAS)

    # Save and clean
    parent_dir = os.getcwd()
    if not os.path.exists(temp_file_path):
        os.makedirs(str(temp_file_path))
    os.chdir(temp_file_path)
    datamatrix.save('qrcode.png', 'png')
    os.chdir(parent_dir)
    os.remove("datamatrix_temp.png")
    print("  -> Created copy qrcode for test #%s"
          % (encoded_str))

if __name__ == '__main__':
    gen_qr_code("This is a test", "test")