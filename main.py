import os
import numpy as np
import tempfile
import cv2
import unittest
from compress import *
from extract import *
import numpy



def psnr(data1, data2, max_pixel=255):
    mse = np.mean((data1 - data2) ** 2)
    if mse:
        return 20 * math.log10(max_pixel / mse ** 0.5)
    return math.inf



def compress_and_extract(spec):
    with open(spec['fn'], 'rb') as raw_file:
        compressed = comp(
            raw_file,
            size=spec['size'],
            grey_level=spec['grey_level'],
            quality=spec['quality'],
            subsampling_mode=spec['subsampling_mode']
        )
    header = compressed['header']
    
    
    with tempfile.TemporaryFile() as compressed_file:
        compressed['data'].tofile(compressed_file)
        compressed_file.seek(0)
        extracted = extract(
            compressed_file,
            header={
                'size': header['size'],
                'grey_level': header['grey_level'],
                'quality': header['quality'],
                'subsampling_mode': header['subsampling_mode'],
                'remaining_bits_length': header['remaining_bits_length'],
                'data_slice_lengths': header['data_slice_lengths']
            }
        )
    return extracted




def read_img(fn):
    with open(fn, 'rb') as f:
        ret = np.fromfile(f, dtype=np.uint8)
    return ret



def main():

    print('please input filename(eq:Baboon.raw) or input ALL to show all imgs')
    
    filename = input()
    
   
    
    if not os.path.isfile( filename ) or  not os.path.isfile( filename ) :
        print("file no exist")
    else:
    
        
        
        original = read_img(filename).reshape((512, 512) if ('RGB' not in filename) else (512, 512, 3))
        
        print('please input QF (ex:90)')
        q = int(input())
        
        extracted = compress_and_extract({
            'fn': filename,
            'size': (512, 512),
            'grey_level': ('RGB' not in filename),
            'quality': q,
            'subsampling_mode': 1}).reshape((512, 512) if ('RGB' not in filename) else (512, 512, 3))
            
        print('PSNR:  ',psnr(original, extracted))
         
        print(extracted.shape)
        
        
        if 'RGB' not in filename :
            filename = filename.replace('.raw', ".png")
            filename = 'QF_' + str(q) + '_GRAY_'+filename 
            
            print('filename: ' +filename)
            cv2.imwrite(os.path.join('./', filename), extracted)
        
        
        else :
            filename = filename.replace('.raw', ".png")
            filename = 'QF_' + str(q) + '_RGB_'+filename 
            img = cv2.cvtColor(numpy.asarray(extracted),cv2.COLOR_RGB2BGR) 
            print('filename: ' +filename)
            cv2.imwrite(os.path.join('./', filename), img)
            
            
           
            
    
    
    

    


if __name__ == '__main__':
    main()
