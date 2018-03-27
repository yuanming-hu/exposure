import os
import urllib.request
import urllib
import zipfile


def download(url, fn=None, path=None):
  if path is not None:
    os.makedirs(path, exist_ok=True)
  else:
    path = '.'

  if fn == None:
    fn = url.split('/')[-1]
  print('Downloading file [{}]'.format(fn))
  print('  URL        : {}'.format(url))
  dest_fn = os.path.join(path, fn)
  print('  Destination: {}'.format(dest_fn))
  u = urllib.request.urlopen(url)
  meta = u.info()
  file_size = int(meta.get_all("Content-Length")[0])
  
  with open(dest_fn, 'wb') as f:
    print('Downloading: [{}] ({:.2f} MB)'.format(fn, file_size / 1024 ** 2))
    
    downloaded = 0
    block_size = 65536
    
    while True:
      buffer = u.read(block_size)
      if not buffer:
        break
      f.write(buffer)
      downloaded += len(buffer)
      progress = '  {:.2f}MB  [{:3.2f}%]'.format(downloaded / 1024 ** 2, downloaded * 100 / file_size)
      print(progress, end='\r')
  print()
  
if __name__== '__main__':
  print('This file downloads ready-to-use package of Adobe-MIT FiveK dataset.')
  print('Total download size = ~2.5GB')
  
  fn_template = 'https://github.com/yuanming-hu/exposure_models/releases/download/v0.0.1/{}'

  
  print('# File 1/3')
  download(fn_template.format('FiveK_C.zip'), path='data/artists/')
  print('  Extracting...')
  with zipfile.ZipFile('data/artists/FiveK_C.zip', 'r') as zip_ref:
    zip_ref.extractall('data/artists/FiveK_C')
  
  print('# File 2/3')
  download(fn_template.format('image_raw.npy'), path='data/fivek_dataset/')
  
  print('# File 3/3')
  download(fn_template.format('meta_raw.pkl'), path='data/fivek_dataset/')
  
  print('Congratulation: the MIT-Adobe FiveK Dataset is ready. You can train your own model with \'python3 train.py example\'')

  

