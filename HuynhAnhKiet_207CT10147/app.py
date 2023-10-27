from flask import Flask, render_template, url_for, redirect, request, flash, get_flashed_messages
import urllib.request
import os
from werkzeug.utils import secure_filename

from PIL import Image
import math
import scipy
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['bmp', 'bmp', 'dib',
'jpeg','jpg','jpe', 'jp2',
'png','webp','pbm','pgm',
'ppm', 'pxm','pnm','tiff',
'tif','exr','hdr','pic'])

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def Inverse(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    #convert image 1 into an ndarray
    im_1 = np.asarray(img)
    #inversion operation
    im_2 = 255 - im_1
    #convert image 2 from ndarray to image
    new_img = Image.fromarray(im_2)
    fig = plt.figure(figsize=(10, 7))

    plt.ioff() 
    plt.figure()
    plt.title('Hinh inverse transformation')
    plt.imshow(new_img)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Inverse.jpg')

def Gamma(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im_1 = np.asarray(img)
    gamma = 0.5
    b1 = im_1.astype(float)
    b2 = np.max(b1)
    b3 = b1/b2
    b2 = np.log(b3) * gamma
    c = np.exp(b2) * 255.0
    c1 = c.astype(float)
    d = Image.fromarray(c1)

    fig = plt.figure(figsize=(10, 7))
    plt. ioff() 
    plt.title('Hinh Gamma Correction')
    plt.imshow(d)
    
    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Gamma.jpg')

def Log(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im_1 = np.asarray(img)
    gamma = 5
    b1 = im_1.astype(float)
    b2 = np.max(b1)
    c = (128.0 * np.log(1 + b1))/np.log(1 + b2)
    c1 = c.astype(float)
    d = Image.fromarray(c1)

    fig = plt.figure(figsize=(10, 7))
    plt.ioff() 
    plt.title('Hinh Log Transformation')
    plt.imshow(d)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Log.jpg')


def Histogram(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im1 = np.asarray(img)
    b1 = im1.flatten()
    hist, bins = np.histogram(im1, 256, [0,255])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    num_cdf_m = (cdf_m - cdf_m.min()) * 255
    den_cdf_m = (cdf.max() - cdf_m.min())
    cdf_m = num_cdf_m/den_cdf_m
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    im2 = cdf[b1]
    im3 = np.reshape(im2, im1.shape)
    im4 = Image.fromarray(im3)

    fig = plt.figure(figsize=(10, 7))
    plt.ioff() 
    plt.title('Hinh Histogram equalization')
    plt.imshow (im4)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Histogram.jpg')

def Contrast(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im1 = np.asarray(img)
    b = im1.max()
    a = im1.min()
    print(a,b)
    c = im1.astype(float)
    im2 = 255* (c-a)/(b-a)
    im3 = Image.fromarray(im2)

    plt.ioff() 
    plt.title('Hinh Contrast Stretching')
    plt.imshow(im3)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Contrast.jpg')

def FFT(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im1 = np.asarray(img)
    c = abs(scipy.fftpack.fft2(im1))
    d = scipy.fftpack.fftshift(c)
    d = d.astype(float)
    im3 = Image.fromarray(d)

    plt.ioff() 
    fig = plt.figure(figsize=(10, 7))
    plt.title('Hinh Fast Fourier Transform')
    plt.imshow(im3)
    
    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/FFT.jpg')


def Lowpass(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im1 = np.asarray(img)
    c = abs(scipy.fftpack.fft2(im1))
    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]
    H = np.ones((M,N))

    center1 = M/2
    center2 = N/2
    d_0 = 30.0 
    t1 = 1 
    t2 = 2 * t1
    for i in range(1,M):
        for j in range (1,N):
            r1 = (i- center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r > d_0:
                H[i,j] = 1/(1 + (r/d_0)**t1)

    H = H.astype(float)
    H = Image.fromarray(H)
    con = d * H
    e = abs(scipy.fftpack.ifft2(con))
    e = e.astype(float)
    im3 = Image.fromarray(e)
    plt.ioff() 
    plt.title('Butterworth Lowpass Filter')
    plt.imshow(im3)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Lowpass.jpg')

def Highpass(filename):
    img = Image.open('static/uploads/'+filename).convert('L')
    im1 = np.asarray(img)
    c = abs(scipy.fftpack.fft2(im1))
    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]
    H = np.ones((M,N))
    center1 = M/2
    center2 = N/2
    d_0 = 30.0 
    t1 = 1 
    t2 = 2 * t1
    for i in range(1,M):
        for j in range (1,N):
            r1 = (i- center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r > d_0:
                H[i,j] = 1/(1 + (r/d_0)**t2)

    H = H.astype(float)
    H = Image.fromarray(H)
    con = d * H
    e = abs(scipy.fftpack.ifft2(con))
    e = e.astype(float)
    im3 = Image.fromarray(e)
    fig = plt.figure(figsize=(10, 7))

    plt.ioff() 
    plt.title('Butterworth highpass Filter')
    plt.imshow (im3)

    fig = plt.gcf()
    img = fig2img(fig)
    img = img.convert('RGB')    
    img.save('static/uploads/Highpass.jpg')
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded dayo')
        Inverse(filename)
        Gamma(filename)
        Log(filename)
        Histogram(filename)
        Contrast(filename)
        Lowpass(filename)
        Highpass(filename)
        FFT(filename)
        return render_template('index.html', filename=filename)

    else:
        flash('Allowed image types are - bmp, bmp, dib, jpeg, jpg, jpe, jp2, png, webp, pbm, pgm, ppm, pxm, pnm, tiff, tif, exr, hdr, pic')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/work', methods=['POST'])
def work():
    if request.form['submit_button'] == 'Inverse':
        return redirect(url_for('static', filename='uploads/Inverse.jpg'), code=301)
    elif request.form['submit_button'] == 'Gamma Correction':
        return redirect(url_for('static', filename='uploads/Gamma.jpg'), code=301)
    elif request.form['submit_button'] == 'Log Transformation':
        return redirect(url_for('static', filename='uploads/Log.jpg'), code=301)
    elif request.form['submit_button'] == 'Histogram equalization':
        return redirect(url_for('static', filename='uploads/Histogram.jpg'), code=301)
    elif request.form['submit_button'] == 'Contrast Stretching':
        return redirect(url_for('static', filename='uploads/Contrast.jpg'), code=301)
    elif request.form['submit_button'] == 'Fast Fourier Transformation':
        return redirect(url_for('static', filename='uploads/FFT.jpg'), code=301)
    elif request.form['submit_button'] == 'Butterworth Lowpass Filter':
        return redirect(url_for('static', filename='uploads/Lowpass.jpg'), code=301)
    elif request.form['submit_button'] == 'Butterworth highpass Filter':
        return redirect(url_for('static', filename='uploads/highpass.jpg'), code=301)
    else:
        pass

if __name__ == "__main__":
    app.run(debug=True)
