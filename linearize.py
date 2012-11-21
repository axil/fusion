#import numpy as np
#import matplotlib.pyplot as plt
#
#x = np.arange(0, 5, 0.1)
#plt.plot(x, np.sin(x))
#plt.show()

import Image
import numpy as np
import matplotlib.pyplot as plt
from color.color import rgb2lab
import matplotlib.pyplot as plt

def myopen(filename):
    return rgb2lab(np.asarray(Image.open(filename)).astype(np.float32)/255.)

def a():
    print 'a'
    import Image
    print 'a'
    import numpy as np
    print 'a'
    print 'imported'
    a = Image.open('window_exp_1_1.jpg')
    print 'opened'
    b = np.asarray(a)
    c = b.astype(np.float32)/255.

    from color.color import rgb2lab
    
    print '0..1'
    d = rgb2lab(c)
    print d[0,0]
    return d

def b():
    from skimage import io, color
    print 'imported'
    rgb = io.imread('window_exp_1_1.jpg')
    print 'opened'
    lab = color.rgb2lab(rgb)
    print lab[0,0]

a = myopen('D:\Documents\window_series\window_exp_1_4.jpg')
b = myopen('D:\Documents\window_series\window_exp_1_1.jpg')
q = np.zeros(101)
h = np.zeros(101)
for y in xrange(a.shape[0]):
    for x in xrange(a.shape[1]):
        aa = a[y,x,0]
        bb = b[y,x,0]
        if aa not in (0, 100) and bb not in (0, 100):
            q[aa] = bb/aa
            h[aa] += 1

plt.subplot(211)
plt.plot(q)
plt.subplot(212)
plt.semilogy(h)
