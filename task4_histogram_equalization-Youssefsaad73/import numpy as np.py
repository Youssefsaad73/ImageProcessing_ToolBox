import numpy as np
import cv2

######## special configuration of matplotlib ######
import matplotlib as mpl
#reload interactive backend Qt5Agg
mpl.use('Qt5Agg')

#read the absolute filepath to stylesheet





import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
########  ends here ###############################


# initialize list for indices of the image
list_counters = []
list_space_shape = [0,0,0,0]
list_pulse_index = []
list_iter_index = []
list_exp_index = []
list_img_file = []
list_counters = [list_space_shape, list_pulse_index, list_iter_index, list_exp_index, list_img_file]


# find the indices 


list_space_shape = list_counters[0]
list_pulse_index = list_counters[1]
list_iter_index = list_counters[2]
list_exp_index = list_counters[3]
list_img_file = list_counters[4]


# number of images
n_images = len(list_img_file)


def iterateImages():
    """
    this method has the following attributes
    
    0. counter
    1. space
    2. img
    3. img_color
    4. img_restored
    
    
    
    Upon being called, this method reads the image file
    corrsponding to the function attribute `counter`.
    Then this image is "blitted" to the first image of the widget.
    Blitting the image instead of updating the complete figure results
    in very high performance gains in terms of latency, as it involves
    updating only the portion of pixels which change when an event is
    triggered.
    
    Reference: Blitting tutorial
    https://matplotlib.org/3.3.1/tutorials/advanced/blitting.html#sphx-glr-tutorials-advanced-blitting-py
    
    The other attributes space, img_color and img_restored are set
    in the `on_select()` callback method for the rectangle selection event
    generated by the myWidget object of class type RectangleSelector
    """
    
    # get the counter of the function
    i = iterateImages.counter

    if i < n_images:
        
        img_file = list_img_file[i]
            
        img = cv2.imread(img_file, 0)
        
        # set the img attribute of iterateImages
        iterateImages.img = img
        
        #update the image
        blit_img = myWidget.ax.imshow(img, cmap='gray', animated=True)
        
        # draw the animated artist, this uses a cached renderer
        myWidget.ax.figure.axes[1].draw_artist(blit_img)
        
        # show the result to the screen, this pushes the updated RGBA buffer from the
        # renderer to the GUI framework so you can see it
        myWidget.ax.figure.canvas.blit(myWidget.ax.figure.bbox)    
    
        ##### **IMPORTANT** update the background ################
        myWidget.background = myWidget.canvas.copy_from_bbox(myWidget.ax.bbox)
        
    else:
        #deactivate the window
        myWidget.set_active(False)
        
        ###### IMPORTANT save the space in the end ######################
        space = iterateImages.space
        
        saveSpace(space)
        
    
# event handling functions
def onselect(eclick, erelease):
    """
        This function is called when keypress and release event is created.
        
        eclick and erelease are matplotlib events at press and release.
        
        # print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        # print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        # print('used button  : ', eclick.button)
        
        
        However we will use `myWidget.corners` attribute
        to retrieve the four corners of the rectangle
    """
    
    # reset the background back in the canvas state, screen unchanged
    myWidget.ax.figure.canvas.restore_region(bg)
    
    coordinates = myWidget.corners
    
    # update the attribute of iterateImage coordinate
    iterateImages.coordinates = coordinates
    
    # get the image attribute
    img = iterateImages.img
    
    # save the image_crop
    x0 = int(eclick.xdata)
    y0 = int(eclick.ydata)
    x1 = int(erelease.xdata)
    y1 = int(erelease.ydata)
    img_crop = img[y0:y1, x0:x1]
    


    # read the coordinates
    # coordinates = np.load(coordinates_file)
    if deep_debug: print(f"coordinates: {coordinates}")  
    
    # current counter
    i = iterateImages.counter
    
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    ###### ** important ** pass coordinates as optional keyword argument to findEdge()
    pulse, img_color, windowROI = findEdge(pulse_index, iter_index, exp_index, img, coordinates=coordinates)


    ############## now restore the crop ####################
    
    img_background = np.ones(img.shape, dtype = np.uint8)*255
    img_crop_restored = restoreROI(img_crop, img_background, windowROI)

    # write the img_crop_restored to cropped axis 
    blit_crop_restored = myWidget.ax.figure.axes[1].imshow(img_crop_restored, cmap='gray', vmin=0, vmax=255, animated=True)
    
    # myWidget.ax.figure.axes[1].imshow(img_crop, cmap='gray')

    # draw the animated artist, this uses a cached renderer
    myWidget.ax.figure.axes[1].draw_artist(blit_crop_restored)
    # show the result to the screen, this pushes the updated RGBA buffer from the
    # renderer to the GUI framework so you can see it
    myWidget.ax.figure.canvas.blit(myWidget.ax.figure.bbox)    
    
    ###########now restore the fit ###############    

    img_restored = restoreColorROI(img_color, img, windowROI)
    
    # update the subplot restored image
    blit_restored = myWidget.ax.figure.axes[2].imshow(img_restored, animated=True)
    
    # draw the animated artist, this uses a cached renderer
    myWidget.ax.figure.axes[2].draw_artist(blit_restored)
    
    # show the result to the screen, this pushes the updated RGBA buffer from the
    # renderer to the GUI framework so you can see it
    myWidget.ax.figure.canvas.blit(myWidget.ax.figure.bbox)    
    
    # myWidget.update()
    
    # update the space attribute
    
    space  = iterateImages.space
    space[exp_index, iter_index, pulse_index] = pulse
    iterateImages.space = space
    
    # update the img_color attribute
    
    iterateImages.img_color = img_color    
    
    # update the img_restored attribute

    iterateImages.img_restored = img_restored
    
    return


def saveIteration():
    """
    0. This function is callback to keypress event 'n' or 'N' handled byy
        to toggle_selector() method
    1. Upon calling, save the params, img_fit, and img_restored.
    
    2. Increment the counter attribute of iterateImages().
    
    We are using function attributes to record state across callbacks.
    
    First we "get" the `counter`, `img_color`, `img_restored` and `space` attributes
    
    which were first "set" by the `onselect()` callback method.
    
    After saving respectively the fit, restored image and parameters
    
    we "set" the `counter` attribute by incremented to point to the next image. 
    """
    
    # get the counter
    i = iterateImages.counter
    print(f"i is {i}")
    

    
    # read the img_color attribute
    img_color = iterateImages.img_color
    
    # get the restored_image attribute
    img_restored = iterateImages.img_restored
    
    #read the space attribute
    space = iterateImages.space
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # get the pulse params from space
    pulse = space[exp_index, iter_index, pulse_index] 
    
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img_color)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img_restored) 
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse)
    
    #save the space to file to allow restart options
    np.save(space_filepath, space)
    
    # iterate the counter after the save
    iterateImages.counter += 1

    return

def discardIteration():
    """
    0. This function is callback to keypress event 'd' or 'D' handled byy
        to toggle_selector() method.
    1. It does the job of discarding the image.
        Upon calling, set the params to dummy i.e. `np.nan`,
        img_fit to img, img_color to img
    2. Increment the counter attribute of iterateImages().
    """
    
    # get the counter
    i = iterateImages.counter
    print(f"i is {i}")
    
    # read the image attribute
    img = iterateImages.img
    
    #read  the space attribute
    space = iterateImages.space
    
    # indices of the image
    pulse_index = list_pulse_index[i]
    iter_index = list_iter_index[i]
    exp_index = list_exp_index[i]
    
    # set the pulse params to np.nan except confidence which is set to 0
    pulse = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # update the pulse params to space
    space[exp_index, iter_index, pulse_index] = pulse
    iterateImages.space = space
    
    #save the image to file    
    saveImage(pulse_index, iter_index, exp_index, img)
    
    # save the restored image to file
    saveRestoredImage(pulse_index, iter_index, exp_index, img)
    
    # save the params
    savePulse(pulse_index, iter_index, exp_index, pulse)
    
    #save the space to file to allow restart options
    np.save(space_filepath, space)
    
    # iterate the counter after the save
    iterateImages.counter += 1

    return

def toggle_selector(event):
    """
        0. Key press event handler for the `myWidget` RectangleSelector widget.
        
        1. 'n' or 'N' event calls `saveIteration()` followed by `iterateImages()`
        2. 'd' or 'D' event calls `discardIteration()` followed by `iterateImages()`
        
    """
    
    print('Key pressed.')
    if event.key in ['N', 'n'] and myWidget.get_active():
        print('Image iterated.')
        myWidget.set_active(True)

        saveIteration()
        iterateImages()

    if event.key in ['D', 'd'] and myWidget.get_active():
        print('Image discarded.')
        myWidget.set_active(True)

        discardIteration()
        iterateImages()
        
    if event.key in ['A', 'a'] and not myWidget.get_active():
        print('RectangleSelector activated.')
        myWidget.set_active(True)

if __name__ == '__main__':
    
    

        ##### set the matplotlib figure params ##################
        mpl.rcParams['figure.figsize'] = 9.0,4.0
        mpl.rcParams['figure.subplot.bottom'] = 0.1
        mpl.rcParams['figure.subplot.left'] = 0.1
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.top'] = 0.7
    
        fig, axes = plt.subplots(1,3)
        
        
        fig.suptitle('widget ROI selection', fontsize= 24)
        fig.text(0.5,0.85, '''Press `N' to iterate image, `D' to discard image, `Q' to kill the window''',
                 ha='center', va='center')
        axes[0].set_title('raw image')
        axes[1].set_title('cropped image')
        axes[2].set_title('restored image')
    
    
        ############# initialize the widget ##################
        # use a white background image
        img_white = np.ones(image_shape, dtype=np.uint8)*255
        axes[0].imshow(img_white, cmap='gray', vmin=0, vmax=255)
    
        axes[1].imshow(img_white, cmap='gray', vmin=0, vmax=255)
        # and also set the autoscale OFF
        axes[1].set_autoscale_on(False)
    
        # also draw the restored image to set the bbox dimensions
        axes[2].imshow(img_white, cmap='gray', vmin=0, vmax=255)    
    
        #pause for a 'short' to ensure the figure is rendered
        # atleast one before storing the bbox info (in sec)
        plt.pause(0.1)
        
        # use this as background for future restoration
        # get copy of entire figure (everything inside fig.bbox) sans animated artist
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        
        # a reference to the RectangleSelector widget is needed
        # to prevent from Garbage Collection
        myWidget = RectangleSelector(axes[0], onselect, useblit=True)
    
        ######## initialize attributes of iterateImages #############

        # set the shape of space array
        space = np.zeros((list_space_shape[0], list_space_shape[1], list_space_shape[2], list_space_shape[3]), dtype=np.float)
        np.save(space_filepath, np.array(space))
        
        iterateImages.counter   = 0
        iterateImages.space     = space
        iterateImages.img       = img_white
        iterateImages.img_color = img_white
        iterateImages.img_restored = img_white
        
        iterateImages()
    
        print(f'fig.canvas.supports_blit property: {fig.canvas.supports_blit}' )
    
        #### IMPORTANT: create event loop for matplotlib fig window ####
        myWidget.connect_event('key_press_event', toggle_selector)