import numpy as np
import os
import ntpath
from . import util, html_helper


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html_helper.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses the Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        (Step 2: connect to a visdom server)
        Step 3: create an HTML object for saveing HTML filters
        """
        self.opt = opt  # cache the option
        self.use_html = True
        self.win_size = 256
        self.name = opt.name
        self.saved = False
        if self.use_html:
            self.web_dir = os.path.join(opt.exp_path, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            for curr_dir in [self.web_dir, self.img_dir]:
                os.makedirs(curr_dir, exist_ok=True)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, validation=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        prefix = 'val_epoch' if validation else 'epoch'
        if self.use_html:  # save images to an HTML file if they haven't been saved.
            # save images to the disk
            for label, image in visuals.items():
                if isinstance(image, dict):
                    for label_, image_ in image.items():
                        image_numpy = util.tensor2im(image_)
                        img_path = os.path.join(self.img_dir, '%s%.3d_%s.png' % (prefix, epoch, label_))
                        util.save_image(image_numpy, img_path)
                else:
                    image_numpy = util.tensor2im(image)
                    img_path = os.path.join(self.img_dir, '%s%.3d_%s.png' % (prefix, epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html_helper.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, -1, -1):
                if validation:
                    webpage.add_header('validation epoch [%d]' % n)
                else:
                    webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, dict):
                        img_paths = ['%s%.3d_%s.png' % (prefix, n, curr_label) for curr_label in image_numpy.keys()]
                        ims.append(img_paths)
                        txts.append([curr_label for curr_label in image_numpy.keys()])
                        links.append(img_paths)
                    else:
                        # image_numpy = util.tensor2im(image)
                        img_path = '%s%.3d_%s.png' % (prefix, n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            filename = 'val.html' if validation else "train.html"
            webpage.save(filename)
