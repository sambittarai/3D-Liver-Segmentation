def calc_pad(img_shape, vol_size):
    new_shape = [0,0,0]

    if img_shape[0] % vol_size[0] != 0:
        new_shape[0] = (img_shape[0]//vol_size[0] + 1)*vol_size[0]
    else:
        new_shape[0] = img_shape[0]

    if img_shape[1] % vol_size[1] != 0:
        new_shape[1] = (img_shape[1]//vol_size[1] + 1)*vol_size[1]
    else:
        new_shape[1] = img_shape[1]

    if img_shape[2] % vol_size[2] != 0:
        new_shape[2] = (img_shape[2]//vol_size[2] + 1)*vol_size[2]
    else:
        new_shape[2] = img_shape[2]

    diff_x, diff_y, diff_z = (new_shape[0] - img_shape[0]), \
                             (new_shape[1] - img_shape[1]), \
                             (new_shape[2] - img_shape[2])

    before_x, before_y, before_z = diff_x//2, diff_y//2, diff_z//2
    # before_x, before_y, before_z = 0, 0, 0
    after_x, after_y, after_z = diff_x - before_x, \
                                diff_y - before_y, \
                                diff_z - before_z
    pad_width  = ((before_x, after_x),
                  (before_y, after_y),
                  (before_z, after_z))
    return pad_width


def two_split(x):
    return x//2, x - x//2

def calc_pad_for_fit(img_shape, vol_size):
    """
    Given two 3D array shapes `img_shape` and `vol_size`, this function calculates the sizes of arrays to be padded to the `img_shape` so that an array of shape `vol_size` can be perfectly packed into the array of shape `img_shape`
    Args : 
    img_shape : shape of the bigger array
    vol_size : shape of smaller array to be fit 
    """
    return calc_pad(img_shape, vol_size)

def calc_pad_for_pred_loss(patch_size, out_size):
    loss_pad = [two_split(patch_size[i] - out_size[i]) for i in range(3)]
    return loss_pad


def crop_pad_width(image, pad_width):
    # print(pad_width)
    image_shape = image.shape
    before_x, after_x = pad_width[0]
    before_y, after_y = pad_width[1]
    before_z, after_z = pad_width[2]

    if image.ndim == 3:
        return image[before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z]
    elif image.ndim == 4:
        return image[ before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z, :]