from numpy import array, dot, vectorize
import numpy as np


def YCbCr4202YCbCr444(y, cb, cr, bitdepth=np.uint8):
	"""Up-sample the input 4:2:0 YCbCr image to 4:4:4 YCbCr

	:param ndarray y: Y image plane
	:param ndarray cb: Cb image plane
	:param ndarray cr: Cr image plane
	:return: 3D 4:4:4 YCbCr image array
	:rtype: ndarray
	"""
	height, width = y.shape
	
	ycbcr444 = np.zeros((height,width,3), dtype=bitdepth)
	
	ycbcr444[:,:,0] = y

	# Cb
	ycbcr444[0::2, 0::2, 1] = cb  # top left
	ycbcr444[1::2, 0::2, 1] = cb  # bottom left
	ycbcr444[0::2, 1::2, 1] = cb  # top right
	ycbcr444[1::2, 1::2, 1] = cb  # bottom right

	# Cr
	ycbcr444[0::2, 0::2, 2] = cr  # top left
	ycbcr444[1::2, 0::2, 2] = cr  # bottom left
	ycbcr444[0::2, 1::2, 2] = cr  # top right
	ycbcr444[1::2, 1::2, 2] = cr  # bottom right
	
	return ycbcr444


def YCbCr420_to_channels(ycbcr420):
	"""Split the input YCbCr 4:2:0 image (represented as 4:4:4 3D array) into its Y, Cb, and Cr planes.

	:param ndarray ycbcr420: 4:2:0 image represented as 4:4:4 3D array
	:return: Tuple (Y, Cb, Cr)
	:rtype: (ndarray, ndarray, ndarray)
	"""
	height, width, channels = ycbcr420.shape
	assert channels == 3

	y = ycbcr420[:, :, 0]
	cb = ycbcr420[::2, ::2, 1]
	cr = ycbcr420[::2, ::2, 2]

	return (y, cb, cr)


def YCbCr4442YCbCr420(ycbcr444):
	"""Convert 3D 4:4:4 YCbCr image array to 4:2:0 image returned as 4:4:4 3D image array

	:param ndarray: 3D 4:4:4 YCbCr image array
	:return: 3D 4:2:0 YCbCr image as 3D 4:4:4 image array
	:rtype: ndarray
	"""
	height, width, channels = ycbcr444.shape
	assert channels == 3
	ycbcr420 = np.zeros((height,width,channels), dtype=np.uint8)
	
	ycbcr444 = ycbcr444.astype(float)
		
	cb420 = np.round((ycbcr444[0:-1:2,0:-1:2,1] + ycbcr444[1::2,0:-1:2,1] + ycbcr444[0:-1:2,1::2,1] + ycbcr444[1::2,1::2,1]) / 4)
	# 0:-1:2 everything except last item
	# (top left + bottom left + top right + bottom right) / 4
	cr420 = np.round((ycbcr444[0:-1:2,0:-1:2,2] + ycbcr444[1::2,0:-1:2,2] + ycbcr444[0:-1:2,1::2,2] + ycbcr444[1::2,1::2,2]) / 4)

	# convert 4:2:0 image to 4:4:4 image
	ycbcr420 = YCbCr4202YCbCr444(ycbcr444[:,:,0], cb420.astype(np.uint8), cr420.astype(np.uint8))

	return ycbcr420


def rgb2ycbcr(rgb, flavor=709):
	"""Convert 3D RGB image array to 3D YCbCr image array

	:param ndarray rgb: 3D RGB array
	:param int flavor: flavor of conversion (601 for BT.601, 709 for BT.709)
	:return: 3D YCbCr array
	:rtype: ndarray
	"""
	height, width, channels = rgb.shape
	assert channels == 3
	ycbcr = np.zeros((height,width,channels), dtype=np.uint8)

	R, G, B = np.dsplit(rgb.astype(np.int32), 3)
	# Split array into multiple sub-arrays along the 3rd axis (depth)

	if flavor == 601:
		Y =  ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16
		# >> 8  = // by 2**8 (bits shifted to the right)
		Cb = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128
		Cr = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128
	elif flavor == 709:
		Y =  (( 47 * R + 157 * G +  16 * B + 128) >> 8) + 16
		Cb = ((-26 * R -  87 * G + 112 * B + 128) >> 8) + 128
		Cr = ((112 * R - 102 * G -  10 * B + 128) >> 8) + 128
	else:
		print('Unknown conversion flavor.')
		return
	
	## correct overflow to headroom
	Y[Y>235] = 235
	Cb[Cb>240] = 240
	Cr[Cr>240] = 240
	
	ycbcr = np.dstack((Y, Cb, Cr))
	
	ycbcr[ycbcr<16] = 16
	
	return ycbcr.astype(np.uint8)


def ycbcr2rgb(ycbcr, flavor=709):
	"""Convert 3D YCbCr 4:4:4 image array to 3D RGB image array.
	Different conversion compared to ycbcr2rgb_citrix(...)!

	:param ndarray: 3D RGB array
	:param int flavor: flavor of conversion (601 for BT.601, 709 for BT.709)
	:return: 3D YCbCr array
	:rtype: ndarray
	"""
	height, width, channels = ycbcr.shape
	assert channels == 3
	rgb = np.zeros((height,width,channels), dtype=np.int32)

	Y, Cb, Cr = np.dsplit(ycbcr.astype(np.int32),3)
	C = Y - 16
	D = Cb - 128
	E = Cr - 128
	
	#R = clip2((298*C + 409*E + 128) >> 8)
	#G = clip2((298*C - 100*D - 208*E + 128) >> 8)
	#B = clip2((298*C + 516*D + 128) >> 8)
	if flavor == 601:
		R = ((298*C         + 409*E + 128) >> 8)
		G = ((298*C - 100*D - 208*E + 128) >> 8)
		B = ((298*C + 516*D         + 128) >> 8)
	elif flavor == 709:
		R = ((298 * C           + 459 * E + 128) >> 8)
		G = ((298 * C -  55 * D - 136 * E + 128) >> 8)
		B = ((298 * C + 541 * D           + 128) >> 8)
	else:
		print('Unknown conversion flavor.')
		return
	
	rgb = np.dstack((R, G, B))
	
	rgb[rgb>255] = 255
	rgb[rgb<0] = 0
	
	return rgb.astype(np.uint8)
