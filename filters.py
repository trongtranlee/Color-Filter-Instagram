import cv2
import numpy as np
import matplotlib.pyplot as plt

def brightness_contrast(img, alpha = 1.0, beta = 0):
	img_contrast = img * (alpha)
	img_bright = img_contrast + (beta)
	img_bright = np.clip(img_bright,0,255)
	img_bright = img_bright.astype(np.uint8)
	return img_bright

def hue_saturation(img_rgb, alpha = 1, beta = 1):
	img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
	hue = img_hsv[:,:,0]
	saturation = img_hsv[:,:,1]
	hue = np.clip(hue * alpha ,0,179)
	saturation = np.clip(saturation * beta,0,255)
	img_hsv[:,:,0] = hue
	img_hsv[:,:,1] = saturation
	img_transformed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
	return img_transformed

def grayscale(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img_gray

def vignette(img,r,g,b,a):
	color = img.copy()
	color[:,:,0] = b
	color[:,:,1] = g
	color[:,:,2] = r
	res = cv2.addWeighted(img,1-a,color,a,0)
	return res

def replace_color(img,hl=0,sl=0,vl=0,hu=0,su=0,vu=0,nred=0,ngreen=0,nblue=0):
	rows, cols = img.shape[:2]
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# define range of color in HSV
	lower = np.array([hl,sl,hu])
	upper = np.array([hu,su,vu])
	# Threshold the HSV image to get only blue colors
	color = cv2.inRange(hsv, lower, upper)
	# Replace color
	img[color>0]=(nblue,ngreen,nred)
	return img

def increase_channel(img,channel,increment):
	img_channel = img[:,:,channel]
	img_channel = img_channel + increment
	img_channel = np.clip(img_channel,0,255)
	img[:,:,channel] = img_channel
	return img

def Clarendon(image):
	processed_image = image.copy()  # Tạo một bản sao của ảnh để xử lý
	
	# Thay đổi giá trị kênh màu
	processed_image[:, :, 0] = np.clip(processed_image[:, :, 0] * 1.2, 0, 255)  # Kênh Blue
	processed_image[:, :, 1] = np.clip(processed_image[:, :, 1] * 1.1, 0, 255)  # Kênh Green
	processed_image[:, :, 2] = np.clip(processed_image[:, :, 2] * 0.9, 0, 255)  # Kênh Red
	
	# Tăng độ tương phản và giảm độ sáng
	alpha = 1.3  # Độ tương phản
	beta = -30  # Độ sáng
	processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
	
	return processed_image

def Gingham(image):
	processed_image = image.copy()  # Tạo một bản sao của ảnh để xử lý
	
	# Ví dụ: Thay đổi giá trị kênh màu của ảnh
	processed_image[:, :, 0] = processed_image[:, :, 0] * 0.8  # Kênh Blue
	processed_image[:, :, 1] = processed_image[:, :, 1] * 1.2  # Kênh Green
	processed_image[:, :, 2] = processed_image[:, :, 2] * 0.9  # Kênh Red
	
	# Ví dụ: Tăng độ tương phản và giảm độ sáng
	alpha = 1.3  # Độ tương phản
	beta = -30  # Độ sáng
	processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
	
	return processed_image

def Reyes(image):
	processed_image = image.copy()
	# Áp dụng bộ lọc Reyes vào ảnh tại đây
	# Ví dụ: thay đổi độ sáng và tăng độ tương phản
	alpha = 1.2  # Độ tương phản
	beta = 20  # Độ sáng
	processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
	return processed_image

def Amaro(image):
	processed_image = image.copy()
	# Áp dụng bộ lọc Amaro vào ảnh tại đây
	# Thay đổi màu sắc hoặc tăng cường độ tương phản, ví dụ:
	processed_image[:, :, 0] = np.clip(processed_image[:, :, 0] * 1.1, 0, 255)  # Kênh Blue
	processed_image[:, :, 1] = np.clip(processed_image[:, :, 1] * 1.2, 0, 255)  # Kênh Green
	processed_image[:, :, 2] = np.clip(processed_image[:, :, 2] * 1.3, 0, 255)  # Kênh Red
	return processed_image

def Inkwell(image):
	processed_image = image.copy()
	# Chuyển ảnh thành đen trắng
	gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
	# Tăng cường độ tương phản
	alpha = 1.5
	beta = 0
	processed_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
	return processed_image

def Nashville(image):
	processed_image = image.copy()
	# Thay đổi màu sắc
	processed_image[:, :, 2] -= 30  # Giảm một lượng màu đỏ
	processed_image[:, :, 1] += 15  # Tăng một lượng màu xanh lá cây
	# Tăng cường ánh sáng
	alpha = 1.2
	beta = 20
	processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
	return processed_image

def Toaster(image):
	processed_image = image.copy()
	# Thay đổi màu sắc
	processed_image[:, :, 2] -= 20  # Giảm một lượng màu đỏ
	processed_image[:, :, 1] += 10  # Tăng một lượng màu xanh lá cây
	
	# Tăng cường ánh sáng
	alpha = 1.5
	beta = 30
	processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
	return processed_image

def _1977(image):
	processed_image = image.copy()
	# Thay đổi màu sắc
	processed_image[:, :, 0] -= 20  # Giảm một lượng màu xanh
	processed_image[:, :, 2] += 20  # Tăng một lượng màu đỏ
	
	# Làm mịn ảnh
	processed_image = cv2.GaussianBlur(processed_image, (7, 7), 0)
	
	# Áp dụng một số hiệu ứng loá
	rows, cols, _ = processed_image.shape
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)  # Tạo ma trận xoay
	processed_image = cv2.warpAffine(processed_image, M, (cols, rows))  # Xoay ảnh một chút
	
	return processed_image

def Kelvin(image):
	processed_image = image.copy()
	
	# Tăng độ ấm và tông màu vàng
	processed_image[:, :, 1] += 20  # Tăng một lượng màu xanh lá cây
	processed_image[:, :, 2] += 20  # Tăng một lượng màu đỏ
	
	# Tạo hiệu ứng vignette
	rows, cols, _ = processed_image.shape
	kernel_x = cv2.getGaussianKernel(cols, 200)
	kernel_y = cv2.getGaussianKernel(rows, 200)
	kernel = kernel_y * kernel_x.T
	mask = 255 * kernel / np.linalg.norm(kernel)
	mask = np.expand_dims(mask, axis=-1)  # Mở rộng ma trận mask thành 3 kênh màu
	mask = np.tile(mask, [1, 1, 3])  # Sao chép mask để phù hợp với số kênh màu của ảnh
	
	processed_image = np.clip(processed_image + mask, 0, 255)
	processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)  # Chuẩn hóa lại dữ liệu ảnh
	processed_image = cv2.convertScaleAbs(processed_image)  # Chuyển đổi về dạng số nguyên 8-bit
	
	return processed_image



def Original(img, hue = 1, saturation = 1, contrast = 1, brightness = 0):
	img = hue_saturation(img, hue, saturation)
	img = brightness_contrast(img, contrast, brightness)
	return img

filters = { #Thư viện lưu trữ filters
	'None': Original,
	'Clarendon': Clarendon,
	'Gingham': Gingham,
	'Reyes': Reyes,
	'Amaro': Amaro,
	'Inkwell': Inkwell,
	'Nashville': Nashville,
	'Toaster': Toaster,
	'_1977': _1977,
	'Kelvin': Kelvin,
	# Thêm các filter khác ở đây đây
}



