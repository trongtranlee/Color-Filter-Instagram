import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import filters
import threading

def filter_func(label):
	reset()
	global current_filter
	current_filter = label  # Lưu filter được chọn

def update_camera():
	global color, img
	while True:
		ret, img = cap.read()
		if current_filter:
			color = filters.filters[current_filter](img)
		else:
			color = img.copy()
		if radio.value_selected == 'color':
			if len(color.shape) == 3:  # Kiểm tra kích thước của mảng
				l.set_data(color[:,:,::-1])
			else:
				l.set_data(color)
		else:
			gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
			l.set_data(gray)
			l.set_cmap('gray')
		fig.canvas.draw_idle()

def update(val):
	global color, img
	color = filters.brightness_contrast(img, scont.val / 10, sbright.val - 50)
	color = filters.hue_saturation(color, shue.val / 10, ssat.val / 10)
	if radio.value_selected == 'color':
		l.set_data(color[:,:,::-1])
	else:
		gray = filters.grayscale(color[:, :, ::-1])
		l.set_data(gray)
		l.set_cmap('gray')
	fig.canvas.draw_idle()


def colorfunc(label):
	if(label == 'color'):
		l.set_data(color[:,:,::-1])
	else:
		l.set_data(gray)
		l.set_cmap('gray')
	fig.canvas.draw_idle()


def reset():
	scont.reset()
	sbright.reset()
	shue.reset()
	ssat.reset()

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)  # Mở kết nối đến camera (0 là camera mặc định)
	
	fig, ax = plt.subplots()
	plt.subplots_adjust(left=0.02, bottom=0.30, right=0.50)
	
	ret, img = cap.read()  # Đọc frame đầu tiên từ camera
	color = img.copy()
	gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
	l = plt.imshow(color[:,:,::-1])
	
	#Tạo ra các thanh trượt
	axcolor = 'lightgoldenrodyellow'  # Màu nền thanh trượt
	axcont = plt.axes([0.55, 0.85, 0.38, 0.03], facecolor=axcolor) #Vị trí và kích thước của thanh trượt tương ứng
	axbright = plt.axes([0.55, 0.80, 0.38, 0.03], facecolor=axcolor) #Vị trí và kích thước của thanh trượt tương ứng
	axhue = plt.axes([0.55, 0.75, 0.38, 0.03], facecolor=axcolor) #Vị trí và kích thước của thanh trượt tương ứng
	axsat = plt.axes([0.55, 0.70, 0.38, 0.03], facecolor=axcolor) #Vị trí và kích thước của thanh trượt tương ứng
	
	scont = Slider(axcont, 'Contrast', 0, 20, valinit=10)  #Tạo giá trị thanh trượt Contrast
	sbright = Slider(axbright, 'Brightness', 0, 100, valinit=50) #Tạo giá trị thanh trượt Brightness
	shue = Slider(axhue, 'Hue', 0, 20, valinit=10)  #Tạo giá trị thanh trượt Hue
	ssat = Slider(axsat, 'Saturation', 0, 20, valinit=10) #Tạo giá trị thanh trượt Saturation
	
	filtax = plt.axes([0.50, 0.02, 0.45, 0.65], facecolor=axcolor)
	filt = RadioButtons(filtax, ('None','Clarendon', 'Gingham', 'Reyes', 'Amaro', 'Inkwell', 'Nashville', 'Toaster', '_1977', 'Kelvin'), active=0)
	
	# RadioButtons cho chuyển đổi giữa hiển thị màu sắc và ảnh xám
	rax = plt.axes([0.1, 0.02, 0.15, 0.15], facecolor=axcolor)
	radio = RadioButtons(rax, ('color', 'grayscale'), active=0)
	current_filter = None
	
	# Tạo luồng riêng cho việc đọc dữ liệu từ camera và cập nhật hình ảnh
	camera_thread = threading.Thread(target=update_camera)
	camera_thread.daemon = True
	camera_thread.start()
	
	scont.on_changed(update)
	sbright.on_changed(update)
	shue.on_changed(update)
	ssat.on_changed(update)
	
	
	radio.on_clicked(colorfunc)
	filt.on_clicked(filter_func)
	plt.show()