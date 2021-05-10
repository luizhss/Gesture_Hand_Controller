import pyautogui
import cv2
from utils import *


class HandMovements:
	"""
		Hand Movements Class.

		This class execute movements using the hand pose predicted from HandPoses.
		The gesture controller uses a specific pyauyogui function to each class.

		Keyword Arguments:
			screen_proportion {float}: the proportion of gesture controller interaction area in 'mouse'
				class, ie, proportion of area to mapper mouse movement.
				(default: {0.75})
			len_moving_average {float}: the moving average is used to
				calculate the average of midpoint of five-fingers landmarks
				in an array with the history of this midpoint. To this calculus, the
				len_moving_average will be the length of this midpoint history array.
				When this value has the tradeoff: increase this number improves the mouse
				sensitivity, but delays the mouse iteration (midpoint update)
				(default: {10})
	"""

	def __init__(self, screen_proportion=0.75, len_moving_average=10):
		self.screen_proportion = screen_proportion

		self.screen_width, self.screen_height = pyautogui.size()
		self.camera_width, self.camera_height = None, None
		self.x_start_screen, self.y_start_screen = None, None
		self.x_end_screen, self.y_end_screen = None, None

		self.angle_now = None

		self.x_moving_average = np.array([])
		self.y_moving_average = np.array([])
		self.len_moving_average = len_moving_average

	def draw_mouse_rectangle(self, frame):
		"""
		This method draw a rectangle of the effective interaction area to mapping mouse movement
		"""

		if self.camera_width is None:
			image_height, image_width, _ = frame.shape

			self.update_width_height(image_height, image_width)

		cv2.rectangle(frame, (self.x_start_screen, self.y_start_screen),
					(self.x_end_screen, self.y_end_screen), (255, 255, 255), 2)

	def update_width_height(self, image_height, image_width):
		"""
		This method update the width and height of the camera and the points
		that limit the effective interaction area to mapping mouse movement
		"""

		self.camera_width, self.camera_height = image_width, image_height
		self.x_start_screen = int((1 - self.screen_proportion) * self.camera_width / 2)
		self.y_start_screen = int((1 - self.screen_proportion) * self.camera_height / 2)
		self.x_end_screen = int((1 + self.screen_proportion) * self.camera_width / 2)
		self.y_end_screen = int((1 + self.screen_proportion) * self.camera_height / 2)

	def execute_movement(self, pose, lm, delay, frame):
		"""
			Execute Movement Method

			This method execute movements (gesture controller) using pose class.

			Arguments:
				pose {string}: predicted hand pose
				lm {string}: hands landmarks detected by HandDetect
				delay {Delay}: class responsible to provoke delays on the execution frames
				frame {cv2 Image, np.ndarray}: webcam frame
		"""

		if pose == 'left_click':
			pyautogui.click(button='left')

			self.angle_now = None
			delay.reset_counter()

		elif pose == 'right_click':
			pyautogui.click(button='right')

			self.angle_now = None
			delay.reset_counter()

		elif pose == 'scroll_up':
			pyautogui.scroll(3)

			self.angle_now = None
			delay.reset_counter()
			delay.set_in_action(True)

		elif pose == 'scroll_down':
			pyautogui.scroll(-3)

			self.angle_now = None
			delay.reset_counter()
			delay.set_in_action(True)

		elif pose == 'zoom':

			if self.angle_now is None:
				self.angle_now = get_angle(lm)
			else:
				angle_old = self.angle_now
				self.angle_now = get_angle(lm)

				pyautogui.keyDown('ctrl')

				if self.angle_now > angle_old:
					angle = min(self.angle_now - angle_old, 90)
					zoom = 3
				else:
					angle = max(angle_old - self.angle_now, 0)
					zoom = -3
				angle = int(angle)//10

				while angle > 0:
					pyautogui.scroll(zoom)
					angle -= 1

				pyautogui.keyUp('ctrl')

			delay.reset_counter(20)
			delay.set_in_action(True)

		elif pose == 'mouse':
			self.angle_now = None
			delay.set_in_action(True)

			x_mouse, y_mouse = get_average_points(lm)
			x_mouse *= self.camera_width
			y_mouse *= self.camera_height

			if self.mouse_on_screen(x_mouse, y_mouse):
				x_mapper = (x_mouse - self.x_start_screen) / (
						self.camera_width * self.screen_proportion) * self.screen_width
				y_mapper = (y_mouse - self.y_start_screen) / (
						self.camera_height * self.screen_proportion) * self.screen_height

				x_average, y_average, idle = self.update_moving_average_xy(x_mapper, y_mapper)

				if not idle:
					pyautogui.moveTo(x_average, y_average)

					x_average_cam = int(
						self.x_start_screen +
						x_average * self.camera_width * self.screen_proportion / self.screen_width)
					y_average_cam = int(
						self.y_start_screen +
						y_average * self.camera_height * self.screen_proportion / self.screen_height)

					cv2.circle(frame, (x_average_cam, y_average_cam), 3, (255, 0, 0), 4)
				else:
					cv2.putText(frame, f"Position locked", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 100), 2)
					delay.reset_counter(40)

		else:
			self.angle_now = None
		return None

	def update_moving_average_xy(self, x_mapper, y_mapper):
		"""
		This method update the width and height of the camera and the points
		that limit the effective interaction area to mapping mouse movement
		"""

		self.x_moving_average = np.append(self.x_moving_average, x_mapper)
		self.y_moving_average = np.append(self.y_moving_average, y_mapper)

		if self.x_moving_average.size > self.len_moving_average:
			self.x_moving_average = np.delete(self.x_moving_average, 0)
			self.y_moving_average = np.delete(self.y_moving_average, 0)

		x_average = self.x_moving_average.mean()
		y_average = self.y_moving_average.mean()

		x_std = self.x_moving_average.std()
		y_std = self.y_moving_average.std()

		dist = np.sqrt(x_std ** 2 + y_std ** 2)

		return x_average, y_average, (dist < 7. and self.x_moving_average.size == self.len_moving_average)

	def mouse_on_screen(self, x_detected, y_detected):
		"""
		This method return if the position (x, y) of detected of midpoint of five-fingers
		landmarks is inside in effective interaction area to mapping mouse movement
		"""
		return \
			self.x_start_screen <= x_detected <= self.x_end_screen \
			and self.y_start_screen <= y_detected <= self.y_end_screen
