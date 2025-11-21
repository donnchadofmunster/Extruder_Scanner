import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import glob

class Frame:
    def __init__(
            self, 
            image: np.ndarray
            ): 
        self.image = image
        self.cropped_image = self.crop_to_laser_area(image)
        self.x_position_array = self.extract_laser_line_centre_array()
    
    @staticmethod
    def gaussian(
            x, 
            amplitude, 
            mean, 
            standard_deviation
            ) -> np.ndarray:
        '''
        Gaussian function for curve fitting.

        Parameters:
        - x: The horizontal pixel positions along the row (0 to width-1)
        - amplitude: Peak height of the Gaussian.
        - mean: Center position of the Gaussian.
        - standard_deviation: Standard deviation (width) of the Gaussian.

        Returns:
        - Gaussian function evaluated at x.
        '''
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation ** 2))

    @staticmethod
    def create_red_white_mask(
            frame
            ) -> np.ndarray:

        '''
        Converts the input frame to HSV color space and creates a binary mask that highlights pixels in the red and white color ranges.

        Parameters:
        - frame: The input image frame in BGR color space.

        Returns:
        - red_white_mask: A binary mask where pixels within the red and white color ranges are white (255), 
          and all other pixels are black (0). 
        '''
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 40, 30])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 40, 30])
        upper_red2 = np.array([180, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

        red_white_mask = cv2.bitwise_or(mask_red, mask_white)
        red_white_mask = cv2.morphologyEx(red_white_mask, cv2.MORPH_OPEN, np.ones((1,1), np.uint8))
        red_white_mask = cv2.morphologyEx(red_white_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(bright_mask, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)  # X gradient

        # Split channels
        # b, g, r = cv2.split(frame)

        # # Threshold red channel
        # _, red_mask = cv2.threshold(r, 180, 255, cv2.THRESH_BINARY)

        # # Optional blur
        # blurred = cv2.GaussianBlur(red_mask, (5, 5), 0)

        # # Sobel X
        # sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)

        return sobel_x

        return sobel_x

        # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, bright_mask = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(red_white_mask, (5, 5), 0)
        return blurred

        return red_white_mask

    @staticmethod
    def crop_to_laser_area(
            image: np.ndarray,
            ) -> np.ndarray:
        
        '''
        Crops the frame to focus on the area where the laser line is expected to be.
        '''
        height, width = image.shape[:2]
        crop_width = int(width * 0.4)
        cropped_image = image[:, :crop_width]
        return cropped_image

    def get_intensity_edge(
            self,
            y_values, 
            invert=False
            ) -> int:
        '''
        Fits a Gaussian to the intensity profile and returns the x-coordinate of the peak (the mean).
        Initial guess is based on the maximum Y value, and a gaussian is fitted to find a more accurate peak position.
        The gaussian is fitted an amount of times equal to maxfev.

        Parameters:
        - y_values: 1D array of intensity values along the x-axis.
        - invert: If True, inverts the y_values to find the opposite edge This allows the left and right edge to be found in one function.

        Returns:
        - x-coordinate of the peak intensity (mean of the fitted Gaussian).
        '''
        if invert:
            y_values = -y_values # Invert to find the opposite edge     
        x_values = np.arange(len(y_values))

        initial_guess = [max(y_values), np.argmax(y_values), 1] # Amplitude, Mean, Standard Deviation
        try:
            params, _ = curve_fit(self.gaussian, 
                                x_values, 
                                y_values, 
                                p0=initial_guess,
                                maxfev=10000)
            _, fitted_mean, _ = params
            return int(np.round(fitted_mean))
        except RuntimeError:
            return np.argmax(y_values)
    
    def extract_laser_line_centre_array(
            self
            ) -> np.ndarray:
        '''
        Extracts the centre of the laser line using gaussian fitting on each row of the red-white mask.
        Gaussian fitting is used to determine the left and right edges of the laser line, and the centre is computed as the average of these edges.

        Parameters:
        - frame: The input image frame in BGR color space.

        Returns:
        - centerline: An array containing the x-coordinates of the laser line centre for each row.

        '''
        red_white_mask = Frame.create_red_white_mask(self.image)
        mask_height, mask_width = red_white_mask.shape
        centerline_array = np.full(mask_height, -1, dtype=int)
        
        for y in range(mask_height): # For every Y value in the frame
            x_values = red_white_mask[y, :] # Look at the pixels intensities on the X axis
            if np.count_nonzero(x_values) < 2:
                continue  # Treat as invalid if less than 2 non-zero pixels
            left_x_position = self.get_intensity_edge(x_values)
            right_x_position = self.get_intensity_edge(x_values, invert=True)
            centerline_array[y] = (left_x_position + right_x_position) // 2
        return centerline_array

        # mask = Frame.create_red_white_mask(self.image)
        # centerline = np.full(mask.shape[0], -1, dtype=float)

        # for y in range(mask.shape[0]):
        #     intensity = mask[y, :].astype(float)
        #     if intensity.sum() == 0:  # skip rows with no signal
        #         continue

        #     x_values = np.arange(len(intensity))
        #     centerline[y] = np.sum(x_values * intensity) / np.sum(intensity)  # weighted centroid

        # return centerline

    def plot_centerline(
            self
            ) -> None:
        '''
        Plots the extracted centerline on the image for visualization.

        Parameters:
        - frame: The input image frame in BGR color space.

        Returns:
        - None
        '''
        overlay = self.image.copy()
        height, width, _ = overlay.shape
        for y in range(height):
            x_pos = int(np.round(self.x_position_array[y]))  # round and cast to int
            if x_pos >= 0 and x_pos < width:  # also ensure x_pos is inside image
                overlay[y, x_pos] = (0, 0, 0)  # Black centerline
        cv2.imshow('Centerline Overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class BeamAnalyser:
    def __init__(
            self, 
            undefected_frame: Frame, # Undefected Line on Beam
            sample_frame: Frame, # Input to compare to Undefected Line
            reference_frame: Frame # Reference Line on Beam
            ):
        self.undefected_frame = undefected_frame
        self.sample_frame = sample_frame
        self.reference_frame = reference_frame
        self.pixels_per_mm, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self.callibrate_camera()

    def callibrate_camera(
            self,
            chessboard_size = (5, 5),
            frame_size = (640, 480),
            square_size_mm = 5.0
            ) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        '''
        Callibrates the camera using chessboard images to find pixels per mm and camera matrix and distortion coefficients.

        Parameters:
        - chessboard_images: List of file paths to chessboard images for calibration.
        - chessboard_size: Tuple indicating the number of inner corners per chessboard row and column (rows, columns).
        - square_size_mm: Size of each chessboard square in millimeters.

        Returns:
        - pixels_per_mm: Average number of pixels per millimeter.
        - camera_matrix: Camera matrix obtained from calibration.
        - dist_coeffs: Distortion coefficients obtained from calibration.
        - rvecs: Rotation vectors for each calibration image.
        - tvecs: Translation vectors for each calibration image.
        '''
        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare 3D points for each corner (0,0,0) ... (8,5,0)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm  # scale to mm

        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        images = glob.glob(f"Extruder_Scanner/callibration/beam_2/*.jpg")

        if not images:
            raise FileNotFoundError(f"No calibration images found")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if not objpoints:
            raise RuntimeError("No valid chessboard patterns detected. Calibration failed.")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, frame_size, None, None
        )
        all_dists = []
        for corners in imgpoints:
            for i in range(len(corners) - 1):
                dx = corners[i+1][0][0] - corners[i][0][0]
                dy = corners[i+1][0][1] - corners[i][0][1]
                all_dists.append(np.hypot(dx, dy))
        avg_pixel_spacing = np.mean(all_dists)
        pixels_per_mm = (avg_pixel_spacing / 2) / square_size_mm

        return pixels_per_mm, camera_matrix, dist_coeffs, rvecs, tvecs

    def extract_pixel_differences(
            self
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Extracts the pixel differences between the reference frame and the warped frame by computing the center of the laser line in both frames.

        Parameters:
        - reference_frame: The reference Frame object.
        - sample_frame: The Frame object looking to be compared to the reference frame.

        Returns:
        - reference_x_position_array: An array containing the x-coordinates of the laser line centre for each row in the reference frame.
        - sample_x_position_array: An array containing the x-coordinates of the laser line centre for each row in the sample frame.
        - pixel_difference_array: An array containing the per-row pixel differences between the warped and reference frames.
        '''
        reference_x_position_array = self.reference_frame.x_position_array
        sample_x_position_array = self.sample_frame.x_position_array

        pixel_difference_array = np.zeros_like(reference_x_position_array, dtype=float)
        valid_rows = (reference_x_position_array >= 0) & (sample_x_position_array >= 0)
        pixel_difference_array[valid_rows] = sample_x_position_array[valid_rows] - reference_x_position_array[valid_rows]
        pixel_difference_array[~valid_rows] = np.nan

        return reference_x_position_array, sample_x_position_array, pixel_difference_array
    
    def overlay_centerlines(
            self
            ) -> np.ndarray:
        '''
        Overlays the centerlines extracted from both the reference and sample frames onto the reference frame for visualization.

        Parameters:
        - reference_frame: The reference Frame object.
        - sample_frame: The Frame object looking to be compared to the reference frame.

        Returns:
        - overlay: The reference frame with centerlines from both frames overlaid.
        '''
        undefected_x_position_array, sample_x_position_array, pixel_difference_array = self.extract_pixel_differences()
        overlay = self.undefected_frame.image.copy()
        height, width, _ = overlay.shape
        
        for y in range(height):
            if undefected_x_position_array[y] >= 0:
                overlay[y, undefected_x_position_array[y]] = (0, 255, 0)  # Green
            if sample_x_position_array[y] >= 0:
                overlay[y, sample_x_position_array[y]] = (255, 0, 0)  # Blue
        
        cv2.imshow('Overlayed Centerlines', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return overlay

    def plot_pixel_differences(
            self
            ) -> None:
        '''
        Plots the pixel differences between the reference frame and the sample frame as a heatmap alongside the image.

        Parameters:
        - reference_frame: The reference Frame object.
        - sample_frame: The Frame object looking to be compared to the reference frame.

        Returns:
        - A heatmap showing the sample frame and the pixel difference
        '''
        _, _, pixel_difference_array = self.extract_pixel_differences()
        overlay_image = self.sample_frame.image.copy()
        centerline = self.undefected_frame.x_position_array
        height, width, _ = overlay_image.shape
        
        for y in range(height):
            if centerline[y] >= 0:
                overlay_image[y, centerline[y]] = (0, 255, 0)  # Green
        
        figure, axes = plt.subplots(1, 2, figsize=(6, 4))
        
        axes[0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Warped image and Reference Centerline", fontsize=8)
        axes[0].axis('off')
        
        # Show heatmap of pixel differences
        im = axes[1].imshow(pixel_difference_array.reshape(-1, 1), 
                            cmap='bwr', aspect='auto')
        axes[1].set_title("Defect Heatmap", fontsize=8)
        axes[1].set_xlabel("")
        axes[1].set_xticks([])
        axes[1].set_ylabel("Y position in picture")
        figure.colorbar(im, ax=axes[1], label="Difference in Pixels")
        
        plt.tight_layout()
        plt.show()

    def find_beam_height_from_reference(
            self,
            reference_height_mm,
            x_camera_offset_mm: float = 37.3,
            ) -> None:
        '''
        Finds the shame of the top of the beam based on the reference frame using laser triangulation

        Parameters:
        - reference_frame: The reference Frame object.
        - sample_frame: The Frame object looking to be compared to the reference frame.

        Returns:
        - None
        '''

        sample_x_position_array = self.sample_frame.x_position_array
        reference_x_position_array = self.reference_frame.x_position_array
        pixel_displacement = np.abs(sample_x_position_array - reference_x_position_array)
        s = pixel_displacement / self.pixels_per_mm
        heights = reference_height_mm * s / (x_camera_offset_mm + s)
        print(sample_x_position_array[:30])
        print(reference_x_position_array[:30])
        print(pixel_displacement[:30])
        print(self.pixels_per_mm)
        return heights

    def plot_heights(
            self,
            ):
        '''
        Finds the shame of the top of the beam based on the reference frame using laser triangulation

        Parameters:
        - self: The self object, used to find the beam height to plot.

        Returns:
        - None
        '''
        heights = self.find_beam_height_from_reference(
            reference_height_mm=65,
            x_camera_offset_mm=37.3
        )

        heights = np.array(heights)

        plt.figure(figsize=(4, 4))
        plt.plot(heights, color='green')
        plt.xlabel('Row (Y position)')
        plt.ylabel('Height (mm)')
        plt.title('Height of object in mm')
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.show()

# ---------- Defect Detection ---------- #    

def analyse_beam():
    '''
    To use, the defected frame is the frame for the object that is intended to be measured. The reference frame is the laser against the base plate.
    These need to be input when creating the Frame Object, and the code should detect the height of the object pictured in the defected frame.
    '''
    # 12/11/2025 Data - 61mm from top
    undefected_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1811_OBJ.jpg"))
    defected_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1811_OBJ.jpg"))
    reference_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1811_REF.jpg"))

    # defected_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1211Laser006.jpg"))
    # reference_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1211Reference005.jpg"))

    # 12/11/2025 Data - 55mm from top
    # defected_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1211Laser001.jpg"))
    # reference_frame = Frame(cv2.imread("Extruder_Scanner/beam_images/1211Laser002.jpg"))

    # Beam Analyser Object
    analyser = BeamAnalyser(undefected_frame, defected_frame, reference_frame)

    # analyser.overlay_centerlines()

    # Plot Heights
    defected_frame.plot_centerline()
    analyser.plot_heights()

analyse_beam()