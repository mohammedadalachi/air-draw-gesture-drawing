from datetime import datetime
# Deque for rolling window / majority vote logic
# https://docs.python.org/3/library/collections.html#collections.deque
from collections import deque
import cv2
import numpy as np
# MediaPipe Hands documentation:
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
import mediapipe as mp


CONFIG = {
"width": 640,
"height": 480,

#padding to ensure choosing colors is safe
"top_bar_height": 60,
"top_bar_padding": 18,
#adjusting thickness
"brush_thickness": 6,
"eraser_radius": 55,
#smothing
"smooth_alpha": 0.35,
#adding stability
"vote_frames": 7,
#ensuring saving is safe
"save_hold_frames": 6,
"save_cooldown": 35,

"min_stroke_points": 14,

#Line snap is safe
"line_r2_min": 0.93,
"min_line_length": 60,

#Circle snap is safe
"circularity_min": 0.72,
"min_circle_radius": 14,
"min_contour_area": 250,
}

#colors to choose in the top bar in bgr format
PALETTE = [
    ("White",  (255, 255, 255)),
    ("Blue",   (255,  80,   0)),
    ("Green",  ( 60, 255, 120)),
    ("Red",    ( 60,  60, 255)),
    ("Yellow", (  0, 255, 255)),
    ("Purple", (255,  80, 255)),
]


#converting landmark to pixels
def lm_to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))
# Compute distance between two points
def distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
#smothing fingertip
def ema(prev, cur, alpha):
    if prev is None:
        return cur
    x = int(prev[0] + alpha * (cur[0] - prev[0]))  # Smooth the X coordinate
    y = int(prev[1] + alpha * (cur[1] - prev[1]))  # Smooth the Y coordinate
    return (x, y)
#checking if the finger is up
def finger_is_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y
#converting x position into pallete index to choose colore
def colour_index_from_x(x, w):
    block = w // len(PALETTE)  # Width of each colour block in the top bar
    return max(0, min(len(PALETTE) - 1, x // block))
# Image masking and overlay using NumPy boolean indexing
# https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
#copying only the drawing pixeles
def paste_layer(base, layer):
    mask = np.any(layer != 0, axis=2)
    base[mask] = layer[mask]
#saving a screenshot of the drawing and the user in the same time
def save_frame(img_bgr):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") # adding real time and date to the name of the screenshot
    name = f"screenshot-{stamp}.png"
    cv2.imwrite(name, img_bgr) #saving into the folder
    return name

# Line detection using PCA (eigenvalues/eigenvectors)
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
#Trying to convert stroke points into a snapped straight line
def try_snap_line(points):
    if len(points) < CONFIG["min_stroke_points"]:  # If not enough points
        return None  #no snapping

    pts = np.array(points, dtype=np.float32) #Converting list of (x,y) points into NumPy array
    mean = np.mean(pts, axis=0) #mean point calculation
    centered = pts - mean  #Centering points

    cov = np.cov(centered.T)  #Build covariance matrix of x/y
    eigvals, eigvecs = np.linalg.eig(cov)  #Compute eigenvalues

    order = np.argsort(eigvals)[::-1]  #Sortong eigenvalues
    eigvals = eigvals[order]  #Reordering eigenvalues
    eigvecs = eigvecs[:, order]  #Reordering eigenvectors

#checking if the line is not valid
    r2 = float(eigvals[0] / (eigvals[0] + eigvals[1] + 1e-6))  #Ratio of main variance to total
    if r2 < CONFIG["line_r2_min"]:
        return None
    #direction
    direction = eigvecs[:, 0]
    t = centered @ direction
    tmin, tmax = float(np.min(t)), float(np.max(t))
    #computing endpoints
    p1 = mean + tmin * direction
    p2 = mean + tmax * direction
    #convering endpoints into pixels
    P1 = (int(p1[0]), int(p1[1]))
    P2 = (int(p2[0]), int(p2[1]))
    #checking if the line is too short
    if distance(P1, P2) < CONFIG["min_line_length"]:
        return None

    return (P1, P2)  #Returning the snapped line endpoints

# Contour detection and circularity check
# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
# Morphological operations (closing gaps)
# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
#Try to convert stroke points into a snapped circle
def try_snap_circle(points):
    if len(points) < CONFIG["min_stroke_points"]: #If not enough points
        return None  # Do not attempt snapping

    pts = np.array(points, dtype=np.float32)  #Converting points list to NumPy array
    #computing minimus and maximus
    minx, miny = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
    maxx, maxy = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

    #checking if the strokes are to small to be a circle
    if (maxx - minx) < 30 and (maxy - miny) < 30:
        return None
    #selecting the roi of the circle
    pad = 18
    x1 = max(0, minx - pad)
    y1 = max(0, miny - pad)
    x2 = min(CONFIG["width"] - 1, maxx + pad)
    y2 = min(CONFIG["height"] - 1, maxy + pad)
    roi_w = x2 - x1 + 1
    roi_h = y2 - y1 + 1
    roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

    #converting stroke into ROI
    prev = None
    for (x, y) in points: #Loop through stroke points
        rx, ry = x - x1, y - y1
        if prev is not None:
            cv2.line(roi, prev, (rx, ry), 255, 3, cv2.LINE_AA)
        prev = (rx, ry)  #updating previous point

    #closing small gapd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #findig ROI contours
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)  #Taking the largest contour
    area = float(cv2.contourArea(c))  #Computing contour area
    if area < CONFIG["min_contour_area"]:  #If contour is too small
        return None

    peri = float(cv2.arcLength(c, True))  #Computing contour perimeter
    if peri < 1e-6:  #Avoiding divide-by-zero
        return None

    circularity = (4.0 * np.pi * area) / (peri * peri)  #Circularity measure
    if circularity < CONFIG["circularity_min"]:  #If not circular enough
        return None

    (cx, cy), r = cv2.minEnclosingCircle(c)  #getting the radius
    if r < CONFIG["min_circle_radius"]:  #If circle radius too small
        return None

    gx, gy, gr = int(x1 + cx), int(y1 + cy), int(r)  #Converting circle centre back to coords
    return ((gx, gy), gr)  #Return snapped circle

# OpenCV drawing functions (line, circle, rectangle, text)
# https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
#UI FOR THE PROGRAM
#top bar and status text CONFIG
def draw_top_ui(img, colour_idx, mode_text, info_text):
    w = CONFIG["width"]
    hbar = CONFIG["top_bar_height"]

    cv2.rectangle(img, (0, 0), (w, hbar), (20, 20, 20), -1) #bar background

    block = w // len(PALETTE)  # Width of each colour button area
    # Looping over palette entries to  draw the butons and the selection outline
    for i, (_, bgr) in enumerate(PALETTE):
        x1 = i * block
        x2 = (i + 1) * block
        cv2.rectangle(img, (x1 + 6, 10), (x2 - 6, hbar - 10), bgr, -1) # Draw color button
        if i == colour_idx:
            cv2.rectangle(img, (x1 + 4, 8), (x2 - 4, hbar - 8), (255, 255, 255), 2)

    cv2.putText(img, mode_text, (10, hbar + 28),  #Drawing the mode label
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)  # Font, size, colour, thickness

    if info_text:  #If we have extra info to show
        cv2.putText(img, info_text, (10, hbar + 55),  # Draw info line below mode
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)  # Font, size, colour, thickness



def main(): #Main function
    W, H = CONFIG["width"], CONFIG["height"]  #Read width and height from config
    bar_h = CONFIG["top_bar_height"]  #Height of the colour bar
    safe_y = bar_h + CONFIG["top_bar_padding"]  # Y coordinate under which drawing is allowed

    colour_idx = 0  #Starting with white colour
    brush_colour = PALETTE[colour_idx][1]  #Get the BGR colour tuple

    brush_thick = CONFIG["brush_thickness"]  #Brush thickness
    eraser_r = CONFIG["eraser_radius"]  #Eraser radius

    #making sure the drawings are on the safe layer
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    preview = np.zeros((H, W, 3), dtype=np.uint8)

    # Variables used to track current drawing stroke
    prev_pt = None
    smooth_pt = None
    stroke_points = []
    was_drawing = False

    # stabilising finger detection
    n = CONFIG["vote_frames"]
    vote_index = deque(maxlen=n)
    vote_middle = deque(maxlen=n)
    vote_ring = deque(maxlen=n)
    vote_pinky = deque(maxlen=n)

   #Variables to prevent accidental  saves
    save_hold = 0
    save_cooldown = 0

    #info messages
    info_text = ""  # Message string
    info_timer = 0  # Timer countdown for message display

    # OpenCV camera capture (VideoCapture)
    # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    # Open the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)  # Set camera width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)  # Set camera height

    mp_hands = mp.solutions.hands  # MediaPipe hands module alias
    mp_draw = mp.solutions.drawing_utils  # MediaPipe drawing utilities
    # MediaPipe Hands detector (real-time hand landmark tracking)
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    with mp_hands.Hands(  #MediaPipe Hands detector
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:  #Main frame loop
            ok, frame = cap.read()  # Read a frame from the camera
            if not ok:
                break

            frame = cv2.flip(frame, 1)  #Mirror the frame
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            if save_cooldown > 0:
                save_cooldown -= 1

            if info_timer > 0:
                info_timer -= 1
            else:  # If timer finished
                info_text = ""  # Clear info text
            # OpenCV color space conversion (BGR to RGB)
            # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
            results = hands.process(rgb)

            mode = "IDLE"  # Default mode at start of frame
            drawing_now = False  # Whether drawing is active this frame
            trigger_save = False  #Whether save is triggered this frame

            if results.multi_hand_landmarks:  #If at least one hand detected
                hand = results.multi_hand_landmarks[0]  #Use the first hand in the camera
                lms = hand.landmark

                tip_raw = lm_to_px(lms[mp_hands.HandLandmark.INDEX_FINGER_TIP], W, H)  #Index fingertip pixel coords
                smooth_pt = ema(smooth_pt, tip_raw, CONFIG["smooth_alpha"])  #Smooth fingertip position

                idx_up_raw = finger_is_up(  #index finger up checking
                    lms,
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP
                )

                mid_up_raw = finger_is_up(  #middle finger up checking
                    lms,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                )  #

                ring_up_raw = finger_is_up(  #ring finger up checking
                    lms,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP
                )

                pink_up_raw = finger_is_up(  #pinky finger up checking
                    lms,
                    mp_hands.HandLandmark.PINKY_TIP,
                    mp_hands.HandLandmark.PINKY_PIP
                )

                vote_index.append(1 if idx_up_raw else 0)
                vote_middle.append(1 if mid_up_raw else 0)
                vote_ring.append(1 if ring_up_raw else 0)
                vote_pinky.append(1 if pink_up_raw else 0)

               #stabilising gesture detection by majority vote
                idx_up = sum(vote_index) >= (len(vote_index) // 2 + 1)  # Stable index up
                mid_up = sum(vote_middle) >= (len(vote_middle) // 2 + 1)  # Stable middle up
                ring_up = sum(vote_ring) >= (len(vote_ring) // 2 + 1)  # Stable ring up
                pink_up = sum(vote_pinky) >= (len(vote_pinky) // 2 + 1)  # Stable pinky up

                draw_gesture = idx_up and (not mid_up) and (not ring_up) and (not pink_up)  #Only index up
                fist_gesture = (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up)  #All down
                peace_gesture = idx_up and mid_up and (not ring_up) and (not pink_up)  #Index+middle up

                #making sure the selection in colors tab be smooth
                if smooth_pt is not None and smooth_pt[1] <= safe_y:  # If fingertip is inside top area
                    mode = "UI"  # Switch to UI mode
                    colour_idx = colour_index_from_x(smooth_pt[0], W)
                    brush_colour = PALETTE[colour_idx][1]
                    save_hold = 0
                else:
                    if peace_gesture and save_cooldown == 0:
                        save_hold += 1
                        if save_hold >= CONFIG["save_hold_frames"]:
                            trigger_save = True  # Trigger a save
                            save_hold = 0
                    else:  # If not holding peace gesture
                        save_hold = 0  # Reset hold counter

                    if trigger_save:  # If save triggered
                        mode = "SAVE"
                    elif fist_gesture:
                        mode = "ERASE"
                    elif draw_gesture:
                        mode = "DRAW"
                        drawing_now = True
                    else:
                        mode = "IDLE"

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)  #Draw the hand skeleton on camera frame

            else:  # If no hand is detected
            #reset
                mode = "IDLE"
                smooth_pt = None
                vote_index.clear()
                vote_middle.clear()
                vote_ring.clear()
                vote_pinky.clear()
                save_hold = 0

            #making sure the smart snap goes after fnishing the strokes
            if was_drawing and (not drawing_now):  # If we were drawing, but now stopped
                if len(stroke_points) >= CONFIG["min_stroke_points"]:
                    circle = try_snap_circle(stroke_points)
                    if circle is not None:  # If circle snap succeeded
                        (cx, cy), r = circle
                        cv2.circle(canvas, (cx, cy), r, brush_colour, brush_thick, cv2.LINE_AA)
                        info_text = "AI SNAP: CIRCLE"  # Set info message
                        info_timer = 90  # Show message for 90 frames
                    else:  # If circle snap failed
                        line = try_snap_line(stroke_points)
                        if line is not None:
                            p1, p2 = line
                            cv2.line(canvas, p1, p2, brush_colour, brush_thick, cv2.LINE_AA)
                            info_text = "AI SNAP: LINE"  # Set info message
                            info_timer = 90  # Show message for 90 frames
                        else:  # If both circle and line snap failed
                            paste_layer(canvas, preview)
                            info_text = "FREEHAND"
                            info_timer = 55  # Show message for 55 frames
                else:  # If stroke is too short for snapping
                    paste_layer(canvas, preview)  #Just commit the preview stroke to canvas
                #reset
                preview[:] = 0
                stroke_points = []
                prev_pt = None

            # Applying drawing/erasing/saving based on current mode
            if mode == "DRAW" and smooth_pt is not None:  # If in drawing mode and we have a point
                if smooth_pt[1] > safe_y:
                    if prev_pt is not None:  # If we have a previous point
                        cv2.line(preview, prev_pt, smooth_pt, brush_colour, brush_thick, cv2.LINE_AA)
                    prev_pt = smooth_pt
                    stroke_points.append(smooth_pt)
                else:  #If finger goes into the bar area
                    prev_pt = None

            elif mode == "ERASE" and smooth_pt is not None:  # If in erase mode and we have a point
                if smooth_pt[1] > safe_y:
                    cv2.circle(canvas, smooth_pt, eraser_r, (0, 0, 0), -1)
                prev_pt = None  #Reset previous point

            elif mode == "SAVE":  # If in save mode
                combined_now = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
                paste_layer(combined_now, preview)
                fname = save_frame(combined_now)  #Save combined image and get filename
                info_text = f"SAVED: {fname}"
                info_timer = 140
                save_cooldown = CONFIG["save_cooldown"]
                prev_pt = None  # Reset previous point

            else:  #For UI and any other mode
                prev_pt = None
            was_drawing = drawing_now

            # OpenCV image blending using addWeighted
            # https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
            #Building the frame we will show to the user
            out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)  # Combine camera with permanent canvas
            paste_layer(out, preview)  # Overlay preview stroke

            if mode == "UI":  # If in UI mode
                mode_text = f"COLOUR: {PALETTE[colour_idx][0]}"  # Display selected colour name
            elif mode == "DRAW":  # If drawing
                mode_text = f"DRAW ({PALETTE[colour_idx][0]})"  # Display draw status + colour
            elif mode == "ERASE":  # If erasing
                mode_text = "ERASE (fist)"  # Display erase status
            elif mode == "SAVE":  # If saving
                mode_text = "SAVE (peace)"  # Display save status
            else:  # Otherwise
                mode_text = "default"  # Display deafault status

            draw_top_ui(out, colour_idx, mode_text, info_text)  #Drawing UI bar and texts
            cv2.imshow("Air Draw (Q quit, C clear)", out)  # Show output window

            key = cv2.waitKey(1) & 0xFF  # Read keyboard input
            if key in (ord("q"), ord("Q")):  #If user presses q
                break  # Exit loop
            if key in (ord("c"), ord("C")):  # If user presses c
            #reset everything
                canvas[:] = 0
                preview[:] = 0
                stroke_points = []
                prev_pt = None
                smooth_pt = None
                vote_index.clear()
                vote_middle.clear()
                vote_ring.clear()
                vote_pinky.clear()
                was_drawing = False
                save_hold = 0
                save_cooldown = 0
                info_text = ""
                info_timer = 0

    cap.release()  #Releasing the camera
    cv2.destroyAllWindows()  #Closing all windows


if __name__ == "__main__":
    main()
