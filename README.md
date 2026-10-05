# Air Draw

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-university%20project-lightgrey)

![stack](https://icon-marquee.giann.dev/v1/icons?i=py,opencv,numpy)

Draw on your screen with your hand, no mouse or touchscreen needed. Air Draw tracks your hand through a webcam, so you can draw, erase, change colours and save a screenshot using finger gestures alone. Strokes that look like a circle or a straight line are cleaned up automatically.

The idea came from teaching: writing straight onto the screen instead of standing at a board.

<!-- Add a demo GIF here: ![Air Draw demo](docs/demo.gif) -->

## Gestures

| Gesture | Action |
|---|---|
| Index finger up, other fingers down | Draw |
| Closed fist | Erase (circular eraser) |
| Index and middle finger up (peace sign), held briefly | Save a screenshot of the camera view plus your drawing |
| Fingertip inside the top colour bar | Pick a colour |
| `C` key | Clear the canvas |
| `Q` key | Quit |

Six colours are available: white, blue, green, red, yellow and purple. The current mode and colour are shown on screen.

## Shape snapping

When you lift your finger at the end of a stroke, the program checks whether the stroke was meant to be a shape. It uses classical computer vision. There is no trained model.

1. **Circle:** the stroke is rendered into a small mask, gaps are closed with a morphological operation, and the largest contour is found. If its circularity (4 x pi x area / perimeter squared) is at least 0.72 and it is large enough, it is replaced with a clean circle.
2. **Straight line:** if it is not a circle, PCA is run on the stroke points. If the first principal component explains at least 93% of the variance, the stroke is replaced with a straight line between its two extremes.
3. **Otherwise** it is kept as a freehand stroke.

Short or tiny strokes are never snapped. All thresholds are in the `CONFIG` dictionary at the top of `air_draw.py`.

## How it works

- **Hand tracking:** MediaPipe Hands returns 21 landmarks per frame. A finger counts as up when its tip is above its middle joint.
- **Stable gestures:** each finger's state is decided by a majority vote over the last 7 frames, so one noisy frame doesn't switch modes.
- **Smooth lines:** the fingertip position goes through an exponential moving average (alpha 0.35) to remove jitter.
- **Two layers:** the stroke being drawn lives on a temporary preview layer and is only committed to the permanent canvas when it ends. This is what makes snapping possible, and it keeps erasing from touching the camera image.
- **Safe zones:** drawing and erasing are disabled in the colour bar area, so choosing a colour never leaves a mark.
- **Accidental saves:** the peace sign has to be held for 6 frames, and there is a cooldown after each save.

## Getting started

Python 3.10 is recommended. MediaPipe does not work well on 3.12 or newer, and NumPy must stay on 1.x.

```bash
git clone https://github.com/mohammedadalachi/air-draw-gesture-drawing.git
cd air-draw-gesture-drawing
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python air_draw.py
```

Use good lighting and keep your hand facing the camera with your fingers pointing up. Screenshots are saved next to the script as `screenshot-<date>_<time>.png`.

If the camera does not open, close other apps that use the webcam, or change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

## Project structure

```
air_draw.py         the whole application
requirements.txt
```

## Limitations

- One hand only
- A finger counts as up by comparing heights in the image, so the hand needs to stay roughly upright
- Poor lighting hurts tracking
- Snapping can turn an intentionally wobbly line into a straight one
- The canvas is fixed at 640 x 480

## Ideas for improvement

- Undo, adjustable brush size and more colours
- More shapes (rectangles, triangles, arrows)
- Rotation-independent finger detection using joint angles
- Writing on top of a shared screen instead of a camera feed

## Author

Dalachi Mohammed Abderrahmane ([@mohammedadalachi](https://github.com/mohammedadalachi))

Developed as a university project for AIT102 Python and Tensorflow Programming.

## License

MIT. See [LICENSE](LICENSE).
