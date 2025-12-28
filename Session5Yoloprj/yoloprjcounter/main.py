import cv2
from ultralytics import solutions

# ===============================
# تنظیمات
# ===============================
VIDEO_PATH = "v1.mp4"
MODEL_PATH = "best(2).pt"

# خط شمارش (افقی – مناسب حرکت بالا به پایین)
LINE_P1 = (100, 750)
LINE_P2 = (620, 750)

CONF = 0.5       # آستانه confidence برای تشخیص اشیا (0 تا 1)
IOU = 0.6        # آستانه IoU برای tracker

# سرعت نمایش فریم (میلی‌ثانیه بین فریم‌ها)
# کاهش عدد → سریع‌تر، افزایش → کندتر
FRAME_DELAY = 4  # 14s برای تقریباً سرعت واقعی ویدیو

# ===============================
# بارگذاری کانتر
# ===============================
counter = solutions.ObjectCounter(
    model=MODEL_PATH,
    region=[LINE_P1, LINE_P2],
    classes=[0],            # فقط luggage
    show=True,              # باکس + شمارنده
    show_in=True,
    show_out=True,
    line_width=3,
    conf=CONF,
    iou=IOU
)

# ===============================
# ویدیو
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("ویدیو باز نشد!")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(
    "output_counted.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ===============================
# شمارنده‌ها
# ===============================
frame_id = 0

# فونت حرفه‌ای و تمیز
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.7
FONT_COLOR_FRAME = (0, 255, 255)
FONT_COLOR_OBJ = (255, 255, 255)
FONT_COLOR_TOTAL = (0, 200, 255)
FONT_COLOR_CURRENT = (0, 255, 0)
LINE_COLOR = (0, 0, 255)
LINE_THICK = 3
PANEL_COLOR = (0, 0, 0)
PANEL_ALPHA = 0.6  # شفافیت پنل

print("شروع پردازش... ESC برای خروج")

# ===============================
# حلقه اصلی
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # پردازش با ObjectCounter
    results = counter(frame)  # ← SolutionResults

    # گرفتن تصویر رسم‌شده
    output = results.imgs[0] if hasattr(results, 'imgs') else frame

    # رسم خط شمارش
    cv2.line(output, LINE_P1, LINE_P2, LINE_COLOR, LINE_THICK)
    cv2.putText(
        output,
        "COUNT LINE",
        (LINE_P1[0], LINE_P1[1] - 10),
        FONT,
        0.8,
        LINE_COLOR,
        2
    )

    # رسم پنل شفاف
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (400, 180), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, output, 1 - PANEL_ALPHA, 0, output)

    # نمایش اطلاعات روی پنل
    panel_x, panel_y = 20, 40
    line_h = 35
    cv2.putText(output, f"Frame: {frame_id}", (panel_x, panel_y),
                FONT, FONT_SCALE, FONT_COLOR_FRAME, 2)
    cv2.putText(output, f"Total seen (In): {counter.in_count}", (panel_x, panel_y + 3 * line_h),
                FONT, FONT_SCALE, FONT_COLOR_TOTAL, 2)

    # نمایش و ذخیره
    cv2.imshow("Luggage Counter", output)
    video_writer.write(output)

    if cv2.waitKey(FRAME_DELAY) & 0xFF == 27:
        break

# ===============================
# پایان
# ===============================
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("پردازش تمام شد")
print(f"Total luggage counted: {counter.in_count}")
