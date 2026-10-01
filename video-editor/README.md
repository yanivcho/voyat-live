# vedit: מנוע עריכת וידאו

המנוע הוא Python שמרכיב כל פריים בעצמו עם OpenCV ו-numpy, ומשתמש ב-ffmpeg לפענוח, לקידוד ולסאונד.
טקסט עברי מצויר תמיד עם Pillow ומנוע RAQM (bidi + HarfBuzz), כך שהוא יוצא מימין לשמאל ולא הפוך.

## התקנה
```bash
./setup.sh            # venv ב-/opt/vedit-venv, whisper.cpp, מודלים (silero VAD, u2net_human_seg, ivrit.ai)
```
מודל התמלול של ivrit.ai יורד מ-huggingface.co, ולכן הסביבה צריכה גישה לדומיין הזה.

## זרימת עבודה
```bash
PY=/opt/vedit-venv/bin/python
$PY -m vedit new path/to/video.mp4 --name my-video   # יוצר projects/my-video ומנרמל את המקור
$PY -m vedit transcribe projects/my-video            # מילים עם חותמות זמן -> work/words.md
$PY -m vedit mask projects/my-video                  # מסכת אדם לכל פריים -> work/masks.npy (פעם אחת)
$PY -m vedit render projects/my-video --name v1      # מרנדר את edit.py -> out/v1.mp4
$PY -m vedit contact projects/my-video projects/my-video/out/v1.mp4   # גיליון קונטקט, 6 פריימים לשנייה
$PY selftest.py                                      # בדיקה מקצה לקצה מול אמת ידועה (צריך espeak-ng)
```

## איך מתארים עריכה (`projects/<name>/edit.py`)
```python
def build(p):
    p.set_format("vertical")        # vertical | horizontal | square ; fit="contain" = כל הפריים + רקע מטושטש
    p.set_style("bold")             # clean | bold (vedit/styles.py)
    p.cut_silences()                # חיתוך שתיקות עם מעבר רך (וידאו ואודיו)
    p.captions()                    # כתוביות מילה-מילה מהתמלול
    p.zoom(at="שלום", scale=1.12, until=14)            # פאנץ' על מילה
    p.text("כותרת", at=2.0, until=5.5)                  # טקסט חופשי
    p.add(BehindPerson(Captions(STYLES["bold"])))       # טקסט מאחורי הדמות
    p.sound("whoosh.wav", at=12)                        # אפקט קולי על מילה
    p.freeze(at=20, hold=1.0)                           # פריים קפוא
```
**עוגנים:** מספר שלם הוא מספר מילה בטבלת המילים, מחרוזת היא המילה עצמה, ומספר עשרוני (1.5) הוא שניות במקור.
כל אפקט מעוגן לזמן במקור. מפת הזמן (`timemap.py`) ממירה אותו לזמן הסופי, והסאונד נחתך לפי אותה מפה, כך שהכול נשאר מסונכרן.

## מבנה
| קובץ | תפקיד |
|---|---|
| `media.py` | probe, נרמול המקור (קצב פריימים קבוע, סיבוב, HDR→SDR) וקריאת פריימים |
| `timemap.py` | מפת זמן בין המקור לפלט: חיתוכים, מעברים רכים ופריים קפוא |
| `audio.py` | זיהוי דיבור (Silero VAD), חיתוך הסאונד לפי המפה, מיקס ונרמול עוצמה ל-‎-14 LUFS |
| `text.py` | טקסט עברי (RAQM), קומפוזיציה עם premultiplied alpha, סדר bidi למילים |
| `captions.py` | כתוביות מונפשות (עמודים, שורות מאוזנות, הדגשת המילה הנאמרת) |
| `layers.py` | Grade, Letterbox, BehindPerson, ReplaceBackground |
| `mask.py` | rembg u2net_human_seg לכל פריים, נשמר ל-masks.npy |
| `project.py` | `frame_at(t)` (פונקציית הרינדור), רינדור מקבילי בכל הליבות וקידוד H.264 עם סאונד |
| `contact.py` | גיליון קונטקט: זמן פלט וזמן מקור על כל פריים, וצורת גל עם סימון חיתוכים |
| `transcribe.py` | whisper.cpp עם ivrit-large-v3-turbo, טבלת מילים (md/csv/json) |

הסרטונים, התמלולים, המסכות והרינדורים (`projects/*`) לא נכנסים ל-git.
הפונטים Heebo ו-Rubik נמצאים תחת רישיון SIL OFL (`fonts/`).
