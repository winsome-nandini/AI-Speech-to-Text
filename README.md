# AI-Speech-to-Text
-----------------------------------------------------------------------
'''
Code Video Link Of Drive 
https://drive.google.com/file/d/1OndsRLoWbsbqplX4o7Ri0wB2gs2GJJph/view?usp=sharing
'''
------------------------------------------------------------------------
#Code in Python
------------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox, ttk
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import threading
import time
import os
import webbrowser
from datetime import datetime
import numpy as np

# Try importing dependencies with fallback handling
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set TkAgg backend
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Matplotlib import error: {e}")
    messagebox.showerror("Startup Error", "Matplotlib is not installed or backend_tkagg is unavailable. Install with 'pip install matplotlib'.")
    exit(1)

try:
    from googletrans import Translator
except ImportError:
    messagebox.showwarning("Warning", "Googletrans not installed. Translation disabled.")
    def translate_text(text, target_language="en"):
        return text  # Fallback to original text

# NLTK and TextBlob setup with data download and fallback
try:
    import nltk
    nltk.data.find('tokenizers/punkt')  # Check if punkt is available
    nltk.data.find('corpora/stopwords')  # Check if stopwords is available
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Download only if not found (quiet mode)
    if not os.path.exists(nltk.data.find('tokenizers/punkt')):
        nltk.download('punkt', quiet=True)
    if not os.path.exists(nltk.data.find('corpora/stopwords')):
        nltk.download('stopwords', quiet=True)
    from textblob import TextBlob  # Import TextBlob for sentiment analysis
except ImportError as e:
    messagebox.showwarning("Warning", f"NLTK or TextBlob not installed. Word/sentiment analysis disabled. Install with 'pip install nltk textblob'.")
    def word_tokenize(text):
        return text.split()
    stopwords = set()
    def get_sentiment(text):
        return {"polarity": 0.0, "subjectivity": 0.0}  # Fallback
except LookupError as e:
    messagebox.showwarning("Warning", f"NLTK data missing: {e}. Downloading now...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from textblob import TextBlob

# Simulated database with thread safety
stories_archive = []
stories_lock = threading.Lock()

# Real transcription using speech_recognition with enhanced reliability
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000  # Lower threshold for quieter audio
    recognizer.dynamic_energy_threshold = True  # Adapt to background noise
    try:
        if not os.path.exists(filename):
            print(f"Error: Audio file {filename} not found")
            return "Audio file not found"
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)  # Record the entire file
            print(f"Audio duration: {len(audio.get_raw_data()) / 16000} seconds")  # Debug
            for attempt in range(3):  # Retry up to 3 times
                try:
                    print(f"Transcribing attempt {attempt + 1}: {filename}")
                    result = recognizer.recognize_google(audio)
                    if not result or result.strip() == "":
                        print("Warning: Empty transcription from Google API")
                        return "Transcription empty"
                    return result
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return "Could not understand audio"
                except sr.RequestError as e:
                    if attempt < 2:
                        print(f"Request error: {e}. Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    print(f"Transcription error after retries: {e}")
                    return f"Transcription error: {e} (Check internet connection)"
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    return f"Error processing audio: {e}"
            print("Transcription failed after multiple attempts")
            return "Transcription failed after multiple attempts"
    except Exception as e:
        print(f"Unexpected error in transcription: {e}")
        return f"Unexpected error in transcription: {e}"

# Translation with googletrans
def translate_text(text, target_language="en"):
    try:
        translator = Translator()
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        messagebox.showwarning("Translation Error", f"Translation failed: {e}. Using original text.")
        return text

# Analyze and translate words
def analyze_and_translate_words(text, target_language="en"):
    try:
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        key_words = [word for word in words if word.isalnum() and word not in stop_words]
        key_words = list(dict.fromkeys(key_words))
        if not key_words:
            return {}
        translator = Translator()
        translated_words = {}
        for word in key_words:
            try:
                translated_words[word] = translator.translate(word, dest=target_language).text
            except Exception:
                translated_words[word] = word
        return translated_words
    except Exception as e:
        messagebox.showwarning("Analysis Error", f"Word analysis failed: {e}. Returning empty.")
        return {}

# Get sentiment using TextBlob
def get_sentiment(text):
    try:
        blob = TextBlob(text)
        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        return sentiment
    except Exception as e:
        messagebox.showwarning("Sentiment Error", f"Sentiment analysis failed: {e}. Returning neutral.")
        return {"polarity": 0.0, "subjectivity": 0.0}

# Clean up old audio files
def cleanup_old_files(directory=".", max_files=10):
    files = sorted([f for f in os.listdir(directory) if f.startswith("story_") and f.endswith(".wav")], key=os.path.getmtime)
    for f in files[:-max_files]:
        try:
            os.remove(os.path.join(directory, f))
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

# Record audio with real-time waveform
def record_audio(app, canvas, stop_event):
    fs = 16000  # Adjusted to 16kHz for compatibility with speech recognition
    filename = f"story_{int(time.time())}.wav"
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())
        print(f"Audio chunk length: {len(indata)}")

    try:
        stream = sd.InputStream(samplerate=fs, channels=1, callback=callback, dtype='int16')
        stream.start()

        fig, ax = plt.subplots(figsize=(8, 1.5))
        canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
        canvas_width = canvas.winfo_width() or 1150
        canvas_height = canvas.winfo_height() or 500
        x_pos = (canvas_width - 800) // 2
        y_pos = 250 + (250 - 150) // 2
        canvas_widget.get_tk_widget().place(x=x_pos, y=y_pos)
        app.canvas_items.append(canvas_widget.get_tk_widget())

        def update_waveform():
            if not stop_event.is_set():
                if audio_data:
                    data = np.concatenate(audio_data, axis=0)
                    ax.clear()
                    ax.plot(data, color="#4CAF50")
                    ax.axis("off")
                    canvas_widget.draw()
                app.root.after(100, update_waveform)
            else:
                ax.clear()
                canvas_widget.draw()
                canvas_widget.get_tk_widget().place_forget()
                app.update_status("Stopped")

        app.root.after(100, update_waveform)
        app.update_status("Recording...")
        return stream, audio_data, filename
    except Exception as e:
        app.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {e}"))
        return None, None, None

# Save audio after recording
def save_audio(audio_data, filename, fs=16000):
    try:
        if not audio_data:
            print("Error: No audio data captured")
            return None
        data = np.concatenate(audio_data, axis=0)
        print(f"Audio data length before save: {len(data)}")
        if len(data) == 0:
            print("Error: Empty audio data")
            return None
        print(f"Saving audio to: {filename}")
        write(filename, fs, data)
        cleanup_old_files()
        return filename
    except Exception as e:
        messagebox.showerror("Error", f"Saving audio failed: {e}")
        return None

# Process and store story with sentiment analysis
def process_story(filename, user_id="anonymous"):
    try:
        transcription = transcribe_audio(filename)
        translated = translate_text(transcription)
        analyzed_words = analyze_and_translate_words(transcription)
        sentiment = get_sentiment(transcription)
        story = {
            "user_id": user_id,
            "transcription": transcription,
            "translated_text": translated,
            "analyzed_words": analyzed_words,
            "sentiment": sentiment,
            "audio_path": os.path.abspath(filename) if filename else None,
            "timestamp": time.time()
        }
        with stories_lock:
            stories_archive.append(story)
        return story
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {e}")
        return None

# Create playlist
def create_playlist(user_id="anonymous", max_stories=5):
    with stories_lock:
        user_stories = sorted(
            [s for s in stories_archive if s["user_id"] == user_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )
    return user_stories[:max_stories]

# GUI class with enhanced graphics
class StoryWeaverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Story Weaver Network")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f4f8")

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 12), background="#f0f4f8")
        style.configure("TEntry", font=("Arial", 12))
        style.configure("Stop.TButton", font=("Arial", 12), padding=10, background="#FF0000", foreground="white")
        style.map("Stop.TButton", background=[("active", "#CC0000")], foreground=[("active", "white")])
        style.map("TButton", background=[("active", "#4CAF50")], foreground=[("active", "white")])
        style.configure("Text.TButton", font=("Arial", 10), padding=5)

        self.frame = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        self.frame.pack(fill=tk.BOTH, padx=15, pady=15, expand=True)

        ttk.Label(self.frame, text="Story Weaver Network", font=("Arial", 18, "bold"), foreground="#333").pack(pady=10)

        input_frame = ttk.Frame(self.frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Name:").pack(side=tk.LEFT, padx=5)
        self.user_id = ttk.Entry(input_frame, width=20)
        self.user_id.pack(side=tk.LEFT, padx=5)

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.record_button = ttk.Button(button_frame, text="🎤 Record Story", command=self.start_recording, style="TButton")
        self.record_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📜 Show Playlist", command=self.show_playlist, style="TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗣 Voice Command", command=self.start_voice_command, style="TButton").pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="🛑 Stop Recording", command=self.stop_recording, style="Stop.TButton")
        self.stop_button.pack_forget()

        self.canvas = tk.Canvas(root, width=1150, height=500, bg="white", highlightthickness=2, highlightbackground="#ccc")
        self.canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        self.canvas_items = []

        self.status_text = self.canvas.create_text(575, 20, text="Ready", font=("Arial", 12), fill="#333")
        self.canvas_items.append(self.status_text)

        for i in range(500):
            shade = f"#F5F6F{i % 2}"
            self.canvas.create_line(0, i, 1150, i, fill=shade)

        self.stop_event = None
        self.stream = None
        self.audio_data = None
        self.current_story = None

    def update_status(self, status):
        self.canvas.itemconfig(self.status_text, text=status)

    def clear_canvas(self):
        for item in self.canvas_items:
            if isinstance(item, tk.Widget):
                item.place_forget()
            else:
                self.canvas.delete(item)
        self.canvas_items = []

    def show_story_text(self, story):
        self.clear_canvas()
        y_pos = 40
        self.canvas.create_text(575, y_pos, text="Story Weaver Network", font=("Arial", 18, "bold"), fill="#333")
        y_pos += 50
        if story and story.get("audio_path"):
            self.canvas.create_rectangle(10, y_pos, 1140, y_pos + 200, fill="#e8f4f8", outline="#ccc")  # Increased height to accommodate sentiment
            self.canvas_items.append(
                self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Story: {story.get('transcription', 'No transcription')}", font=("Arial", 14), width=1100)
            )
            y_pos += 50
            self.canvas_items.append(
                self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Translated: {story.get('translated_text', 'No translation')}", font=("Arial", 12), width=1100)
            )
            y_pos += 40
            if os.path.exists(story.get('audio_path', '')) and story['audio_path'].endswith(".wav"):
                audio_text = self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Audio: {os.path.basename(story['audio_path'])}", font=("Arial", 10, "underline"), fill="#4CAF50", tags="clickable")
                self.canvas.tag_bind("clickable", "<Button-1>", lambda e: webbrowser.open(f"file://{story['audio_path']}"))
                self.canvas.tag_bind("clickable", "<Enter>", lambda e: self.canvas.itemconfig(audio_text, fill="#2E7D32"))
                self.canvas.tag_bind("clickable", "<Leave>", lambda e: self.canvas.itemconfig(audio_text, fill="#4CAF50"))
            y_pos += 40
            if story.get("analyzed_words"):
                self.canvas_items.append(
                    self.canvas.create_text(20, y_pos + 10, anchor="nw", text="Analyzed Words:", font=("Arial", 12, "bold"), fill="#333")
                )
                y_pos += 20
                for word, translation in story["analyzed_words"].items():
                    text = f"{word} -> {translation}"
                    self.canvas_items.append(
                        self.canvas.create_text(20, y_pos + 10, anchor="nw", text=text, font=("Arial", 10), fill="#000000")
                    )
                    y_pos += 20
            if story.get("sentiment"):
                y_pos += 10
                polarity = story["sentiment"]["polarity"]
                subjectivity = story["sentiment"]["subjectivity"]
                sentiment_text = f"Sentiment: {'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'} (Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f})"
                self.canvas_items.append(
                    self.canvas.create_text(20, y_pos + 10, anchor="nw", text=sentiment_text, font=("Arial", 12), fill="#333")
                )
        self.root.after(5000, lambda: self.show_playlist())

    def update_canvas(self, story=None, playlist=None):
        self.clear_canvas()
        y_pos = 40
        self.canvas.create_text(575, y_pos, text="Story Weaver Network", font=("Arial", 18, "bold"), fill="#333")
        y_pos += 50

        if story and story.get("audio_path"):
            self.canvas.create_rectangle(10, y_pos, 1140, y_pos + 200, fill="#e8f4f8", outline="#ccc")  # Increased height to accommodate sentiment
            transcription_text = story.get('transcription', 'No transcription')
            self.canvas_items.append(
                self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Story: {transcription_text}", font=("Arial", 14), width=1100)
            )
            y_pos += 50
            self.canvas_items.append(
                self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Translated: {story.get('translated_text', 'No translation')}", font=("Arial", 12), width=1100)
            )
            y_pos += 40
            if os.path.exists(story.get('audio_path', '')) and story['audio_path'].endswith(".wav"):
                audio_text = self.canvas.create_text(20, y_pos + 10, anchor="nw", text=f"Audio: {os.path.basename(story['audio_path'])}", font=("Arial", 10, "underline"), fill="#4CAF50", tags="clickable")
                self.canvas.tag_bind("clickable", "<Button-1>", lambda e: webbrowser.open(f"file://{story['audio_path']}"))
                self.canvas.tag_bind("clickable", "<Enter>", lambda e: self.canvas.itemconfig(audio_text, fill="#2E7D32"))
                self.canvas.tag_bind("clickable", "<Leave>", lambda e: self.canvas.itemconfig(audio_text, fill="#4CAF50"))
            y_pos += 40
            if story.get("analyzed_words"):
                self.canvas_items.append(
                    self.canvas.create_text(20, y_pos + 10, anchor="nw", text="Analyzed Words:", font=("Arial", 12, "bold"), fill="#333")
                )
                y_pos += 20
                for word, translation in story["analyzed_words"].items():
                    text = f"{word} -> {translation}"
                    self.canvas_items.append(
                        self.canvas.create_text(20, y_pos + 10, anchor="nw", text=text, font=("Arial", 10), fill="#000000")
                    )
                    y_pos += 20
            if story.get("sentiment"):
                y_pos += 10
                polarity = story["sentiment"]["polarity"]
                subjectivity = story["sentiment"]["subjectivity"]
                sentiment_text = f"Sentiment: {'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'} (Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f})"
                self.canvas_items.append(
                    self.canvas.create_text(20, y_pos + 10, anchor="nw", text=sentiment_text, font=("Arial", 12), fill="#333")
                )
            y_pos += 20

        if playlist:
            self.canvas_items.append(
                self.canvas.create_text(20, y_pos, anchor="nw", text="Recordings:", font=("Arial", 16, "bold"), fill="#333")
            )
            y_pos += 40
            user_name = self.user_id.get() or "Anonymous"
            for i, story in enumerate(playlist, 1):
                if story.get("audio_path") and os.path.exists(story.get('audio_path', '')) and story['audio_path'].endswith(".wav"):
                    box_y = y_pos
                    self.canvas.create_rectangle(10, box_y, 1120, box_y + 60, fill="#e8f4f8", outline="#ccc")
                    link_text = f"{user_name}{i}.mp3"
                    audio_text = self.canvas.create_text(20, box_y + 10, anchor="nw", text=link_text, font=("Arial", 12, "underline"), fill="#4CAF50", tags="clickable")
                    self.canvas.tag_bind("clickable", "<Button-1>", lambda e, f=story.get("audio_path", ""): webbrowser.open(f"file://{f}" if f else ""))
                    self.canvas.tag_bind("clickable", "<Enter>", lambda e: self.canvas.itemconfig(audio_text, fill="#2E7D32"))
                    self.canvas.tag_bind("clickable", "<Leave>", lambda e: self.canvas.itemconfig(audio_text, fill="#4CAF50"))
                    text_button = ttk.Button(self.canvas, text="Show Text", command=lambda s=story: self.show_story_text(s), style="Text.TButton")
                    text_button.place(x=1030, y=box_y + 15)
                    self.canvas_items.append(text_button)
                    y_pos += 80

    def start_recording(self):
        self.stop_event = threading.Event()
        self.record_button.pack_forget()
        self.stop_button.pack(side=tk.RIGHT, anchor=tk.SE, padx=15, pady=15)
        self.update_status("Recording...")
        def task():
            self.stream, self.audio_data, filename = record_audio(self, self.canvas, self.stop_event)
            if self.stream:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                self.stream.stop()
                self.stream.close()
                saved_filename = save_audio(self.audio_data, filename)
                if saved_filename:
                    user_id = self.user_id.get() or "anonymous"
                    self.update_status("Processing audio...")
                    story = process_story(saved_filename, user_id)
                    if story:
                        self.current_story = story  # Store the story
                        self.root.after(0, lambda: self.update_canvas(story=self.current_story))  # Update with stored story
                        self.root.after(0, lambda: self.root.update())  # Force GUI refresh
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to process story. Check transcription."))
            self.root.after(0, lambda: self.stop_button.pack_forget())
            self.root.after(0, lambda: self.record_button.pack(side=tk.LEFT, padx=5))
            self.root.after(0, lambda: self.update_status("Ready"))
        threading.Thread(target=task, daemon=True).start()

    def stop_recording(self):
        if self.stop_event:
            self.stop_event.set()

    def show_playlist(self):
        user_id = self.user_id.get() or "anonymous"
        playlist = create_playlist(user_id)
        self.root.after(0, lambda: self.update_canvas(playlist=playlist))

    def start_voice_command(self):
        def task():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                self.update_status("Listening for command...")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    command = recognizer.recognize_google(audio).lower()
                    if "record" in command:
                        self.root.after(0, self.start_recording)
                    elif "playlist" in command:
                        self.root.after(0, self.show_playlist)
                    else:
                        self.update_status(f"Unknown command: {command}")
                except Exception as e:
                    self.update_status(f"Voice command failed: {e}")
                self.root.after(100, lambda: self.update_status("Ready"))
        threading.Thread(target=task, daemon=True).start()

# Main application
if __name__ == "__main__":
    try:
        import numpy, matplotlib, nltk, textblob
        root = tk.Tk()
        app = StoryWeaverApp(root)
        root.mainloop()
    except ImportError as e:
        messagebox.showerror("Startup Error", f"Missing dependency: {e}")
    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")
