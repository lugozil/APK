
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from random import randint
from midiutil.MidiFile3 import MIDIFile
from pydub import AudioSegment
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from midiutil import MIDIFile 
from plyer import filechooser 
from midi2audio import FluidSynth
# from midiutil import MIDIFile

import sys 
import math
import numpy as np 
import cv2
import numpy as np
# import threading
import os
import subprocess
from kivy.app import App
from kivy.uix.filechooser import FileChooserIconView
import pygame


from jnius import autoclass 

ruta_imagen = ''
panel = ''
staff_files = [ 
    "resources/template/staff2.png", 
    "resources/template/staff.png"]
quarter_files = [
    "resources/template/quarter.png", 
    "resources/template/solid-note.png"]
sharp_files = [
    "resources/template/sharp.png"]
flat_files = [
    "resources/template/flat-line.png", 
    "resources/template/flat-space.png" ]
half_files = [
    "resources/template/half-space.png", 
    "resources/template/half-note-line.png",
    "resources/template/half-line.png", 
    "resources/template/half-note-space.png"]
whole_files = [
    "resources/template/whole-space.png", 
    "resources/template/whole-note-line.png",
    "resources/template/whole-line.png", 
    "resources/template/whole-note-space.png"]

staff_imgs = [cv2.imread(staff_file, 0) for staff_file in staff_files]
quarter_imgs = [cv2.imread(quarter_file, 0) for quarter_file in quarter_files]
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in sharp_files]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in flat_files]
half_imgs = [cv2.imread(half_file, 0) for half_file in half_files]
whole_imgs = [cv2.imread(whole_file, 0) for whole_file in whole_files]

staff_lower, staff_upper, staff_thresh = 50, 150, 0.77
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77
quarter_lower, quarter_upper, quarter_thresh = 50, 150, 0.70
half_lower, half_upper, half_thresh = 50, 150, 0.70
whole_lower, whole_upper, whole_thresh = 50, 150, 0.70


#-------------------------Notas-------------------------- 
note_step = 0.2450

note_defs = {
     -4 : ("g5", 79),
     -3 : ("f5", 77),
     -2 : ("e5", 76),
     -1 : ("d5", 74),
      0 : ("c5", 72),
      1 : ("b4", 71),
      2 : ("a4", 69),
      3 : ("g4", 67),
      4 : ("f4", 65),
      5 : ("e4", 64),
      6 : ("d4", 62),
      7 : ("c4", 60),
      8 : ("b3", 59),
      9 : ("a3", 57),
     10 : ("g3", 55),
     11 : ("f3", 53),
     12 : ("e3", 52),
     13 : ("d3", 50),
     14 : ("c3", 48),
     15 : ("b2", 47),
     16 : ("a2", 45),
     17 : ("f2", 41),           # Cambio, 53 a 41
}

# Obtén el contexto de la aplicación
PythonActivity = autoclass('org.kivy.android.PythonActivity')
context = PythonActivity.mActivity.getApplicationContext()


class Note(object):
    def __init__(self, rec, sym, staff_rec, sharp_notes = [], flat_notes = []):
        self.rec = rec
        self.sym = sym

        middle = rec.y + (rec.h / 2.0)
        height = (middle - staff_rec.y) / staff_rec.h
        note_def = note_defs[int(height/note_step + 0.5)]
        self.note = note_def[0]
        self.pitch = note_def[1]
        if any(n for n in sharp_notes if n.note[0] == self.note[0]):
            self.note += "#"
            self.pitch += 1
        if any(n for n in flat_notes if n.note[0] == self.note[0]):
            self.note += "b"
            self.pitch -= 1

class Rectangle(object):
    def __init__(self, x, y, w, h):
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x));
        overlap_y = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y));
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return Rectangle(x, y, w, h)

    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w), (int)(self.y + self.h))
        cv2.rectangle(img, pos, size, color, thickness)


# Agrega las funciones necesarias como locate_images, merge_recs, Rectangle, Note y otras funciones de procesamiento de imagen aquí
def fit(img, templates, start_percent, stop_percent, threshold):
    best_location_count = -1
    best_locations = []
    best_scale = 1
    x = []
    y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 3)]:
        locations = []
        location_count = 0
        for template in templates:
            template = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            result = np.where(result >= threshold)
            location_count += len(result[0])
            locations += [result]
        x.append(location_count)
        y.append(scale)
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
    return best_locations, best_scale

#encuentra y ubica plantilla con una imagen en base a limites de escala y umbral
def locate_images(img, templates, start, stop, threshold):
    locations, scale = fit(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([Rectangle(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return img_locations

#fusiona y filtra una lista de objetos `recs` basándose en el solapamiento y la distancia
def merge_recs(recs, threshold):
    filtered_recs = []
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while(merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w/2 + recs[i].w/2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)
    return filtered_recs

#redimensiona  
def resize(imagen):
    max_lado = 800 
    ancho, alto = imagen.size
    if ancho > max_lado or alto > max_lado:
        if ancho > alto:
            new_ancho = max_lado
            new_alto = int(alto * (max_lado / ancho))
        else:
            new_alto = max_lado
            new_ancho = int(ancho * (max_lado / alto))
        imagen = imagen.resize((new_ancho, new_alto))
    return imagen

def convert_midi_to_wav(midi_file, wav_file):
    fs = FluidSynth() 
    fs.midi_to_audio(midi_file, wav_file)

# Función para abrir archivos
def open_file(path):
    file_chooser = FileChooserIconView(path=path)
    Window.add_widget(file_chooser)

# Función para reproducir audio usando plyer
def play_audio(file_path):
    sound = SoundLoader.load(file_path)
    if sound:
        sound.play()
    else:
        print("Error al cargar el archivo {file_path}")


# Función para generar un archivo MIDI
def generate_midi(note_groups):
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    volume = 100

    midi.addTrackName(track, time, "Track")
    midi.addTempo(track, time, 140)

    for note_group in note_groups:
        duration = None
        for note in note_group:
            note_type = note.sym
            if note_type == "1":
                duration = 4
            elif note_type == "2":
                duration = 2
            elif note_type == "4,8":
                duration = 1 if len(note_group) == 1 else 0.5
            pitch = note.pitch
            midi.addNote(track, channel, pitch, time, duration, volume)
            time += duration

    midi.addNote(track, channel, pitch, time, 4, 0)
    midi_file = "output.mid"
    with open(midi_file, 'wb') as binfile:
        midi.writeFile(binfile)

    return midi_file

# DEFINE las pantallas
class FirstWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

# KV String para definir la interfaz
kv = '''
WindowManager:
    FirstWindow:
    SecondWindow:

<FirstWindow>:
    name: "first"

    BoxLayout:
        orientation: "vertical"
        size: root.width, root.height
        padding: 10
        spacing: 10

        BoxLayout:
            orientation: "horizontal"
            spacing: 10

            RoundedButtonA:
                text: "Salir"
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                on_release:
                    app.stop()

            RoundedButtonA:
                text: "Cargar Partitura"
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                on_release:
                    app.root.current = "second"
                    root.manager.transition.direction = "left"
                    app.on_button_press()

        BoxLayout:
            orientation: "horizontal"
            spacing: 10

            RoundedButtonA:
                text: "Audioguía"
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                # on_release:
                #     sound = app.load_sound('audio.mp3')
                    

            RoundedButtonB:
                background_normal: 'logo.png'
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla


<SecondWindow>:
    name: "second"

    BoxLayout:
        orientation: "vertical"
        size: root.width, root.height
        padding: 10
        spacing: 10

        BoxLayout:
            orientation: "horizontal"
            spacing: 20

            RoundedButtonA:
                text: "Volver a Menú Inicio"
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                on_release:
                    app.root.current = "first"
                    root.manager.transition.direction = "right"

            RoundedButtonA:
                text: "Volver a Reproducir"
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                on_release:
                    sound = app.load_sound('output.mp3')

        BoxLayout:
            orientation: "horizontal"

            Widget:  # Widget de relleno para centrar el botón
                size_hint_x: None
                width: (self.parent.width - root.width * 0.5 - 20) / 2  # Calcula el ancho del widget de relleno

            RoundedButtonB:
                background_normal: 'logo.png'
                font_size: 32
                size_hint_x: None
                width: root.width * 0.5  # 50% del ancho de la pantalla
                pos_hint: {"center_x": 0.5}

        BoxLayout:
            id: image_box
            orientation: 'vertical'

<RoundedButtonA@Button>
    background_color: (0,0,0,0)
    background_normal: ''
    canvas.before:
        Color:
            rgba: (28/255,80/255,70/255,1)
        RoundedRectangle:
            size: self.size
            pos: self.pos


<RoundedButtonB@Button>
    canvas.before:
        Color:
            rgba: (128/255,187/255,43/255,1)  # Cambia el color del botón a verde
        RoundedRectangle:
            size: self.size
            pos: self.pos
'''

class MyLayout(BoxLayout):
    pass

class UharmonyApp(App):

    def build(self):
        self.root = Builder.load_string(kv)
        return self.root
   
    def load_sound(self, filename):
        # Implementa tu lógica para cargar sonido aquí
        pass

    def on_button_press(self):
        filechooser.open_file(on_selection=self.cargar_imagen,mime_type='image/*')



    def obtener_ruta_interna(img_file):
        # Obtén el directorio de archivos internos
        files_dir = context.getFilesDir().getAbsolutePath()
        return os.path.join(files_dir, img_file)
            
        


    def cargar_imagen(self, selection):
        if selection:
            img_file = selection[0]
            image_box = self.root.ids.get('image_box')

            if image_box:
                image_box.source = img_file
                image_box.reload()

            self.procesar_imagen(img_file) 


    def procesar_imagen(self, img_file):
        img_file_path = self.obtener_ruta_interna(img_file)
        print(f"Procesando imagen: {img_file_path}")
    
        if not os.path.isfile(img_file_path):
            print(f"El archivo no existe: {img_file_path}")
            return
    
        img = cv2.imread(img_file_path, 0)
    
        if img is None:
            print("Error al leer la imagen. La imagen puede estar corrupta o el formato no es compatible.")
        else:
            print("Imagen cargada correctamente")
        # Procesar la imagen
        img_gray = img
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        ret, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        img_width, img_height = img_gray.shape[::-1]

        staff_recs = locate_images(img_gray, staff_imgs, staff_lower, staff_upper, staff_thresh)
        staff_recs = [j for i in staff_recs for j in i]
        heights = [r.y for r in staff_recs] + [0]
        histo = [heights.count(i) for i in range(0, max(heights) + 1)]
        avg = np.mean(list(set(histo)))
        staff_recs = [r for r in staff_recs if histo[r.y] > avg]

        staff_recs = merge_recs(staff_recs, 0.01)
        staff_recs_img = img.copy()
        for r in staff_recs:
            r.draw(staff_recs_img, (0, 0, 255), 2)

        staff_boxes = merge_recs([Rectangle(0, r.y, img_width, r.h) for r in staff_recs], 0.01)
        staff_boxes_img = img.copy()
        for r in staff_boxes:
            r.draw(staff_boxes_img, (0, 0, 255), 2)

        sharp_recs = locate_images(img_gray, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)
        sharp_recs = merge_recs([j for i in sharp_recs for j in i], 0.5)
        sharp_recs_img = img.copy()
        for r in sharp_recs:
            r.draw(sharp_recs_img, (0, 0, 255), 2)

        flat_recs = locate_images(img_gray, flat_imgs, flat_lower, flat_upper, flat_thresh)
        flat_recs = merge_recs([j for i in flat_recs for j in i], 0.5)
        flat_recs_img = img.copy()
        for r in flat_recs:
            r.draw(flat_recs_img, (0, 0, 255), 2)

        quarter_recs = locate_images(img_gray, quarter_imgs, quarter_lower, quarter_upper, quarter_thresh)
        quarter_recs = merge_recs([j for i in quarter_recs for j in i], 0.5)
        quarter_recs_img = img.copy()
        for r in quarter_recs:
            r.draw(quarter_recs_img, (0, 0, 255), 2)

        half_recs = locate_images(img_gray, half_imgs, half_lower, half_upper, half_thresh)
        half_recs = merge_recs([j for i in half_recs for j in i], 0.5)
        half_recs_img = img.copy()
        for r in half_recs:
            r.draw(half_recs_img, (0, 0, 255), 2)

        whole_recs = locate_images(img_gray, whole_imgs, whole_lower, whole_upper, whole_thresh)
        whole_recs = merge_recs([j for i in whole_recs for j in i], 0.5)
        whole_recs_img = img.copy()
        for r in whole_recs:
            r.draw(whole_recs_img, (0, 0, 255), 2)

        note_groups = []
        for box in staff_boxes:
            staff_sharps = [Note(r, "sharp", box) for r in sharp_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            staff_flats = [Note(r, "flat", box) for r in flat_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            quarter_notes = [Note(r, "4,8", box, staff_sharps, staff_flats) for r in quarter_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            half_notes = [Note(r, "2", box, staff_sharps, staff_flats) for r in half_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            whole_notes = [Note(r, "1", box, staff_sharps, staff_flats) for r in whole_recs if abs(r.middle[1] - box.middle[1]) < box.h * 5.0 / 8.0]
            staff_notes = quarter_notes + half_notes + whole_notes
            staff_notes.sort(key=lambda n: n.rec.x)
            staffs = [r for r in staff_recs if r.overlap(box) > 0]
            staffs.sort(key=lambda r: r.x)
            note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            note_group = []
            i = 0
            j = 0
            while i < len(staff_notes):
                if staff_notes[i].rec.x > staffs[j].x and j < len(staffs):
                    r = staffs[j]
                    j += 1
                    if len(note_group) > 0:
                        note_groups.append(note_group)
                        note_group = []
                    note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
                else:
                    note_group.append(staff_notes[i])
                    staff_notes[i].rec.draw(img, note_color, 2)
                    i += 1
            note_groups.append(note_group)

        # Aquí puedes llamar a más funciones según tus necesidades específicas después del procesamiento de la imagen
        # Por ejemplo, guardar resultados, actualizar la interfaz, etc.
        for r in staff_boxes:
            r.draw(img, (0, 0, 255), 2)
        for r in sharp_recs:
            r.draw(img, (0, 0, 255), 2)
        flat_recs_img = img.copy()
        for r in flat_recs:
            r.draw(img, (0, 0, 255), 2)
                    
        # cv2.imwrite('res.png', img)
        # img = Image.open('res.png')
        # img = resize(img)
        # panel.config(image=img)
        # panel.image = img


        # genercion de archivo midi 
        midi = MIDIFile(1)
        track = 0   
        time = 0
        channel = 0
        volume = 100
        
        midi.addTrackName(track, time, "Track")
        midi.addTempo(track, time, 140)
        
        for note_group in note_groups:
            duration = None
            for note in note_group:
                note_type = note.sym
                if note_type == "1":
                    duration = 4
                elif note_type == "2":
                    duration = 2
                elif note_type == "4,8":
                    duration = 1 if len(note_group) == 1 else 0.5
                pitch = note.pitch
                midi.addNote(track,channel,pitch,time,duration,volume)
                time += duration

        midi_file = generate_midi(note_groups)
        wav_file = 'output.wav'
        convert_midi_to_wav(midi_file,wav_file)
        play_audio(wav_file)
        cargado = True


# Main principal
if __name__ == '__main__':
    UharmonyApp().run()