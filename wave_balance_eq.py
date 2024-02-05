import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel
import pyaudio
import numpy as np
import threading

class SoundEqualizer(QWidget):
    def __init__(self):
        super().__init__()

        # Inicjalizacja PyAudio
        print("Inicjalizacja PyAudio")
        self.p = pyaudio.PyAudio()

        # Parametry
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 2
        self.RATE = 44100
        self.CHUNK = 1024

        # Inicjalizacja wzmocnień dla konkretnych pasm
        self.low_gain = 1.0
        self.mid_gain = 1.0
        self.high_gain = 1.0

        # Inicjalizacja przedziałów częstotliwości
        self.low_band = (0, 200)
        self.mid_band = (200, 2000)
        self.high_band = (2000, 10000)

        # Ustawienie BlackHole jako urządzenia wejściowego
        self.input_device_index = 2
        self.output_device_index = 8
        print(f"Ustawienie urządzenia wejściowego na index: {self.input_device_index}")

        # Otwarcie strumienia dźwiękowego
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            input_device_index=self.input_device_index,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.CHUNK
        )
        print("Otwarcie strumienia dźwiękowego")

        # Utworzenie interfejsu graficznego
        self.init_ui()
        print("Inicjalizacja interfejsu graficznego")

        # Ustawienia bufora opóźniającego
        self.delay_buffer = []
        self.delay_size = 5  # Ilość buforowanych ramek
        self.delay_value = 0.2  # Przykładowa wartość opóźnienia (zmień na pożądaną wartość)

        # Uruchomienie osobnego wątku do obsługi strumienia dźwiękowego
        self.gui_thread = threading.Thread(target=self.update_equalizer)
        self.gui_thread.start()
        print("Uruchomienie wątku obsługującego strumień dźwiękowy")

    def init_ui(self):
        # Ustawienia okna
        self.setWindowTitle('Sound Equalizer')
        self.setGeometry(100, 100, 400, 200)

        # Utworzenie suwaków do manipulacji wzmocnień
        self.low_slider = QSlider()
        self.low_slider.setOrientation(1)
        self.low_slider.setMinimum(0)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(int(self.low_gain * 100))
        self.low_slider.valueChanged.connect(self.update_low_gain)

        self.mid_slider = QSlider()
        self.mid_slider.setOrientation(1)
        self.mid_slider.setMinimum(0)
        self.mid_slider.setMaximum(100)
        self.mid_slider.setValue(int(self.mid_gain * 100))
        self.mid_slider.valueChanged.connect(self.update_mid_gain)

        self.high_slider = QSlider()
        self.high_slider.setOrientation(1)
        self.high_slider.setMinimum(0)
        self.high_slider.setMaximum(100)
        self.high_slider.setValue(int(self.high_gain * 100))
        self.high_slider.valueChanged.connect(self.update_high_gain)

        # Dodanie etykiet do suwaków
        self.low_label = QLabel('Low Frequency')
        self.mid_label = QLabel('Mid Frequency')
        self.high_label = QLabel('High Frequency')

        # Utworzenie układu pionowego
        layout = QVBoxLayout()
        layout.addWidget(self.low_label)
        layout.addWidget(self.low_slider)
        layout.addWidget(self.mid_label)
        layout.addWidget(self.mid_slider)
        layout.addWidget(self.high_label)
        layout.addWidget(self.high_slider)

        # Ustawienie układu na główne okno
        self.setLayout(layout)

        # Wyświetlenie okna
        self.show()

    def update_low_gain(self):
        self.low_gain = self.low_slider.value() / 100.0
        print(f"Aktualne wzmocnienie dla niskich częstotliwości: {self.low_gain}")

    def update_mid_gain(self):
        self.mid_gain = self.mid_slider.value() / 100.0
        print(f"Aktualne wzmocnienie dla średnich częstotliwości: {self.mid_gain}")

    def update_high_gain(self):
        self.high_gain = self.high_slider.value() / 100.0
        print(f"Aktualne wzmocnienie dla wysokich częstotliwości: {self.high_gain}")

    def delay(self, data):
        # Dodanie danych do bufora opóźniającego
        self.delay_buffer.append(data)

        # Sprawdzenie, czy bufor osiągnął zadaną wielkość
        if len(self.delay_buffer) > self.delay_size:
            delayed_data = self.delay_buffer.pop(0) * (1.0 - self.delay_value)

            # Zastosowanie opóźnionych danych
            return np.fft.ifft(delayed_data).astype(np.float32)

        return data

    def update_equalizer(self):
        while True:
            # Odczyt danych audio ze strumienia
            data = self.stream.read(self.CHUNK)
            audio_array = np.frombuffer(data, dtype=np.float32)

            # Zastosowanie wzmocnień korektora do pasm częstotliwości
            fft_data = np.fft.fft(audio_array)
            fft_data[:self.low_band[0]] *= self.low_gain
            fft_data[self.low_band[0]:self.low_band[1]] *= self.low_gain
            fft_data[self.mid_band[0]:self.mid_band[1]] *= self.mid_gain
            fft_data[self.high_band[0]:self.high_band[1]] *= self.high_gain
            fft_data[self.high_band[1]:] *= self.high_gain

            # Zastosowanie opóźnienia
            processed_data = self.delay(fft_data)

            # Odwrotna FFT w celu uzyskania sygnału w dziedzinie czasu
            equalized_data = np.fft.ifft(processed_data).astype(np.float32)

            # Odtwarzanie zrównoważonego dźwięku
            self.stream.write(equalized_data.tobytes())

    def closeEvent(self, event):
        # Zatrzymanie strumienia i zamknięcie aplikacji
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        event.accept()
        print("Zamknięcie aplikacji")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundEqualizer()
    sys.exit(app.exec_())
